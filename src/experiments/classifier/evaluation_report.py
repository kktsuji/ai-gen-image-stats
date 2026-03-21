"""Evaluation Report Generator

Aggregates evaluation.json files from all classifier experiments
into comparison tables for analyzing synthetic augmentation effectiveness.

Supports both single-seed (legacy) and multi-seed layouts:
  - Single-seed: {base_dir}/{experiment}/reports/evaluation.json
  - Multi-seed:  {base_dir}/{experiment}/seed{N}/reports/evaluation.json

When multi-seed results are detected, the report includes:
  - Mean +/- std across seeds for each metric
  - Statistical significance table (paired t-test + Cohen's d)

Usage:
    python -m src.experiments.classifier.evaluation_report [--output-dir outputs/evaluation_report]
"""

import json
import logging
import math
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.statistical_testing import compare_experiment_pair

_logger = logging.getLogger(__name__)

# Key metrics for comparison (ordered by importance for imbalanced classification)
KEY_METRICS = [
    "recall_1",  # Minority class recall (abnormal detection)
    "balanced_accuracy",
    "f1_1",  # Minority class F1
    "pr_auc",
    "roc_auc",
    "accuracy",
    "precision_1",
    "loss",
]


def _parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """Parse experiment name into dimension components.

    Dimensions are separated by "__" (double underscore). Hyphens and single
    underscores may appear within a dimension value (e.g., "n100-gs3").
    See scripts/run_pipeline.py for the naming convention.

    Expected formats:
        - Synthetic: {train}__{gen}__{sel}__{cls}  e.g. "ws__n100-gs3__topk__all"
        - Baseline:  baseline__{strategy}          e.g. "baseline__vanilla"

    Returns:
        Dictionary with keys: type, diffusion_variant, gen_config, selection, aug_limit
    """
    parts = exp_name.split("__")

    if parts[0] == "baseline" and len(parts) == 2:
        return {
            "type": "baseline",
            "diffusion_variant": "-",
            "gen_config": "-",
            "selection": "-",
            "aug_limit": "-",
            "baseline_strategy": parts[1],
        }

    if len(parts) == 4:
        return {
            "type": "synthetic",
            "diffusion_variant": parts[0],
            "gen_config": parts[1],
            "selection": parts[2],
            "aug_limit": parts[3],
            "baseline_strategy": "-",
        }

    return {
        "type": "unknown",
        "diffusion_variant": exp_name,
        "gen_config": "-",
        "selection": "-",
        "aug_limit": "-",
        "baseline_strategy": "-",
    }


def load_evaluation_results(
    base_dir: str = "outputs/classifier",
) -> List[Dict[str, Any]]:
    """Scan for evaluation.json files and load all results.

    Supports both single-seed and multi-seed directory layouts.
    For multi-seed, each result includes a "seed" field.

    Args:
        base_dir: Base directory containing classifier experiment outputs.

    Returns:
        List of dictionaries, each containing experiment name + metrics.
    """
    results: List[Dict[str, Any]] = []

    # Try multi-seed pattern first
    multi_seed_pattern = f"{base_dir}/*/seed*/reports/evaluation.json"
    multi_seed_paths = sorted(glob(multi_seed_pattern))

    if multi_seed_paths:
        for json_path in multi_seed_paths:
            path = Path(json_path)
            # path: base_dir/exp_name/seed{N}/reports/evaluation.json
            seed_dir = path.parent.parent.name  # "seed0", "seed1", etc.
            exp_name = path.parent.parent.parent.name

            try:
                seed_num = int(seed_dir.replace("seed", ""))
            except ValueError:
                _logger.warning(f"Cannot parse seed from {seed_dir}, skipping")
                continue

            entry = _load_single_result(json_path, exp_name)
            if entry is not None:
                entry["seed"] = seed_num
                results.append(entry)

    # Also load single-seed results (backward compatibility)
    single_seed_pattern = f"{base_dir}/*/reports/evaluation.json"
    for json_path in sorted(glob(single_seed_pattern)):
        path = Path(json_path)
        exp_name = path.parent.parent.name
        # Skip if this experiment already has multi-seed results
        if any(r["experiment"] == exp_name for r in results):
            continue

        entry = _load_single_result(json_path, exp_name)
        if entry is not None:
            results.append(entry)

    return results


def _load_single_result(json_path: str, exp_name: str) -> Optional[Dict[str, Any]]:
    """Load a single evaluation.json file.

    Args:
        json_path: Path to the evaluation.json file.
        exp_name: Experiment name.

    Returns:
        Dictionary with experiment metadata + metrics, or None on error.
    """
    try:
        with open(json_path) as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _logger.warning(f"Skipping malformed file {json_path}: {e}")
        return None
    if not isinstance(metrics, dict):
        _logger.warning(
            "Skipping malformed file %s: expected a JSON object, got %s",
            json_path,
            type(metrics).__name__,
        )
        return None

    entry: Dict[str, Any] = {"experiment": exp_name}
    entry.update(_parse_experiment_name(exp_name))

    # Prevent evaluation.json keys from overwriting metadata columns
    reserved_keys = set(entry.keys())
    conflicts = reserved_keys & metrics.keys()
    if conflicts:
        _logger.warning(f"Skipping reserved keys in {json_path}: {conflicts}")
        metrics = {k: v for k, v in metrics.items() if k not in conflicts}

    entry.update(metrics)
    return entry


def build_comparison_dataframe(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Build a pandas DataFrame from evaluation results.

    Args:
        results: List of experiment result dictionaries.

    Returns:
        DataFrame with one row per experiment.
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def aggregate_multi_seed(
    df: pd.DataFrame,
    metric_names: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate multi-seed results by experiment name.

    Groups rows by experiment name and collects metric values across seeds.

    Args:
        df: DataFrame with "experiment", "seed", and metric columns.
        metric_names: List of metric column names to aggregate.

    Returns:
        Dict mapping experiment_name -> {metric_name -> np.ndarray of seed values}.
        Only includes experiments with >= 2 seeds.
    """
    if "seed" not in df.columns:
        return {}

    aggregated: Dict[str, Dict[str, np.ndarray]] = {}

    for exp_name, group in df.groupby("experiment"):
        if len(group) < 2:
            continue

        metric_arrays: Dict[str, np.ndarray] = {}
        for metric in metric_names:
            if metric in group.columns:
                values = group[metric].dropna().values
                if len(values) >= 2:
                    metric_arrays[metric] = np.array(values, dtype=np.float64)

        if metric_arrays:
            aggregated[str(exp_name)] = metric_arrays

    return aggregated


def build_mean_std_dataframe(
    df: pd.DataFrame,
    metric_names: List[str],
) -> pd.DataFrame:
    """Build a DataFrame with mean +/- std for multi-seed experiments.

    For experiments with multiple seeds, computes mean and std.
    For single-seed experiments, uses the original values.

    Args:
        df: DataFrame with evaluation results (may include "seed" column).
        metric_names: Metrics to include.

    Returns:
        DataFrame with one row per unique experiment, metrics as mean values,
        and {metric}_std columns for multi-seed experiments.
    """
    if "seed" not in df.columns:
        return df

    rows: List[Dict[str, Any]] = []
    meta_cols = [
        "experiment",
        "type",
        "diffusion_variant",
        "gen_config",
        "selection",
        "aug_limit",
        "baseline_strategy",
    ]

    for _, group in df.groupby("experiment"):
        row: Dict[str, Any] = {}
        # Copy metadata from first entry
        for col in meta_cols:
            if col in group.columns:
                row[col] = group.iloc[0][col]

        row["n_seeds"] = len(group)

        for metric in metric_names:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    row[metric] = float(values.mean())
                    if len(values) > 1:
                        row[f"{metric}_std"] = float(values.std(ddof=1))

                # Preserve CI columns if present (from bootstrap)
                for suffix in ["_ci_lower", "_ci_upper"]:
                    ci_col = f"{metric}{suffix}"
                    if ci_col in group.columns:
                        ci_values = group[ci_col].dropna()
                        if len(ci_values) > 0:
                            row[ci_col] = float(ci_values.mean())

        rows.append(row)

    return pd.DataFrame(rows)


def generate_statistical_comparison_table(
    df: pd.DataFrame,
    alpha: float = 0.05,
    correction_method: str = "benjamini-hochberg",
    baseline_name: Optional[str] = None,
) -> str:
    """Generate table of paired t-test results comparing baselines to synthetics.

    For each baseline, compares against all synthetic experiments across key
    metrics using paired t-test (paired by seed) and Cohen's d effect size.

    Args:
        df: DataFrame with multi-seed evaluation results (must have "seed" column).
        alpha: Significance threshold after correction.
        correction_method: P-value correction method.
        baseline_name: Specific baseline to use. If None, uses the best
            baseline (highest recall_1 mean).

    Returns:
        Markdown-formatted table string, or empty string if not applicable.
    """
    if "seed" not in df.columns:
        return ""

    metric_names = [m for m in KEY_METRICS if m in df.columns]
    if not metric_names:
        return ""

    aggregated = aggregate_multi_seed(df, metric_names)
    if not aggregated:
        return ""

    # Identify baselines and synthetics
    baselines = {
        name: vals
        for name, vals in aggregated.items()
        if _parse_experiment_name(name)["type"] == "baseline"
    }
    synthetics = {
        name: vals
        for name, vals in aggregated.items()
        if _parse_experiment_name(name)["type"] == "synthetic"
    }

    if not baselines or not synthetics:
        return ""

    # Select which baseline to compare against
    if baseline_name and baseline_name in baselines:
        selected_baseline = baseline_name
    else:
        # Use baseline with highest mean recall_1 (or first available)
        best_bl_name = None
        best_bl_recall = -float("inf")
        for bl_name, bl_vals in baselines.items():
            if "recall_1" in bl_vals:
                mean_recall = float(np.mean(bl_vals["recall_1"]))
                if mean_recall > best_bl_recall:
                    best_bl_recall = mean_recall
                    best_bl_name = bl_name
        selected_baseline = best_bl_name or next(iter(baselines))

    bl_values = baselines[selected_baseline]

    # Collect all comparison results across all synthetics
    all_results: List[tuple] = []  # (synthetic_name, ComparisonResult)
    for syn_name, syn_vals in sorted(synthetics.items()):
        # Find common metrics with matching seed counts
        common_metrics = [
            m
            for m in metric_names
            if m in bl_values
            and m in syn_vals
            and len(bl_values[m]) == len(syn_vals[m])
        ]
        if not common_metrics:
            continue

        comparisons = compare_experiment_pair(
            bl_values,
            syn_vals,
            common_metrics,
            alpha=alpha,
            correction_method=correction_method,
        )
        for result in comparisons:
            all_results.append((syn_name, result))

    if not all_results:
        return ""

    # Build table
    rows: List[Dict[str, Any]] = []
    sig_count = 0
    for syn_name, r in all_results:
        sig_marker = "*" if r.significant else ""
        rows.append(
            {
                "experiment": syn_name,
                "metric": r.metric,
                "baseline": f"{r.baseline_mean:.4f} +/- {r.baseline_std:.4f}",
                "synthetic": f"{r.treatment_mean:.4f} +/- {r.treatment_std:.4f}",
                "diff": f"{r.mean_diff:+.4f}",
                "cohens_d": (
                    f"{r.cohens_d:.3f} ({r.effect_size_interpretation})"
                    if math.isfinite(r.cohens_d)
                    else "N/A"
                ),
                "p_corrected": (
                    f"{r.p_value_corrected:.4f}{sig_marker}"
                    if math.isfinite(r.p_value_corrected)
                    else "N/A"
                ),
            }
        )
        if r.significant:
            sig_count += 1

    table_df = pd.DataFrame(rows)
    table_str: Optional[str] = table_df.to_markdown(index=False, disable_numparse=True)

    total = len(all_results)
    header_lines = [
        f"Baseline: **{selected_baseline}**",
        f"Correction: {correction_method}, alpha={alpha}",
        f"Significant: {sig_count}/{total} comparisons",
        "",
    ]

    return "\n".join(header_lines) + (table_str or "")


def _format_value_with_ci(
    val: float,
    lo: Optional[float],
    hi: Optional[float],
    floatfmt: str = ".4f",
) -> str:
    """Format a single metric value with optional CI bounds.

    Args:
        val: Metric value.
        lo: CI lower bound (None or NaN to omit).
        hi: CI upper bound (None or NaN to omit).
        floatfmt: Float format string.

    Returns:
        Formatted string, e.g. "0.8500 [0.8200, 0.8800]" or "0.8500".
    """
    val_str = f"{float(val):{floatfmt}}"
    if lo is not None and hi is not None and bool(pd.notna(lo)) and bool(pd.notna(hi)):
        return f"{val_str} [{float(lo):{floatfmt}}, {float(hi):{floatfmt}}]"
    return val_str


def _format_with_ci(df: pd.DataFrame, metric: str, floatfmt: str = ".4f") -> pd.Series:
    """Format a metric column with CI bounds if available.

    If {metric}_ci_lower and {metric}_ci_upper exist in df, formats cells as
    "0.8500 [0.82, 0.88]". Otherwise returns the plain formatted values.

    Args:
        df: DataFrame with evaluation results.
        metric: Metric column name.
        floatfmt: Float format string.

    Returns:
        Series of formatted strings.
    """
    lower_col = f"{metric}_ci_lower"
    upper_col = f"{metric}_ci_upper"
    has_ci = lower_col in df.columns and upper_col in df.columns

    formatted = []
    for idx in df.index:
        val = df.loc[idx, metric]
        if pd.notna(val):
            lo = df.loc[idx, lower_col] if has_ci else None
            hi = df.loc[idx, upper_col] if has_ci else None
            formatted.append(_format_value_with_ci(val, lo, hi, floatfmt))
        else:
            formatted.append("")
    return pd.Series(formatted, index=df.index)


def _format_mean_std(df: pd.DataFrame, metric: str, floatfmt: str = ".4f") -> pd.Series:
    """Format a metric column as mean +/- std if std column exists.

    Args:
        df: DataFrame with evaluation results.
        metric: Metric column name.
        floatfmt: Float format string.

    Returns:
        Series of formatted strings.
    """
    std_col = f"{metric}_std"
    has_std = std_col in df.columns

    formatted = []
    for idx in df.index:
        val = df.loc[idx, metric]
        if pd.notna(val):
            val_str = f"{float(val):{floatfmt}}"
            if has_std:
                std_val = df.loc[idx, std_col]
                if pd.notna(std_val):
                    val_str = f"{val_str} +/- {float(std_val):{floatfmt}}"
            formatted.append(val_str)
        else:
            formatted.append("")
    return pd.Series(formatted, index=df.index)


def generate_classifier_table(df: pd.DataFrame) -> str:
    """Generate markdown table of classifier performance.

    For multi-seed results, shows mean +/- std across seeds.

    Args:
        df: DataFrame with evaluation results.

    Returns:
        Markdown-formatted table string.
    """
    if df.empty:
        return "No evaluation results found.\n"

    display_cols = ["experiment", "type"]
    metric_cols = [m for m in KEY_METRICS if m in df.columns]
    cols = display_cols + metric_cols

    # Add n_seeds column if present
    if "n_seeds" in df.columns:
        display_cols = ["experiment", "type", "n_seeds"]
        cols = display_cols + metric_cols

    available = [c for c in cols if c in df.columns]
    subset = df[available].copy()

    # Sort by minority recall descending
    sort_col = "recall_1" if "recall_1" in subset.columns else "balanced_accuracy"
    if sort_col in subset.columns:
        subset = subset.sort_values(by=sort_col, ascending=False)  # type: ignore[call-overload]

    # Format metric columns
    has_std = any(f"{m}_std" in df.columns for m in metric_cols)

    for metric in metric_cols:
        if metric in subset.columns:
            if has_std:
                subset[metric] = _format_mean_std(df.loc[subset.index], metric).values
            else:
                subset[metric] = _format_with_ci(df.loc[subset.index], metric).values

    # disable_numparse prevents tabulate from re-parsing formatted strings as numbers
    result: Optional[str] = subset.to_markdown(index=False, disable_numparse=True)
    return result if result is not None else ""


def generate_best_per_metric(df: pd.DataFrame) -> str:
    """Generate table showing best experiment per metric.

    Args:
        df: DataFrame with evaluation results.

    Returns:
        Markdown-formatted table string.
    """
    if df.empty:
        return "No evaluation results found.\n"

    rows = []
    for metric in KEY_METRICS:
        if metric not in df.columns:
            continue

        if metric == "loss":
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()

        if best_idx is not None and bool(pd.notna(best_idx)):
            best_row = df.loc[best_idx]
            value = best_row[metric]
            lo = best_row.get(f"{metric}_ci_lower")
            hi = best_row.get(f"{metric}_ci_upper")
            value_str = _format_value_with_ci(value, lo, hi)

            rows.append(
                {
                    "metric": metric,
                    "best_experiment": best_row["experiment"],
                    "value": value_str,
                    "type": best_row.get("type", "unknown"),
                }
            )

    result: Optional[str] = pd.DataFrame(rows).to_markdown(
        index=False,
        disable_numparse=True,
    )
    return result if result is not None else ""


def generate_report(
    base_dir: str = "outputs/classifier",
    output_dir: str = "outputs/evaluation_report",
    selection_summary_pattern: Optional[str] = None,
    alpha: float = 0.05,
    correction_method: str = "benjamini-hochberg",
) -> None:
    """Generate full evaluation report.

    Args:
        base_dir: Base directory for classifier experiments.
        output_dir: Output directory for report files.
        selection_summary_pattern: Optional glob pattern for selection summary JSONs.
        alpha: Significance threshold for statistical testing.
        correction_method: P-value correction method.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_evaluation_results(base_dir)
    _logger.info(f"Found {len(results)} evaluation results")

    if not results:
        _logger.warning("No evaluation results found. Run evaluations first.")
        return

    df = build_comparison_dataframe(results)

    # If multi-seed, aggregate to mean +/- std for tables 2 and 3
    is_multi_seed = "seed" in df.columns
    display_df = build_mean_std_dataframe(df, KEY_METRICS) if is_multi_seed else df

    # Build report
    n_experiments = len(display_df)
    n_baselines = len(display_df[display_df["type"] == "baseline"])
    n_synthetic = len(display_df[display_df["type"] == "synthetic"])

    report_lines = [
        "# Evaluation Report: Synthetic Augmentation Effectiveness",
        "",
        f"Total experiments: {n_experiments}",
        f"Baselines: {n_baselines}",
        f"Synthetic augmentation: {n_synthetic}",
    ]
    if is_multi_seed:
        seed_counts = df.groupby("experiment")["seed"].nunique()
        report_lines.append(
            f"Seeds per experiment: {int(seed_counts.min())}-{int(seed_counts.max())}"
        )
    report_lines.append("")

    # Table 1: Generation quality (from selection summaries)
    if selection_summary_pattern:
        report_lines.append("## Table 1: Generation Quality")
        report_lines.append("")
        gen_results = []
        for json_path in sorted(glob(selection_summary_pattern)):
            path = Path(json_path)
            try:
                with open(json_path) as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                _logger.warning(f"Skipping malformed file {json_path}: {e}")
                continue
            if not isinstance(summary, dict):
                _logger.warning(
                    "Skipping malformed file %s: expected a JSON object, got %s",
                    json_path,
                    type(summary).__name__,
                )
                continue

            entry: Dict[str, Any] = {"path": str(path)}
            reserved_keys = set(entry.keys())
            conflicts = reserved_keys & summary.keys()
            if conflicts:
                _logger.warning(f"Skipping reserved keys in {json_path}: {conflicts}")
                summary = {k: v for k, v in summary.items() if k not in conflicts}
            entry.update(summary)
            gen_results.append(entry)

        if gen_results:
            gen_df = pd.DataFrame(gen_results)
            gen_table: Optional[str] = gen_df.to_markdown(index=False, floatfmt=".4f")
            report_lines.append(gen_table or "")
        else:
            report_lines.append("No generation quality data found.")
        report_lines.append("")

    # Table 2: Classifier performance
    report_lines.append("## Table 2: Classifier Performance")
    report_lines.append("")
    report_lines.append(generate_classifier_table(display_df))
    report_lines.append("")

    # Table 3: Best config per metric
    report_lines.append("## Table 3: Best Configuration per Metric")
    report_lines.append("")
    report_lines.append(generate_best_per_metric(display_df))
    report_lines.append("")

    # Key comparisons
    baselines = display_df[display_df["type"] == "baseline"]
    synthetics = display_df[display_df["type"] == "synthetic"]

    if not baselines.empty and not synthetics.empty:
        report_lines.append("## Key Comparisons")
        report_lines.append("")

        # Best baseline vs best synthetic
        for metric in ["recall_1", "balanced_accuracy", "f1_1", "pr_auc"]:
            if metric not in display_df.columns:
                continue

            bl_best_idx = baselines[metric].idxmax()  # type: ignore[union-attr]
            syn_best_idx = synthetics[metric].idxmax()  # type: ignore[union-attr]
            # Skip if either index is NaN (all-NaN column)
            if bool(pd.isna(bl_best_idx)) or bool(pd.isna(syn_best_idx)):
                continue

            best_baseline_val = float(baselines.loc[bl_best_idx, metric])
            best_synthetic_val = float(synthetics.loc[syn_best_idx, metric])
            best_baseline_name = baselines.loc[bl_best_idx, "experiment"]
            best_synthetic_name = synthetics.loc[syn_best_idx, "experiment"]
            delta = best_synthetic_val - best_baseline_val
            sign = "+" if delta >= 0 else ""

            # Format with CI if available
            lower_col = f"{metric}_ci_lower"
            upper_col = f"{metric}_ci_upper"
            bl_lo = (
                baselines.loc[bl_best_idx, lower_col]
                if lower_col in display_df.columns
                else None
            )
            bl_hi = (
                baselines.loc[bl_best_idx, upper_col]
                if upper_col in display_df.columns
                else None
            )
            syn_lo = (
                synthetics.loc[syn_best_idx, lower_col]
                if lower_col in display_df.columns
                else None
            )
            syn_hi = (
                synthetics.loc[syn_best_idx, upper_col]
                if upper_col in display_df.columns
                else None
            )

            bl_str = _format_value_with_ci(best_baseline_val, bl_lo, bl_hi)
            syn_str = _format_value_with_ci(best_synthetic_val, syn_lo, syn_hi)

            report_lines.append(
                f"- **{metric}**: best baseline={bl_str} "
                f"({best_baseline_name}), "
                f"best synthetic={syn_str} "
                f"({best_synthetic_name}), delta={sign}{delta:.4f}"
            )

        report_lines.append("")

    # Table 4: Statistical significance (multi-seed only)
    if is_multi_seed:
        stat_table = generate_statistical_comparison_table(
            df, alpha=alpha, correction_method=correction_method
        )
        if stat_table:
            report_lines.append(
                "## Table 4: Statistical Significance (Paired t-test + Cohen's d)"
            )
            report_lines.append("")
            report_lines.append(stat_table)
            report_lines.append("")

    # Write report
    report_text = "\n".join(report_lines)

    # Save markdown
    report_md_path = output_path / "evaluation_report.md"
    with open(report_md_path, "w") as f:
        f.write(report_text)
    _logger.info(f"Report saved to: {report_md_path}")

    # Save CSV
    csv_path = output_path / "evaluation_results.csv"
    display_df.to_csv(csv_path, index=False)
    _logger.info(f"CSV saved to: {csv_path}")

    # Log summary
    _logger.info("Report summary:\n%s", report_text)


def main() -> None:
    """CLI entry point for evaluation report generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--base-dir",
        default="outputs/classifier",
        help="Base directory for classifier experiments",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation_report",
        help="Output directory for report files",
    )
    parser.add_argument(
        "--selection-summary-pattern",
        default=None,
        help="Glob pattern for selection summary JSONs",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for statistical testing (default: 0.05)",
    )
    parser.add_argument(
        "--correction-method",
        default="benjamini-hochberg",
        choices=["benjamini-hochberg", "bonferroni"],
        help="P-value correction method (default: benjamini-hochberg)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_report(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        selection_summary_pattern=args.selection_summary_pattern,
        alpha=args.alpha,
        correction_method=args.correction_method,
    )


if __name__ == "__main__":
    main()
