"""Evaluation Report Generator

Aggregates evaluation.json files from all classifier experiments
into comparison tables for analyzing synthetic augmentation effectiveness.

Usage:
    python -m src.experiments.classifier.evaluation_report [--output-dir outputs/evaluation_report]
"""

import json
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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

    Args:
        base_dir: Base directory containing classifier experiment outputs.

    Returns:
        List of dictionaries, each containing experiment name + metrics.
    """
    pattern = f"{base_dir}/*/reports/evaluation.json"
    results = []

    for json_path in sorted(glob(pattern)):
        path = Path(json_path)
        exp_name = path.parent.parent.name

        try:
            with open(json_path) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            _logger.warning(f"Skipping malformed file {json_path}: {e}")
            continue
        if not isinstance(metrics, dict):
            _logger.warning(
                "Skipping malformed file %s: expected a JSON object, got %s",
                json_path,
                type(metrics).__name__,
            )
            continue

        entry: Dict[str, Any] = {"experiment": exp_name}
        entry.update(_parse_experiment_name(exp_name))

        # Prevent evaluation.json keys from overwriting metadata columns
        reserved_keys = set(entry.keys())
        conflicts = reserved_keys & metrics.keys()
        if conflicts:
            _logger.warning(f"Skipping reserved keys in {json_path}: {conflicts}")
            metrics = {k: v for k, v in metrics.items() if k not in conflicts}

        entry.update(metrics)
        results.append(entry)

    return results


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

    if lower_col in df.columns and upper_col in df.columns:
        formatted = []
        for idx in df.index:
            val = df.loc[idx, metric]
            lo = df.loc[idx, lower_col]
            hi = df.loc[idx, upper_col]
            if pd.notna(val):
                val_str = f"{float(val):{floatfmt}}"
                if bool(pd.notna(lo)) and bool(pd.notna(hi)):
                    formatted.append(
                        f"{val_str} [{float(lo):{floatfmt}}, {float(hi):{floatfmt}}]"
                    )
                else:
                    formatted.append(val_str)
            else:
                formatted.append("")
        return pd.Series(formatted, index=df.index)
    else:
        return pd.Series(
            [f"{float(v):{floatfmt}}" if pd.notna(v) else "" for v in df[metric]],
            index=df.index,
        )


def generate_classifier_table(df: pd.DataFrame) -> str:
    """Generate markdown table of classifier performance.

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

    available = [c for c in cols if c in df.columns]
    subset = df[available].copy()

    # Sort by minority recall descending
    sort_col = "recall_1" if "recall_1" in subset.columns else "balanced_accuracy"
    if sort_col in subset.columns:
        subset = subset.sort_values(by=sort_col, ascending=False)  # type: ignore[call-overload]

    # Format metric columns with CI bounds (convert to object dtype to keep strings)
    for metric in metric_cols:
        if metric in subset.columns:
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
            value_str = f"{value:.4f}"

            lower_col = f"{metric}_ci_lower"
            upper_col = f"{metric}_ci_upper"
            if (
                lower_col in df.columns
                and upper_col in df.columns
                and bool(pd.notna(best_row.get(lower_col)))
                and bool(pd.notna(best_row.get(upper_col)))
            ):
                value_str = (
                    f"{value:.4f} [{best_row[lower_col]:.4f}, "
                    f"{best_row[upper_col]:.4f}]"
                )

            rows.append(
                {
                    "metric": metric,
                    "best_experiment": best_row["experiment"],
                    "value": value_str,
                    "type": best_row.get("type", "unknown"),
                }
            )

    result: Optional[str] = pd.DataFrame(rows).to_markdown(index=False)
    return result if result is not None else ""


def generate_report(
    base_dir: str = "outputs/classifier",
    output_dir: str = "outputs/evaluation_report",
    selection_summary_pattern: Optional[str] = None,
) -> None:
    """Generate full evaluation report.

    Args:
        base_dir: Base directory for classifier experiments.
        output_dir: Output directory for report files.
        selection_summary_pattern: Optional glob pattern for selection summary JSONs.
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

    # Build report
    report_lines = [
        "# Evaluation Report: Synthetic Augmentation Effectiveness",
        "",
        f"Total experiments: {len(df)}",
        f"Baselines: {len(df[df['type'] == 'baseline'])}",
        f"Synthetic augmentation: {len(df[df['type'] == 'synthetic'])}",
        "",
    ]

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
    report_lines.append(generate_classifier_table(df))
    report_lines.append("")

    # Table 3: Best config per metric
    report_lines.append("## Table 3: Best Configuration per Metric")
    report_lines.append("")
    report_lines.append(generate_best_per_metric(df))
    report_lines.append("")

    # Key comparisons
    baselines = df[df["type"] == "baseline"]
    synthetics = df[df["type"] == "synthetic"]

    if not baselines.empty and not synthetics.empty:
        report_lines.append("## Key Comparisons")
        report_lines.append("")

        # Best baseline vs best synthetic
        for metric in ["recall_1", "balanced_accuracy", "f1_1", "pr_auc"]:
            if metric not in df.columns:
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
            bl_ci = ""
            syn_ci = ""
            if lower_col in df.columns and upper_col in df.columns:
                bl_lo = baselines.loc[bl_best_idx, lower_col]
                bl_hi = baselines.loc[bl_best_idx, upper_col]
                if bool(pd.notna(bl_lo)) and bool(pd.notna(bl_hi)):
                    bl_ci = f" [{float(bl_lo):.4f}, {float(bl_hi):.4f}]"
                syn_lo = synthetics.loc[syn_best_idx, lower_col]
                syn_hi = synthetics.loc[syn_best_idx, upper_col]
                if bool(pd.notna(syn_lo)) and bool(pd.notna(syn_hi)):
                    syn_ci = f" [{float(syn_lo):.4f}, {float(syn_hi):.4f}]"

            report_lines.append(
                f"- **{metric}**: best baseline={best_baseline_val:.4f}{bl_ci} "
                f"({best_baseline_name}), "
                f"best synthetic={best_synthetic_val:.4f}{syn_ci} "
                f"({best_synthetic_name}), delta={sign}{delta:.4f}"
            )

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
    df.to_csv(csv_path, index=False)
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_report(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        selection_summary_pattern=args.selection_summary_pattern,
    )


if __name__ == "__main__":
    main()
