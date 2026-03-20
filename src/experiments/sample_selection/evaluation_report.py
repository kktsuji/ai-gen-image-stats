"""Selection Evaluation Report Generator

Aggregates evaluation.json files from all selection-eval experiments
into comparison tables for analyzing generation and selection quality
across diffusion variants, generation configs, and selection methods.

Usage:
    python -m src.experiments.sample_selection.evaluation_report [--output-dir outputs/evaluation_report]
"""

import json
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_logger = logging.getLogger(__name__)

# Key metrics for comparison (ordered by importance)
KEY_METRICS = [
    "rvs_fid",
    "rvs_precision",
    "rvs_recall",
    "rvg_fid",
    "rvg_precision",
    "rvg_recall",
]

# Metrics where lower is better (FID); all others are higher-is-better.
# gvs_fid is excluded from KEY_METRICS (less important than real-vs-* metrics)
# but included here for correct direction handling if used in downstream analysis.
LOWER_IS_BETTER = {"rvs_fid", "rvg_fid", "gvs_fid"}

# Prefix mapping for comparison pair flattening
_COMPARISON_PREFIXES = {
    "real_vs_selected": "rvs",
    "real_vs_generated": "rvg",
    "generated_vs_selected": "gvs",
}

# Known selection method tokens (selection names must not contain underscores)
_KNOWN_SELECTIONS = {"topk", "percentile", "threshold"}


def _parse_selection_eval_path(json_path: str) -> Dict[str, str]:
    """Parse selection-eval path into dimension components.

    Expected path structure:
        outputs/diffusion-{train}/selection-eval/{gen}_{sel}/reports/evaluation.json

    The combo directory name uses underscore as the separator between gen_config
    and selection. Since gen_config may contain underscores, we split from the
    right (rsplit) to keep selection as the last token.

    Returns:
        Dictionary with keys: diffusion_variant, gen_config, selection
    """
    path = Path(json_path)

    # Path parts: .../diffusion-{train}/selection-eval/{gen}_{sel}/reports/evaluation.json
    try:
        reports_dir = path.parent  # reports/
        combo_dir = reports_dir.parent  # {gen}_{sel}/
        selection_eval_dir = combo_dir.parent  # selection-eval/
        diffusion_dir = selection_eval_dir.parent  # diffusion-{train}/

        # Extract diffusion variant from "diffusion-{train}"
        diffusion_name = diffusion_dir.name
        if diffusion_name.startswith("diffusion-"):
            diffusion_variant = diffusion_name[len("diffusion-") :] or "-"
        else:
            diffusion_variant = diffusion_name or "-"

        # Extract gen and sel from "{gen}_{sel}"
        # rsplit from right: selection is always a simple token (topk, percentile,
        # threshold), while gen_config may contain underscores.
        combo = combo_dir.name
        parts = combo.rsplit("_", 1)
        if len(parts) == 2:
            gen_config = parts[0]
            selection = parts[1]
        else:
            gen_config = combo
            selection = "-"

        if selection not in _KNOWN_SELECTIONS and selection != "-":
            _logger.warning("Unexpected selection token %r in %s", selection, json_path)

        return {
            "diffusion_variant": diffusion_variant,
            "gen_config": gen_config,
            "selection": selection,
        }
    except (
        AttributeError
    ):  # Defensive: pathlib ops are safe, but guards against non-Path input
        return {
            "diffusion_variant": "-",
            "gen_config": "-",
            "selection": "-",
        }


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested evaluation.json into flat columns.

    Flattens:
        - comparisons.real_vs_selected.fid → rvs_fid
        - comparisons.real_vs_generated.precision → rvg_precision
        - dataset_sizes.real → ds_real
    """
    flat: Dict[str, Any] = {}

    # Flatten comparisons
    comparisons = metrics.get("comparisons", {})
    if isinstance(comparisons, dict):
        for pair_key, prefix in _COMPARISON_PREFIXES.items():
            pair_metrics = comparisons.get(pair_key, {})
            if isinstance(pair_metrics, dict):
                for metric_name, value in pair_metrics.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        flat[f"{prefix}_{metric_name}"] = value
                    else:
                        _logger.debug(
                            "Skipping non-scalar comparisons.%s.%s",
                            pair_key,
                            metric_name,
                        )

    # Flatten dataset_sizes
    dataset_sizes = metrics.get("dataset_sizes", {})
    if isinstance(dataset_sizes, dict):
        for size_key, value in dataset_sizes.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                flat[f"ds_{size_key}"] = value
            else:
                _logger.debug("Skipping non-scalar dataset_sizes.%s", size_key)

    return flat


def load_selection_eval_results(
    base_dir: str = "outputs",
    diffusion_pattern: str = "diffusion-*",
) -> List[Dict[str, Any]]:
    """Scan for selection-eval evaluation.json files and load all results.

    Args:
        base_dir: Base directory containing experiment outputs.
        diffusion_pattern: Glob pattern for diffusion variant directories.

    Returns:
        List of dictionaries, each containing experiment name + flattened metrics.
    """
    pattern = str(
        Path(base_dir)
        / diffusion_pattern
        / "selection-eval"
        / "*"
        / "reports"
        / "evaluation.json"
    )
    results = []

    for json_path in sorted(glob(pattern)):
        try:
            with open(json_path, encoding="utf-8") as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            _logger.warning("Skipping malformed file %s: %s", json_path, e)
            continue
        if not isinstance(metrics, dict):
            _logger.warning(
                "Skipping malformed file %s: expected a JSON object, got %s",
                json_path,
                type(metrics).__name__,
            )
            continue

        dimensions = _parse_selection_eval_path(json_path)
        experiment = (
            f"{dimensions['diffusion_variant']}"
            f"_{dimensions['gen_config']}"
            f"_{dimensions['selection']}"
        )

        entry: Dict[str, Any] = {"experiment": experiment}
        entry.update(dimensions)

        # Flatten nested metrics
        flat = _flatten_metrics(metrics)

        # Prevent flattened keys from overwriting metadata columns
        reserved_keys = set(entry.keys())
        conflicts = reserved_keys & flat.keys()
        if conflicts:
            _logger.warning("Skipping reserved keys in %s: %s", json_path, conflicts)
            flat = {k: v for k, v in flat.items() if k not in conflicts}

        entry.update(flat)
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

    return pd.DataFrame(results)


def generate_selection_eval_table(df: pd.DataFrame) -> str:
    """Generate markdown table of selection evaluation performance.

    Args:
        df: DataFrame with evaluation results.

    Returns:
        Markdown-formatted table string.
    """
    if df.empty:
        return "No evaluation results found.\n"

    display_cols = ["experiment", "diffusion_variant", "gen_config", "selection"]
    metric_cols = [m for m in KEY_METRICS if m in df.columns]
    cols = display_cols + metric_cols

    available = [c for c in cols if c in df.columns]
    subset = df[available].copy()

    # Sort by rvs_fid ascending (lower FID = better)
    if "rvs_fid" in subset.columns:
        subset = subset.sort_values(by="rvs_fid", ascending=True)  # type: ignore[call-overload]

    result: Optional[str] = subset.to_markdown(index=False, floatfmt=".4f")
    return result if result is not None else "No evaluation results found.\n"


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
        if metric not in df.columns or bool(df[metric].isna().all()):
            continue

        if metric in LOWER_IS_BETTER:
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()

        if best_idx is not None and bool(pd.notna(best_idx)):
            best_row = df.loc[best_idx]
            rows.append(
                {
                    "metric": metric,
                    "best_experiment": best_row["experiment"],
                    "value": best_row[metric],
                    "diffusion_variant": best_row["diffusion_variant"],
                }
            )

    if not rows:
        return "No best-per-metric data available.\n"

    result: Optional[str] = pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")
    return result if result is not None else ""


def generate_report(
    base_dir: str = "outputs",
    output_dir: str = "outputs/evaluation_report",
) -> None:
    """Generate full selection evaluation report.

    Args:
        base_dir: Base directory containing experiment outputs.
        output_dir: Output directory for report files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_selection_eval_results(base_dir)
    _logger.info("Found %d selection-eval results", len(results))

    if not results:
        _logger.warning("No selection-eval results found. Run evaluations first.")
        return

    df = build_comparison_dataframe(results)

    # Build report
    report_lines = [
        "# Selection Evaluation Report: Generation & Selection Quality",
        "",
        f"Total experiments: {len(df)}",
        "",
    ]

    # Table 1: All results
    report_lines.append("## Table 1: Selection Evaluation Results")
    report_lines.append("")
    report_lines.append(generate_selection_eval_table(df))
    report_lines.append("")

    # Table 2: Best config per metric
    report_lines.append("## Table 2: Best Configuration per Metric")
    report_lines.append("")
    report_lines.append(generate_best_per_metric(df))
    report_lines.append("")

    # Write report
    report_text = "\n".join(report_lines)

    # Save markdown
    report_md_path = output_path / "selection_eval_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    _logger.info("Report saved to: %s", report_md_path)

    # Save CSV
    csv_path = output_path / "selection_eval_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    _logger.info("CSV saved to: %s", csv_path)

    # Log summary
    _logger.info("Report summary:\n%s", report_text)


def main() -> None:
    """CLI entry point for selection evaluation report generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate selection evaluation report")
    parser.add_argument(
        "--base-dir",
        default="outputs",
        help="Base directory containing experiment outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation_report",
        help="Output directory for report files",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_report(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
