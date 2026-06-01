"""Decision-Threshold Analysis (post-hoc operating-point optimization).

Track B / item 3: the classifier reports metrics at the default argmax (0.5)
operating point, which is conservative for the minority (abnormal) class and
leaves recall on the table even when pr_auc is strong. This module re-selects a
better operating point *without retraining*, reusing the probabilities already
saved by evaluate mode (``reports/predictions_{split}.npz``).

Two modes:

- Default (val -> test): for each experiment/seed, pick the threshold on the
  VALIDATION split (criterion configurable), then report metrics on the
  held-out TEST split at that threshold. This is the leak-free protocol.
- ``--in-sample`` (LEAKY probe): when no val predictions are available (e.g. an
  archived run that only saved test predictions), select AND report on TEST.
  The numbers are an optimistic upper bound, NOT reportable; used only as a
  pre-retraining go/no-go probe. Output is clearly labelled "IN-SAMPLE/LEAKY".

Directory layout mirrors ``evaluation_report.py``:
  - Multi-seed:  {base_dir}/{experiment}/seed{N}/reports/predictions_{split}.npz
  - Single-seed: {base_dir}/{experiment}/reports/predictions_{split}.npz
Filename resolution falls back to the unsuffixed ``predictions.npz`` (used by
legacy/archived runs that saved a single split).

Usage:
    python -m src.experiments.classifier.threshold_analysis \
        [--base-dir outputs/classifier] [--output-dir outputs/threshold_analysis] \
        [--criterion max_f1_1|precision_at_recall] [--target-recall 0.9] \
        [--in-sample]
"""

import argparse
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

from src.experiments.classifier.evaluation_report import (
    _format_mean_std,
    _parse_experiment_name,
)

_logger = logging.getLogger(__name__)

# The positive (minority / abnormal) class index for the binary task.
POSITIVE_CLASS = 1

# Metric base names produced per experiment (each may gain a "_std" column when
# aggregated across seeds). "_tau" = at the selected threshold, "_base" = at the
# default 0.5 operating point (equivalent to argmax for the binary task).
METRIC_COLUMNS = [
    "tau",
    "recall_1_tau",
    "precision_1_tau",
    "f1_1_tau",
    "balanced_accuracy_tau",
    "recall_1_base",
    "precision_1_base",
    "f1_1_base",
    "balanced_accuracy_base",
    "delta_recall_1",
]

VALID_CRITERIA = ("max_f1_1", "precision_at_recall")


def metrics_at_threshold(
    probs_pos: np.ndarray,
    targets: np.ndarray,
    tau: float,
) -> Dict[str, float]:
    """Compute minority-class metrics for a binary task at a given threshold.

    Mirrors the sklearn calls in ``ClassifierTrainer._compute_classification_metrics``
    (labels=[0, 1], zero_division=0.0) so threshold-derived numbers are directly
    comparable to the trainer's reported metrics.

    Args:
        probs_pos: Predicted probability of the positive class, shape (N,).
        targets: Ground-truth labels (0/1), shape (N,).
        tau: Decision threshold; predict positive iff probs_pos >= tau.

    Returns:
        Dict with recall_1, precision_1, f1_1, balanced_accuracy.
    """
    preds = (probs_pos >= tau).astype(int)
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        targets,
        preds,
        labels=[0, 1],
        zero_division=0.0,  # type: ignore[arg-type]
    )
    return {
        "recall_1": float(rec_arr[POSITIVE_CLASS]),  # type: ignore[index]
        "precision_1": float(prec_arr[POSITIVE_CLASS]),  # type: ignore[index]
        "f1_1": float(f1_arr[POSITIVE_CLASS]),  # type: ignore[index]
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
    }


def select_threshold(
    probs_pos: np.ndarray,
    targets: np.ndarray,
    criterion: str = "max_f1_1",
    target_recall: float = 0.9,
    grid_points: int = 201,
) -> float:
    """Select a decision threshold on a (validation) split.

    Args:
        probs_pos: Positive-class probabilities, shape (N,).
        targets: Ground-truth labels (0/1), shape (N,).
        criterion:
            - "max_f1_1": threshold maximizing the minority-class F1.
            - "precision_at_recall": among thresholds with recall_1 >=
              target_recall, the one maximizing precision_1; if none reach the
              floor, the threshold with the highest recall_1 (lowest tau wins
              ties) so the operating point degrades gracefully.
        target_recall: Recall floor for the "precision_at_recall" criterion.
        grid_points: Number of evenly spaced thresholds over [0, 1].

    Returns:
        The selected threshold tau*.

    Raises:
        ValueError: If criterion is not recognized.
    """
    if criterion not in VALID_CRITERIA:
        raise ValueError(
            f"Invalid criterion: {criterion!r}. Must be one of {VALID_CRITERIA}"
        )

    grid = np.linspace(0.0, 1.0, grid_points)

    if criterion == "max_f1_1":
        best_tau = 0.5
        best_f1 = -1.0
        for tau in grid:
            f1 = metrics_at_threshold(probs_pos, targets, float(tau))["f1_1"]
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
        return best_tau

    # precision_at_recall
    best_tau = float(grid[0])
    best_precision = -1.0
    best_recall = -1.0
    # Fallback bookkeeping when no threshold reaches the recall floor.
    fallback_tau = float(grid[0])
    fallback_recall = -1.0
    for tau in grid:
        m = metrics_at_threshold(probs_pos, targets, float(tau))
        recall, precision = m["recall_1"], m["precision_1"]
        if recall > fallback_recall:
            fallback_recall = recall
            fallback_tau = float(tau)
        if recall >= target_recall and precision > best_precision:
            best_precision = precision
            best_recall = recall
            best_tau = float(tau)
    if best_precision < 0.0:
        _logger.debug(
            "No threshold reached recall floor %.3f; using max-recall fallback "
            "tau=%.4f (recall=%.4f)",
            target_recall,
            fallback_tau,
            fallback_recall,
        )
        return fallback_tau
    _ = best_recall  # selected for clarity; precision is the optimization target
    return best_tau


def _load_npz(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a predictions npz and return (targets, positive-class probs).

    Returns None on error or if the data is not a binary-probability array.
    """
    try:
        data = np.load(path)
    except (OSError, ValueError) as e:
        _logger.warning("Skipping unreadable predictions %s: %s", path, e)
        return None
    if "targets" not in data or "probs" not in data:
        _logger.warning("Skipping %s: missing 'targets'/'probs' arrays", path)
        return None
    probs = np.asarray(data["probs"])
    targets = np.asarray(data["targets"]).astype(int)
    if probs.ndim != 2 or probs.shape[1] < 2:
        _logger.warning("Skipping %s: probs shape %s is not binary", path, probs.shape)
        return None
    return targets, probs[:, POSITIVE_CLASS]


def _resolve_predictions(reports_dir: Path, split: str) -> Optional[Path]:
    """Find a predictions file for a split, falling back to the unsuffixed name.

    Looks for ``predictions_{split}.npz`` first, then legacy ``predictions.npz``
    (single-split archived runs).
    """
    suffixed = reports_dir / f"predictions_{split}.npz"
    if suffixed.exists():
        return suffixed
    legacy = reports_dir / "predictions.npz"
    if legacy.exists():
        return legacy
    return None


def _iter_reports_dirs(base_dir: str) -> List[Tuple[str, Optional[int], Path]]:
    """Yield (experiment, seed_or_None, reports_dir) for all runs under base_dir.

    Mirrors evaluation_report's multi-seed-first discovery: a single-seed run is
    only used when the experiment has no multi-seed results.
    """
    found: List[Tuple[str, Optional[int], Path]] = []

    multi = sorted(glob(f"{base_dir}/*/seed*/reports"))
    multi_seed_experiments = set()
    for reports in multi:
        path = Path(reports)
        seed_dir = path.parent.name  # "seed3"
        exp_name = path.parent.parent.name
        try:
            seed_num = int(seed_dir.replace("seed", ""))
        except ValueError:
            _logger.warning("Cannot parse seed from %s, skipping", seed_dir)
            continue
        found.append((exp_name, seed_num, path))
        multi_seed_experiments.add(exp_name)

    for reports in sorted(glob(f"{base_dir}/*/reports")):
        path = Path(reports)
        exp_name = path.parent.name
        if exp_name in multi_seed_experiments:
            continue
        found.append((exp_name, None, path))

    return found


def load_prediction_rows(
    base_dir: str,
    in_sample: bool,
    criterion: str,
    target_recall: float,
    grid_points: int,
) -> List[Dict[str, Any]]:
    """Build per-run threshold rows by selecting on val and reporting on test.

    In ``in_sample`` mode, the test split is used for BOTH selection and
    reporting (leaky upper bound).

    Returns:
        A list of dicts with experiment metadata, seed, tau, and the test-set
        metrics at tau and at the 0.5 baseline.
    """
    rows: List[Dict[str, Any]] = []

    for exp_name, seed, reports_dir in _iter_reports_dirs(base_dir):
        test_path = _resolve_predictions(reports_dir, "test")
        if test_path is None:
            _logger.warning("No test predictions in %s, skipping", reports_dir)
            continue
        test = _load_npz(test_path)
        if test is None:
            continue
        test_targets, test_probs = test

        if in_sample:
            sel_targets, sel_probs = test_targets, test_probs
        else:
            val_path = _resolve_predictions(reports_dir, "val")
            # Avoid silently selecting on test: require a distinct val file.
            if val_path is None or val_path.name == "predictions.npz":
                _logger.warning(
                    "No val predictions in %s (need predictions_val.npz for the "
                    "val->test protocol; use --in-sample for a leaky probe), "
                    "skipping",
                    reports_dir,
                )
                continue
            val = _load_npz(val_path)
            if val is None:
                continue
            sel_targets, sel_probs = val

        tau = select_threshold(
            sel_probs, sel_targets, criterion, target_recall, grid_points
        )
        at_tau = metrics_at_threshold(test_probs, test_targets, tau)
        at_base = metrics_at_threshold(test_probs, test_targets, 0.5)

        row: Dict[str, Any] = {"experiment": exp_name}
        row.update(_parse_experiment_name(exp_name))
        if seed is not None:
            row["seed"] = seed
        row["tau"] = tau
        row["recall_1_tau"] = at_tau["recall_1"]
        row["precision_1_tau"] = at_tau["precision_1"]
        row["f1_1_tau"] = at_tau["f1_1"]
        row["balanced_accuracy_tau"] = at_tau["balanced_accuracy"]
        row["recall_1_base"] = at_base["recall_1"]
        row["precision_1_base"] = at_base["precision_1"]
        row["f1_1_base"] = at_base["f1_1"]
        row["balanced_accuracy_base"] = at_base["balanced_accuracy"]
        row["delta_recall_1"] = at_tau["recall_1"] - at_base["recall_1"]
        rows.append(row)

    return rows


def aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-seed rows into one row per experiment (mean and std).

    Single-seed experiments keep their raw values with no std column.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    has_seed = "seed" in df.columns

    out: List[Dict[str, Any]] = []
    for exp_name, group in df.groupby("experiment"):
        if has_seed and bool(group["seed"].notna().any()):
            if bool(group["seed"].duplicated().any()):
                _logger.warning(
                    "Experiment %r has duplicate seeds; omitting from tables",
                    exp_name,
                )
                continue
        agg: Dict[str, Any] = {
            "experiment": exp_name,
            "type": group.iloc[0].get("type", "unknown"),
            "n_seeds": int(len(group)),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric].astype(float)
            agg[metric] = float(values.mean())
            if len(values) > 1:
                agg[f"{metric}_std"] = float(values.std(ddof=1))
        out.append(agg)

    return pd.DataFrame(out)


def generate_threshold_table(agg_df: pd.DataFrame) -> str:
    """Render the aggregated threshold comparison as a markdown table."""
    if agg_df.empty:
        return "No threshold results found.\n"

    display_cols = ["experiment", "type"]
    if "n_seeds" in agg_df.columns:
        display_cols.append("n_seeds")
    cols = display_cols + METRIC_COLUMNS
    available = [c for c in cols if c in agg_df.columns]
    subset = agg_df[available].copy()

    if "recall_1_tau" in subset.columns:
        subset = subset.sort_values(by="recall_1_tau", ascending=False)  # type: ignore[call-overload]

    for metric in METRIC_COLUMNS:
        if metric in subset.columns:
            subset[metric] = _format_mean_std(agg_df.loc[subset.index], metric).values

    result: Optional[str] = subset.to_markdown(index=False, disable_numparse=True)
    return result if result is not None else ""


def generate_report(
    base_dir: str = "outputs/classifier",
    output_dir: str = "outputs/threshold_analysis",
    criterion: str = "max_f1_1",
    target_recall: float = 0.9,
    grid_points: int = 201,
    in_sample: bool = False,
) -> None:
    """Generate the threshold-analysis report (markdown + CSV)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = load_prediction_rows(
        base_dir, in_sample, criterion, target_recall, grid_points
    )
    _logger.info("Computed threshold rows for %d runs", len(rows))
    if not rows:
        _logger.warning("No prediction files found under %s.", base_dir)
        return

    agg_df = aggregate_rows(rows)

    mode_note = (
        "**IN-SAMPLE / LEAKY** (threshold selected AND reported on TEST; "
        "optimistic upper bound, not a reportable result)"
        if in_sample
        else "val -> test (threshold selected on VAL, reported on held-out TEST)"
    )
    crit_note = (
        f"{criterion} (target_recall={target_recall})"
        if criterion == "precision_at_recall"
        else criterion
    )

    lines = [
        "# Decision-Threshold Analysis",
        "",
        f"Protocol: {mode_note}",
        f"Selection criterion: {crit_note}",
        f"Threshold grid: {grid_points} points over [0, 1]",
        f"Runs: {len(rows)}; experiments: {len(agg_df)}",
        "",
        "Columns: `*_tau` = at the selected threshold, `*_base` = at the default "
        "0.5 operating point, `delta_recall_1` = recall_1_tau - recall_1_base.",
        "",
        "## Minority-class metrics at selected vs default threshold",
        "",
        generate_threshold_table(agg_df),
        "",
    ]
    report_text = "\n".join(lines)

    md_path = output_path / "threshold_analysis.md"
    with open(md_path, "w") as f:
        f.write(report_text)
    _logger.info("Report saved to: %s", md_path)

    # Per-seed raw rows (drop verbose parsed-metadata columns kept only for tables)
    csv_path = output_path / "threshold_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _logger.info("CSV saved to: %s", csv_path)

    _logger.info("Report summary:\n%s", report_text)


def main() -> None:
    """CLI entry point for threshold analysis."""
    parser = argparse.ArgumentParser(description="Decision-threshold analysis")
    parser.add_argument(
        "--base-dir",
        default="outputs/classifier",
        help="Base directory for classifier experiments",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/threshold_analysis",
        help="Output directory for report files",
    )
    parser.add_argument(
        "--criterion",
        default="max_f1_1",
        choices=list(VALID_CRITERIA),
        help="Threshold selection criterion (default: max_f1_1)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.9,
        help="Recall floor for the precision_at_recall criterion (default: 0.9)",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=201,
        help="Number of evenly spaced thresholds over [0, 1] (default: 201)",
    )
    parser.add_argument(
        "--in-sample",
        action="store_true",
        help=(
            "LEAKY probe: select AND report on TEST (use when val predictions "
            "are unavailable, e.g. archived runs). Output is an optimistic upper "
            "bound, not a reportable result."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_report(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        criterion=args.criterion,
        target_recall=args.target_recall,
        grid_points=args.grid_points,
        in_sample=args.in_sample,
    )


if __name__ == "__main__":
    main()
