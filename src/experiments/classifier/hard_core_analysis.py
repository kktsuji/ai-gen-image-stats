"""Hard-core direct evaluation report (post-hoc, no retraining).

Computes the abnormal-vs-suspicious restricted ranking metrics (see
:mod:`src.experiments.classifier.hard_core`) from the probabilities already
saved by evaluate mode (``reports/predictions_{split}.npz``), aggregates them
across seeds, and tests significance vs a baseline with paired tests + BH
correction.

This is the **backfill** path for runs that predate the eval-time metric (the
same numbers are stamped into ``evaluation.json`` for new runs and flow through
``evaluation_report``). It is also wired into the pipeline ``summarize`` phase so
split-seed sweeps produce it automatically.

The contrast (suspicious) class is resolved from each run's ``evaluation.json``
(``contrast_class`` if stamped, else the class named ``--contrast-class-name`` in
``class_names``); ``--contrast-class-index`` overrides everything.

Directory layout mirrors ``evaluation_report.py`` / ``threshold_analysis.py``:
  - Multi-seed:  {base_dir}/{experiment}/seed{N}/reports/predictions_{split}.npz
  - Single-seed: {base_dir}/{experiment}/reports/predictions_{split}.npz

Usage:
    python -m src.experiments.classifier.hard_core_analysis \
        [--base-dir outputs/classifier] [--output-dir outputs/hard_core_analysis] \
        [--split test] [--baseline-name baseline__vanilla] \
        [--positive-class-index N] [--contrast-class-index M] \
        [--contrast-class-name suspicious]
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.experiments.classifier.evaluation_report import (
    _format_mean_std,
    _parse_experiment_name,
    generate_statistical_comparison_table,
    load_evaluation_results,
    resolve_positive_class,
)
from src.experiments.classifier.hard_core import (
    DEFAULT_CONTRAST_NAME,
    HARD_CORE_METRIC_KEYS,
    compute_hard_core_metrics,
    resolve_contrast_class,
)
from src.experiments.classifier.threshold_analysis import (
    _iter_reports_dirs,
    _resolve_predictions,
)

_logger = logging.getLogger(__name__)

# Binary oracle ceiling (dedicated abnormal-vs-suspicious 2-class classifier) the
# renormalized hard-core PR-AUC is meant to be contrasted against.
BINARY_ORACLE_PR_AUC = 0.915

# Float metrics aggregated/formatted (NaN-safe). hardcore_n is the integer
# restricted sample count, shown separately. Sourced from the single
# authoritative key list in :mod:`hard_core` so a new metric flows here without
# a second edit.
HARDCORE_FLOAT_COLUMNS: List[str] = list(HARD_CORE_METRIC_KEYS)
HARDCORE_TABLE_COLUMNS: List[str] = HARDCORE_FLOAT_COLUMNS + ["hardcore_n"]


def _load_npz_full(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a predictions npz and return (int targets, full probs matrix).

    Unlike ``threshold_analysis._load_npz`` (which binarizes one-vs-rest), the
    hard-core metric needs the raw labels and the full ``(N, C)`` probability
    matrix. Returns None on error or if probs has fewer than 2 columns.
    """
    try:
        with np.load(path) as data:
            if "targets" not in data or "probs" not in data:
                _logger.warning("Skipping %s: missing 'targets'/'probs' arrays", path)
                return None
            probs = np.asarray(data["probs"], dtype=float)
            targets = np.asarray(data["targets"]).astype(int)
    except (OSError, ValueError) as e:
        _logger.warning("Skipping unreadable predictions %s: %s", path, e)
        return None
    if probs.ndim != 2 or probs.shape[1] < 2:
        _logger.warning(
            "Skipping %s: probs shape %s has fewer than 2 classes", path, probs.shape
        )
        return None
    return targets, probs


def detect_contrast_class(
    results: List[Dict[str, Any]],
    positive_class: int,
    override: Optional[int] = None,
    name: str = DEFAULT_CONTRAST_NAME,
) -> Optional[int]:
    """Resolve the contrast (suspicious) class index across a set of results.

    Precedence: explicit ``override`` wins; else the ``contrast_class`` stamped
    into evaluation.json (most common if runs disagree); else the index of the
    class named ``name`` in the recorded ``class_names``. Returns None when none
    resolve (the caller skips the report).
    """
    if override is not None:
        return override

    stamped = [
        int(r["contrast_class"]) for r in results if r.get("contrast_class") is not None
    ]
    if stamped:
        distinct = set(stamped)
        if len(distinct) > 1:
            most_common = Counter(stamped).most_common(1)[0][0]
            _logger.warning(
                "Results report differing contrast_class values %s; using %s. "
                "Pass --contrast-class-index to override.",
                sorted(distinct),
                most_common,
            )
            return most_common
        return stamped[0]

    # Fall back to name lookup in the first available class_names list.
    for r in results:
        class_names = r.get("class_names")
        if class_names:
            idx = resolve_contrast_class(list(class_names), name=name)
            if idx is not None and idx != positive_class:
                return idx
    return None


def load_hardcore_rows(
    base_dir: str,
    split: str,
    positive_class: int,
    contrast_class: int,
) -> List[Dict[str, Any]]:
    """Build per-run hard-core rows from saved predictions for ``split``."""
    rows: List[Dict[str, Any]] = []
    for exp_name, seed, reports_dir in _iter_reports_dirs(base_dir):
        # Require the split-tagged file: the hard-core metric is reported under a
        # specific split label, so accepting the untagged legacy predictions.npz
        # (which could be any split) would mislabel e.g. val predictions as test.
        pred_path = _resolve_predictions(reports_dir, split, allow_legacy=False)
        if pred_path is None:
            _logger.warning("No %s predictions in %s, skipping", split, reports_dir)
            continue
        loaded = _load_npz_full(pred_path)
        if loaded is None:
            continue
        targets, probs = loaded

        hc = compute_hard_core_metrics(targets, probs, positive_class, contrast_class)
        row: Dict[str, Any] = {"experiment": exp_name}
        row.update(_parse_experiment_name(exp_name))
        if seed is not None:
            row["seed"] = seed
        row.update(hc)
        rows.append(row)
    return rows


def aggregate_hardcore_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-seed hard-core rows into one row per experiment (mean/std).

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
        for metric in HARDCORE_TABLE_COLUMNS:
            if metric not in group.columns:
                continue
            values = np.asarray(group[metric], dtype=float)
            # Average over the finite seeds. Using the pre-filtered ``finite``
            # array (rather than np.nanmean over ``values``) avoids the
            # "Mean of empty slice" RuntimeWarning when every seed is NaN, which
            # happens when the restricted set is degenerate for all seeds.
            finite = values[np.isfinite(values)]
            agg[metric] = float(finite.mean()) if finite.size else float("nan")
            if finite.size > 1:
                agg[f"{metric}_std"] = float(finite.std(ddof=1))
        out.append(agg)

    return pd.DataFrame(out)


def generate_hardcore_table(agg_df: pd.DataFrame) -> str:
    """Render the aggregated hard-core comparison as a markdown table."""
    if agg_df.empty:
        return "No hard-core results found.\n"

    display_cols = ["experiment", "type"]
    if "n_seeds" in agg_df.columns:
        display_cols.append("n_seeds")
    cols = display_cols + [c for c in HARDCORE_TABLE_COLUMNS if c in agg_df.columns]
    subset = agg_df[cols].copy()

    sort_col = "hardcore_pr_auc_renorm"
    if sort_col in subset.columns:
        subset = subset.sort_values(by=sort_col, ascending=False)  # type: ignore[call-overload]

    # hardcore_n is an integer count; format the float metrics as mean +/- std.
    for metric in HARDCORE_FLOAT_COLUMNS:
        if metric in subset.columns:
            subset[metric] = _format_mean_std(agg_df.loc[subset.index], metric).values
    if "hardcore_n" in subset.columns:
        subset["hardcore_n"] = [
            "" if not np.isfinite(v) else f"{v:.0f}"
            for v in agg_df.loc[subset.index, "hardcore_n"]
        ]

    result: Optional[str] = subset.to_markdown(index=False, disable_numparse=True)
    return result if result is not None else ""


def generate_report(
    base_dir: str = "outputs/classifier",
    output_dir: str = "outputs/hard_core_analysis",
    split: str = "test",
    baseline_name: Optional[str] = None,
    alpha: float = 0.05,
    correction_method: str = "benjamini-hochberg",
    positive_class: Optional[int] = None,
    contrast_class: Optional[int] = None,
    contrast_class_name: str = DEFAULT_CONTRAST_NAME,
) -> None:
    """Generate the hard-core direct-evaluation report (markdown + CSV)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Class metadata (positive/contrast indices, class_names) lives in the
    # canonical test report. Class indices are split-invariant, so the test
    # report is authoritative even for a non-test split -- but a sweep that only
    # evaluated `split` has no evaluation.json, so fall back to its split-tagged
    # report rather than aborting with predictions on disk.
    results = load_evaluation_results(base_dir)
    if not results and split != "test":
        results = load_evaluation_results(
            base_dir, report_name=f"evaluation_{split}.json"
        )
    pos = resolve_positive_class(results, positive_class)
    con = detect_contrast_class(results, pos, contrast_class, contrast_class_name)
    if con is None:
        _logger.warning(
            "Could not resolve a contrast class (looked for stamped "
            "contrast_class, then a class named %r). Pass --contrast-class-index "
            "or --contrast-class-name. Aborting hard-core report.",
            contrast_class_name,
        )
        return
    if con == pos:
        _logger.warning(
            "Contrast class %d equals positive class %d; aborting.", con, pos
        )
        return

    rows = load_hardcore_rows(base_dir, split, pos, con)
    _logger.info("Computed hard-core rows for %d runs", len(rows))
    if not rows:
        _logger.warning("No prediction files found under %s.", base_dir)
        return

    agg_df = aggregate_hardcore_rows(rows)
    perseed_df = pd.DataFrame(rows)
    sig_table = generate_statistical_comparison_table(
        perseed_df,
        alpha=alpha,
        correction_method=correction_method,
        baseline_name=baseline_name,
        positive_class=pos,
    )

    lines = [
        "# Hard-Core Direct Evaluation (abnormal vs suspicious)",
        "",
        f"Split: **{split}**; positive class (abnormal) = {pos}; "
        f"contrast class (suspicious) = {con}.",
        f"Runs: {len(rows)}; experiments: {len(agg_df)}",
        "",
        "Evaluation is restricted to samples whose *true* label is the positive "
        "or contrast class. **renorm** = `P(abn)/(P(abn)+P(susp))` (primary; the "
        "apples-to-apples analog of a dedicated 2-class classifier), **raw** = "
        "`P(abn)` (sanity). `hardcore_leak_rate` = fraction of true-abnormal "
        "whose argmax prediction is the contrast class; `hardcore_n` = restricted "
        "sample count.",
        "",
        f"Reference: the binary oracle ceiling (dedicated abnormal-vs-suspicious "
        f"2-class classifier) is PR-AUC ~= {BINARY_ORACLE_PR_AUC}. Compare the "
        "renorm column against it (a renorm value at/below the oracle indicates "
        "the multi-class auxiliary signal did not break the hard core).",
        "",
        "## Aggregated hard-core metrics (mean +/- std across seeds)",
        "",
        generate_hardcore_table(agg_df),
        "",
    ]
    if sig_table:
        lines += [
            "## Significance vs baseline (paired tests, renorm/raw PR-AUC)",
            "",
            sig_table,
            "",
        ]
    report_text = "\n".join(lines)

    md_path = output_path / "hard_core_analysis.md"
    with open(md_path, "w") as f:
        f.write(report_text)
    _logger.info("Report saved to: %s", md_path)

    csv_path = output_path / "hard_core_analysis.csv"
    perseed_df.to_csv(csv_path, index=False)
    _logger.info("CSV saved to: %s", csv_path)

    _logger.info("Report summary:\n%s", report_text)


def main() -> None:
    """CLI entry point for hard-core direct evaluation."""
    parser = argparse.ArgumentParser(
        description="Hard-core direct evaluation (abnormal vs suspicious)"
    )
    parser.add_argument(
        "--base-dir",
        default="outputs/classifier",
        help="Base directory for classifier experiments",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hard_core_analysis",
        help="Output directory for report files",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split's predictions to evaluate (default: test)",
    )
    parser.add_argument(
        "--baseline-name",
        default=None,
        help="Baseline experiment for the significance table (default: auto)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold after correction (default: 0.05)",
    )
    parser.add_argument(
        "--correction-method",
        default="benjamini-hochberg",
        choices=["benjamini-hochberg", "bonferroni"],
        help="Multiple-comparison correction (default: benjamini-hochberg)",
    )
    parser.add_argument(
        "--positive-class-index",
        type=int,
        default=None,
        help=(
            "Positive/abnormal class index (default: auto-detect from "
            "evaluation.json, falling back to 1)"
        ),
    )
    parser.add_argument(
        "--contrast-class-index",
        type=int,
        default=None,
        help=(
            "Contrast/suspicious class index (default: stamped contrast_class, "
            "else the class named --contrast-class-name)"
        ),
    )
    parser.add_argument(
        "--contrast-class-name",
        default=DEFAULT_CONTRAST_NAME,
        help=(
            "Class name to auto-detect the contrast class when no index is given "
            f"(default: {DEFAULT_CONTRAST_NAME})"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_report(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        split=args.split,
        baseline_name=args.baseline_name,
        alpha=args.alpha,
        correction_method=args.correction_method,
        positive_class=args.positive_class_index,
        contrast_class=args.contrast_class_index,
        contrast_class_name=args.contrast_class_name,
    )


if __name__ == "__main__":
    main()
