"""Cross-split robustness report (honest confidence intervals).

Aggregates a repeated stratified k-fold sweep into split-level confidence
intervals. The single-split ``evaluation_report`` reports mean +/- std across
*initialisation seeds* only, which is optimistic for a tiny minority class: it
ignores how much the metric moves when the train/test partition itself changes.
This report makes the **split** the experimental unit instead.

Protocol (see ``outputs/reports/experiment_plan_robustness_ci.md``):

  - Directory layout: ``{base_dir}/split{S}/{family}/{experiment}/seed{N}/reports``.
    One ``{family}`` (e.g. ``binary-depth``) is aggregated per invocation.
  - For each split S, the per-seed metrics are averaged into one value
    ``m_{exp,S}`` (init-seed noise removed). **The within-split seed std is NOT
    pooled** -- it answers the wrong question.
  - Across-split CI = spread of ``m_{exp,S}`` over the splits (t-CI, df = n-1).
  - Treatment-vs-baseline comparisons are **paired by split** (not by seed):
    the n per-split differences feed paired-t, Wilcoxon signed-rank, Cohen's dz
    and BH multiple-comparison correction.

Never treat ``(split, seed)`` as an independent unit -- that is
pseudo-replication and inflates significance.

Usage:
    python -m src.experiments.classifier.cross_split_report \
        --base-dir outputs/multisplit --family binary-depth \
        [--baseline-name baseline__ws] [--output-dir DIR] \
        [--report-name evaluation.json] [--positive-class-index N]
"""

import argparse
import logging
import math
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.classifier.evaluation_report import (
    _parse_experiment_name,
    build_comparison_dataframe,
    build_mean_std_dataframe,
    key_metrics,
    load_evaluation_results,
    resolve_positive_class,
)
from src.utils.statistical_testing import (
    apply_correction_with_nan,
    cohens_d_paired,
    interpret_effect_size,
    paired_ttest,
    wilcoxon_signed_rank,
)

_logger = logging.getLogger(__name__)


def _parse_split_label(path: str) -> Optional[int]:
    """Extract the integer split index from a ``.../split{N}/...`` path."""
    for part in Path(path).parts:
        if part.startswith("split"):
            suffix = part[len("split") :]
            if suffix.isdigit():
                return int(suffix)
    return None


def load_per_split_results(
    base_dir: str,
    family: str,
    split_glob: str = "split*",
    report_name: str = "evaluation.json",
) -> Dict[int, List[Dict[str, Any]]]:
    """Load per-(split, seed) evaluation results, grouped by split index.

    Each split directory ``{base_dir}/{split_glob}/{family}`` is handed to the
    existing :func:`load_evaluation_results` (whose 2-level glob is unchanged),
    so this adds the split dimension without touching the per-split aggregators.

    Returns:
        Dict mapping split index -> list of result dicts (each with a "seed"
        field and the parsed experiment metadata).
    """
    per_split: Dict[int, List[Dict[str, Any]]] = {}
    split_dirs = sorted(glob(f"{base_dir}/{split_glob}/{family}"))
    for split_dir in split_dirs:
        split_idx = _parse_split_label(split_dir)
        if split_idx is None:
            _logger.warning("Cannot parse split index from %s, skipping", split_dir)
            continue
        results = load_evaluation_results(base_dir=split_dir, report_name=report_name)
        if results:
            per_split.setdefault(split_idx, []).extend(results)
        else:
            _logger.warning("No evaluation results under %s", split_dir)
    return per_split


def compute_split_means(
    results: List[Dict[str, Any]],
    metric_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Average per-seed metrics into one value per (experiment, metric).

    Reuses :func:`build_mean_std_dataframe` to collapse the seeds, then keeps
    only the mean columns (the ``{metric}_std`` init-seed spread is discarded on
    purpose -- the across-split report recomputes its own SD from these means).

    Aggregation is ``nan_tolerant`` (mean over the finite seeds only): the
    minority-class metrics this report exists to estimate (``pr_auc``,
    ``hardcore_pr_auc_*``) are legitimately NaN for some seeds on a degenerate
    restricted set. The default strict policy would drop the *whole split's*
    contribution to such a metric whenever any single seed was NaN, silently
    shrinking the across-split ``n`` (and the paired-comparison ``n``) without it
    showing in the report header -- the opposite of an honest CI.

    Returns:
        Dict mapping experiment name -> {metric -> seed-mean value}.
    """
    df = build_comparison_dataframe(results)
    if df.empty or "seed" not in df.columns:
        return {}
    msdf = build_mean_std_dataframe(df, metric_names, nan_tolerant=True)
    means: Dict[str, Dict[str, float]] = {}
    # to_dict(records) yields plain-Python scalars (typed Any) so float() below
    # accepts them; iterrows() Series indexing trips the stricter lint gate.
    for row in msdf.to_dict(orient="records"):
        exp = str(row["experiment"])
        per_metric: Dict[str, float] = {}
        for metric in metric_names:
            if metric in row and bool(pd.notna(row[metric])):
                per_metric[metric] = float(row[metric])
        means[exp] = per_metric
    return means


def _t_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Two-sided t confidence interval for the mean of ``values``.

    Returns (mean, ci_lower, ci_upper). With n < 2 the CI bounds are NaN.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean()) if n else float("nan")
    if n < 2:
        return (mean, float("nan"), float("nan"))
    sd = float(arr.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1)) * sd / math.sqrt(n)
    return (mean, mean - half, mean + half)


def build_across_split_summary(
    split_means: Dict[int, Dict[str, Dict[str, float]]],
    metric_names: List[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Across-split mean +/- SD and t-CI per (experiment, metric).

    The unit is the split: each cell aggregates the per-split seed-means.

    Returns:
        Long-format DataFrame: experiment, metric, n_splits, mean, std,
        ci_lower, ci_upper.
    """
    splits = sorted(split_means)
    experiments = sorted({exp for s in splits for exp in split_means[s]})
    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        for metric in metric_names:
            vals = [
                split_means[s][exp][metric]
                for s in splits
                if exp in split_means[s] and metric in split_means[s][exp]
            ]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            mean, lo, hi = _t_ci(arr, confidence)
            rows.append(
                {
                    "experiment": exp,
                    "metric": metric,
                    "n_splits": int(arr.size),
                    "mean": mean,
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )
    return pd.DataFrame(rows)


def _select_baseline(
    split_means: Dict[int, Dict[str, Dict[str, float]]],
    baseline_name: Optional[str],
    metric_names: List[str],
    positive_class: int,
) -> Optional[str]:
    """Resolve which experiment is the comparison baseline.

    Honours an explicit ``baseline_name`` (if present in the data); otherwise
    auto-selects the ``baseline``-typed experiment with the highest across-split
    mean positive-class recall (mirroring ``evaluation_report``).
    """
    all_exps = {exp for s in split_means for exp in split_means[s]}
    if baseline_name and baseline_name in all_exps:
        return baseline_name
    if baseline_name:
        _logger.warning(
            "Requested baseline %r not found; auto-selecting.", baseline_name
        )

    baselines = [e for e in all_exps if _parse_experiment_name(e)["type"] == "baseline"]
    if not baselines:
        return None

    rank_key = f"recall_{positive_class}"
    if not any(
        rank_key in split_means[s][e]
        for s in split_means
        for e in baselines
        if e in split_means[s]
    ):
        rank_key = next((m for m in metric_names), None)

    best_name, best_score = None, -float("inf")
    for exp in sorted(baselines):
        vals = [
            split_means[s][exp][rank_key]
            for s in split_means
            if exp in split_means[s] and rank_key and rank_key in split_means[s][exp]
        ]
        if vals:
            score = float(np.mean(vals))
            if score > best_score:
                best_score, best_name = score, exp
    # Fallback: no baseline had any values for rank_key, so return the
    # alphabetically first baseline. Downstream filters in
    # compute_cross_split_comparisons gracefully handle a baseline with no
    # shared metric data (yielding an empty comparisons DataFrame).
    return best_name or sorted(baselines)[0]


def compute_cross_split_comparisons(
    split_means: Dict[int, Dict[str, Dict[str, float]]],
    baseline: str,
    metric_names: List[str],
    alpha: float = 0.05,
    correction_method: str = "benjamini-hochberg",
) -> pd.DataFrame:
    """Paired-by-split treatment-vs-baseline comparisons.

    For each treatment x metric, the per-split paired differences feed paired-t,
    Wilcoxon signed-rank and Cohen's dz; both p-value families are BH-corrected
    globally across all comparisons.

    Returns:
        DataFrame with one row per (treatment, metric).
    """
    splits = sorted(split_means)
    all_exps = sorted({exp for s in splits for exp in split_means[s]})
    treatments = [
        e
        for e in all_exps
        if e != baseline
        and _parse_experiment_name(e)["type"] in ("synthetic", "transfer", "unknown")
    ]

    raw: List[Dict[str, Any]] = []
    t_pvals: List[float] = []
    w_pvals: List[float] = []

    for treatment in treatments:
        for metric in metric_names:
            # Pair by split: keep only splits where both have the metric.
            common = [
                s
                for s in splits
                if metric in split_means[s].get(baseline, {})
                and metric in split_means[s].get(treatment, {})
            ]
            if len(common) < 2:
                continue
            bl_vec = np.array(
                [split_means[s][baseline][metric] for s in common], dtype=np.float64
            )
            tr_vec = np.array(
                [split_means[s][treatment][metric] for s in common], dtype=np.float64
            )
            t_stat, t_p = paired_ttest(bl_vec, tr_vec)
            w_stat, w_p = wilcoxon_signed_rank(bl_vec, tr_vec)
            d = cohens_d_paired(bl_vec, tr_vec)
            raw.append(
                {
                    "treatment": treatment,
                    "metric": metric,
                    "n_splits": len(common),
                    "baseline_mean": float(bl_vec.mean()),
                    "treatment_mean": float(tr_vec.mean()),
                    "mean_diff": float(tr_vec.mean() - bl_vec.mean()),
                    "t_statistic": t_stat,
                    "p_value_ttest": t_p,
                    "wilcoxon_statistic": w_stat,
                    "p_value_wilcoxon": w_p,
                    "cohens_dz": d,
                    "effect_size": interpret_effect_size(d),
                }
            )
            t_pvals.append(t_p)
            w_pvals.append(w_p)

    if not raw:
        return pd.DataFrame()

    t_corr = apply_correction_with_nan(t_pvals, method=correction_method)
    w_corr = apply_correction_with_nan(w_pvals, method=correction_method)
    for row, tc, wc in zip(raw, t_corr, w_corr):
        row["p_ttest_corrected"] = tc
        row["p_wilcoxon_corrected"] = wc
        # Paired-t is primary, but it is *undefined* (NaN) when the per-split
        # differences have zero variance -- i.e. a constant non-zero shift, which
        # is the STRONGEST possible evidence (every split moved the same way).
        # Falling back to the BH-corrected Wilcoxon p there avoids silently
        # reporting that maximally-consistent case as non-significant.
        if math.isfinite(tc):
            row["significant"] = bool(tc < alpha)
        else:
            row["significant"] = bool(math.isfinite(wc) and wc < alpha)

    return pd.DataFrame(raw)


def _format_summary_table(summary: pd.DataFrame) -> str:
    """Pivot the long across-split summary into a per-experiment markdown table."""
    if summary.empty:
        return "No across-split summary available.\n"
    rows: List[Dict[str, Any]] = []
    for exp, grp in summary.groupby("experiment"):
        row: Dict[str, Any] = {"experiment": exp}
        n_vals = [int(v) for v in grp["n_splits"] if bool(pd.notna(v))]
        row["n_splits"] = max(n_vals) if n_vals else 0
        for _, r in grp.iterrows():
            row[str(r["metric"])] = (
                f"{r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
                if bool(pd.notna(r["ci_lower"]))
                else f"{r['mean']:.4f}"
            )
        rows.append(row)
    out = pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
    return out or ""


def _format_comparison_table(comparisons: pd.DataFrame) -> str:
    """Render the paired-by-split comparison DataFrame as a markdown table."""
    if comparisons.empty:
        return "No cross-split comparisons available.\n"
    rows: List[Dict[str, Any]] = []
    # to_dict(records) yields plain-Python scalars (typed Any), so int()/
    # math.isfinite() below accept them -- iterrows() Series indexing is typed
    # as Series|ndarray under stricter pyright/pandas and fails the lint gate.
    for r in comparisons.to_dict(orient="records"):
        marker = "*" if bool(r["significant"]) else ""
        rows.append(
            {
                "treatment": r["treatment"],
                "metric": r["metric"],
                "n_splits": int(r["n_splits"]),
                "baseline": f"{r['baseline_mean']:.4f}",
                "treatment_mean": f"{r['treatment_mean']:.4f}",
                "diff": f"{r['mean_diff']:+.4f}",
                "cohens_dz": (
                    f"{r['cohens_dz']:.3f} ({r['effect_size']})"
                    if math.isfinite(r["cohens_dz"])
                    else "N/A"
                ),
                "p_t(BH)": (
                    f"{r['p_ttest_corrected']:.4f}{marker}"
                    if math.isfinite(r["p_ttest_corrected"])
                    else "N/A"
                ),
                # When paired-t is degenerate (NaN), significance falls back to
                # Wilcoxon, so the marker must ride on this cell -- otherwise a
                # Wilcoxon-significant row shows no '*' anywhere in the table.
                "p_wilcoxon(BH)": (
                    f"{r['p_wilcoxon_corrected']:.4f}"
                    f"{marker if not math.isfinite(r['p_ttest_corrected']) else ''}"
                    if math.isfinite(r["p_wilcoxon_corrected"])
                    else "N/A"
                ),
            }
        )
    out = pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
    return out or ""


def generate_cross_split_report(
    base_dir: str,
    family: str,
    output_dir: str,
    split_glob: str = "split*",
    report_name: str = "evaluation.json",
    baseline_name: Optional[str] = None,
    positive_class: Optional[int] = None,
    alpha: float = 0.05,
    correction_method: str = "benjamini-hochberg",
    confidence: float = 0.95,
) -> None:
    """Generate the cross-split robustness report (markdown + CSVs)."""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    per_split = load_per_split_results(base_dir, family, split_glob, report_name)
    if not per_split:
        _logger.warning(
            "No per-split results found under %s/%s/%s", base_dir, split_glob, family
        )
        return

    all_results = [r for results in per_split.values() for r in results]
    pos_class = resolve_positive_class(all_results, positive_class)
    metrics = key_metrics(pos_class)

    split_means = {
        s: compute_split_means(results, metrics) for s, results in per_split.items()
    }
    # Drop splits that produced no usable means.
    split_means = {s: m for s, m in split_means.items() if m}
    n_splits = len(split_means)

    summary = build_across_split_summary(split_means, metrics, confidence)
    # Restrict to metrics actually present.
    present = set(summary["metric"]) if "metric" in summary.columns else set()
    present_metrics = [m for m in metrics if m in present]

    baseline = _select_baseline(split_means, baseline_name, present_metrics, pos_class)
    if baseline is None:
        _logger.warning(
            "No baseline experiment found in family %r; the report will contain "
            "only the across-split summary (no treatment-vs-baseline comparisons). "
            "An absent comparison section is NOT the same as 'no significant "
            "differences'.",
            family,
        )
    comparisons = (
        compute_cross_split_comparisons(
            split_means, baseline, present_metrics, alpha, correction_method
        )
        if baseline is not None
        else pd.DataFrame()
    )

    lines = [
        f"# Cross-Split Robustness Report: {family}",
        "",
        f"Splits (experimental unit): {n_splits} ({sorted(split_means)})",
        f"Confidence level: {confidence} (t-CI, df = n_splits - 1)",
        f"Positive class index: {pos_class}",
        "",
        "Unit = split. Each cell averages init-seeds within a split, then "
        "aggregates across splits. Within-split seed std is NOT pooled.",
        "",
        "## Across-Split Summary (mean [CI])",
        "",
        _format_summary_table(summary),
        "",
    ]
    if baseline is not None and not comparisons.empty:
        sig = sum(bool(v) for v in comparisons["significant"])
        total = len(comparisons)
        lines += [
            "## Treatment vs Baseline (paired by split)",
            "",
            f"Baseline: **{baseline}**",
            f"Correction: {correction_method} (global, n={total}), alpha={alpha}",
            f"Significant (paired-t BH; Wilcoxon BH when t is degenerate): "
            f"{sig}/{total}",
            "",
            _format_comparison_table(comparisons),
            "",
        ]

    (out_path / "cross_split_report.md").write_text("\n".join(lines))
    summary.to_csv(out_path / "cross_split_summary.csv", index=False)
    if not comparisons.empty:
        comparisons.to_csv(out_path / "cross_split_comparisons.csv", index=False)
    _logger.info("Cross-split report written to %s", out_path)


def main() -> None:
    """CLI entry point for the cross-split robustness report."""
    parser = argparse.ArgumentParser(description="Cross-split robustness report")
    parser.add_argument("--base-dir", default="outputs/multisplit")
    parser.add_argument(
        "--family",
        required=True,
        help="Experiment family subdir under each split (e.g. binary-depth)",
    )
    parser.add_argument("--split-glob", default="split*")
    parser.add_argument("--report-name", default="evaluation.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--positive-class-index", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--correction-method",
        default="benjamini-hochberg",
        choices=["benjamini-hochberg", "bonferroni"],
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.base_dir}/cross_split/{args.family}"
    logging.basicConfig(level=logging.INFO)
    generate_cross_split_report(
        base_dir=args.base_dir,
        family=args.family,
        output_dir=output_dir,
        split_glob=args.split_glob,
        report_name=args.report_name,
        baseline_name=args.baseline_name,
        positive_class=args.positive_class_index,
        alpha=args.alpha,
        correction_method=args.correction_method,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
