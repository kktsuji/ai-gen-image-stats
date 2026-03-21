"""Statistical Testing Utilities

Provides paired t-test, Cohen's d (with Hedges' g correction),
and multiple comparison correction for comparing classifier experiments
across multiple seeds.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.special import gammaln

_logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of a paired statistical comparison for one metric."""

    metric: str
    baseline_mean: float
    baseline_std: float
    treatment_mean: float
    treatment_std: float
    mean_diff: float
    t_statistic: float
    p_value: float
    p_value_corrected: float
    cohens_d: float
    effect_size_interpretation: str
    significant: bool


def paired_ttest(
    baseline_values: np.ndarray,
    treatment_values: np.ndarray,
) -> Tuple[float, float]:
    """Paired t-test comparing baseline vs treatment.

    Tests whether the mean difference between paired observations is zero.
    Each pair shares the same random seed, so seed-to-seed variance cancels
    out, isolating the effect of the method change.

    Args:
        baseline_values: Metric values from baseline runs, shape (n,).
        treatment_values: Metric values from treatment runs, shape (n,).

    Returns:
        Tuple of (t_statistic, p_value). Returns (nan, nan) if n < 2
        or all differences are zero.
    """
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    treatment_values = np.asarray(treatment_values, dtype=np.float64)

    if len(baseline_values) != len(treatment_values):
        raise ValueError(
            f"Arrays must have same length: "
            f"{len(baseline_values)} vs {len(treatment_values)}"
        )

    n = len(baseline_values)
    if n < 2:
        return (float("nan"), float("nan"))

    diffs = treatment_values - baseline_values
    if np.std(diffs, ddof=1) < np.finfo(np.float64).eps * 100:
        return (float("nan"), float("nan"))

    # Compute as treatment - baseline so t > 0 means treatment is better,
    # consistent with mean_diff and cohens_d sign convention.
    t_stat, p_value = stats.ttest_rel(treatment_values, baseline_values)
    return (float(t_stat), float(p_value))


def cohens_d_paired(
    baseline_values: np.ndarray,
    treatment_values: np.ndarray,
) -> float:
    """Cohen's d for paired samples with Hedges' g correction.

    Computes d = mean(diffs) / std(diffs, ddof=1), then applies the exact
    Hedges' correction factor J(df) = Γ(df/2) / (√(df/2) · Γ((df-1)/2))
    to reduce small-sample bias. The exact formula is used instead of the
    approximation (1 - 3/(4*df - 1)) which breaks down for df <= 1.

    Args:
        baseline_values: Metric values from baseline runs, shape (n,).
        treatment_values: Metric values from treatment runs, shape (n,).

    Returns:
        Corrected effect size (Hedges' g). Returns nan if n < 2 or
        std of differences is zero.
    """
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    treatment_values = np.asarray(treatment_values, dtype=np.float64)

    if len(baseline_values) != len(treatment_values):
        raise ValueError(
            f"Arrays must have same length: "
            f"{len(baseline_values)} vs {len(treatment_values)}"
        )

    n = len(baseline_values)
    if n < 2:
        return float("nan")

    diffs = treatment_values - baseline_values
    sd = float(np.std(diffs, ddof=1))

    if sd < np.finfo(np.float64).eps * 100:
        return float("nan")

    d = float(np.mean(diffs)) / sd

    # Hedges' correction factor reduces small-sample bias.
    # For df=1 (n=2) the exact gamma formula is undefined (Γ(0) = ±∞),
    # so we return uncorrected Cohen's d. For df≥2 we use the exact formula:
    # J(df) = Γ(df/2) / (√(df/2) · Γ((df-1)/2))
    df = n - 1
    if df < 2:
        _logger.warning(
            "Hedges' correction not applicable for n=2 (df=1); "
            "returning uncorrected Cohen's d"
        )
        return d

    correction = float(
        np.exp(gammaln(df / 2) - gammaln((df - 1) / 2)) / np.sqrt(df / 2)
    )
    return d * correction


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude.

    Uses conventional thresholds (Cohen, 1988):
        |d| < 0.2:  negligible
        |d| < 0.5:  small
        |d| < 0.8:  medium
        |d| >= 0.8: large

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string: "negligible", "small", "medium", or "large".
    """
    if not np.isfinite(d):
        return "undefined"

    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def adjust_pvalues(
    pvalues: List[float],
    method: str = "benjamini-hochberg",
) -> List[float]:
    """Adjust p-values for multiple comparisons.

    Args:
        pvalues: List of raw p-values.
        method: Correction method. "bonferroni" or "benjamini-hochberg".

    Returns:
        List of adjusted p-values in the same order as input.

    Raises:
        ValueError: If method is not recognized.
    """
    if not pvalues:
        return []

    n = len(pvalues)
    arr = np.array(pvalues, dtype=np.float64)

    if method == "bonferroni":
        adjusted = np.minimum(arr * n, 1.0)
        return adjusted.tolist()

    elif method == "benjamini-hochberg":
        # Benjamini-Hochberg step-up procedure
        # 1. Sort p-values
        sorted_indices = np.argsort(arr)
        sorted_pvals = arr[sorted_indices]

        # 2. Compute adjusted p-values: p_adj[i] = p[i] * n / rank[i]
        ranks = np.arange(1, n + 1)
        adjusted_sorted = np.minimum(sorted_pvals * n / ranks, 1.0)

        # 3. Enforce monotonicity (cumulative minimum from the right)
        for i in range(n - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

        # 4. Restore original order
        result = np.empty(n)
        result[sorted_indices] = adjusted_sorted
        return result.tolist()

    else:
        raise ValueError(
            f"Unknown correction method: {method!r}. "
            f"Use 'bonferroni' or 'benjamini-hochberg'."
        )


def apply_correction_with_nan(
    pvalues: List[float],
    method: str = "benjamini-hochberg",
) -> List[float]:
    """Apply multiple comparison correction, passing NaN values through.

    Finite p-values are corrected together; NaN entries are preserved in place.

    Args:
        pvalues: List of raw p-values (may contain NaN).
        method: Correction method for ``adjust_pvalues``.

    Returns:
        List of corrected p-values in the same order as input.
    """
    finite_mask = [math.isfinite(p) for p in pvalues]
    finite_pvals = [p for p, m in zip(pvalues, finite_mask) if m]

    if finite_pvals:
        corrected_finite = adjust_pvalues(finite_pvals, method=method)
    else:
        corrected_finite = []

    corrected_iter = iter(corrected_finite)
    return [next(corrected_iter) if m else float("nan") for m in finite_mask]


def compute_raw_comparisons(
    baseline_values: Dict[str, np.ndarray],
    treatment_values: Dict[str, np.ndarray],
    metric_names: List[str],
) -> Tuple[List[dict], List[float]]:
    """Compute raw paired t-test and Cohen's d for each metric (no correction).

    This is the shared building block for both single-pair comparison
    (``compare_experiment_pair``) and multi-pair global correction
    (``generate_statistical_comparison_table``).

    Args:
        baseline_values: Dict mapping metric name -> array of seed values.
        treatment_values: Dict mapping metric name -> array of seed values.
        metric_names: List of metric names to compare.

    Returns:
        Tuple of (raw_results, raw_pvalues) where raw_results is a list of
        dicts with keys: metric, baseline_mean, baseline_std, treatment_mean,
        treatment_std, mean_diff, t_statistic, p_value, cohens_d; and
        raw_pvalues is a parallel list of uncorrected p-values.
    """
    raw_results: List[dict] = []
    raw_pvalues: List[float] = []

    for metric in metric_names:
        if metric not in baseline_values or metric not in treatment_values:
            _logger.warning(f"Metric {metric!r} missing from one or both experiments")
            continue

        bl = np.asarray(baseline_values[metric], dtype=np.float64)
        tr = np.asarray(treatment_values[metric], dtype=np.float64)

        if len(bl) != len(tr):
            _logger.warning(
                f"Metric {metric!r}: mismatched seed counts "
                f"({len(bl)} vs {len(tr)}), skipping"
            )
            continue

        t_stat, p_val = paired_ttest(bl, tr)
        d = cohens_d_paired(bl, tr)

        raw_results.append(
            {
                "metric": metric,
                "baseline_mean": float(np.mean(bl)),
                "baseline_std": float(np.std(bl, ddof=1)) if len(bl) > 1 else 0.0,
                "treatment_mean": float(np.mean(tr)),
                "treatment_std": float(np.std(tr, ddof=1)) if len(tr) > 1 else 0.0,
                "mean_diff": float(np.mean(tr) - np.mean(bl)),
                "t_statistic": t_stat,
                "p_value": p_val,
                "cohens_d": d,
            }
        )
        raw_pvalues.append(p_val)

    return raw_results, raw_pvalues


def finalize_comparisons(
    raw_results: List[dict],
    raw_pvalues: List[float],
    alpha: float = 0.05,
    correction_method: str = "benjamini-hochberg",
) -> List[ComparisonResult]:
    """Apply multiple comparison correction and build ComparisonResult objects.

    Args:
        raw_results: Raw comparison dicts from ``compute_raw_comparisons``.
        raw_pvalues: Parallel list of uncorrected p-values.
        alpha: Significance threshold (after correction). Must be in (0, 1).
        correction_method: P-value correction method.

    Returns:
        List of ComparisonResult, one per entry in raw_results.

    Raises:
        ValueError: If alpha is not in (0, 1).
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    corrected_all = apply_correction_with_nan(raw_pvalues, method=correction_method)

    results: List[ComparisonResult] = []
    for raw, p_corr in zip(raw_results, corrected_all):
        results.append(
            ComparisonResult(
                metric=raw["metric"],
                baseline_mean=raw["baseline_mean"],
                baseline_std=raw["baseline_std"],
                treatment_mean=raw["treatment_mean"],
                treatment_std=raw["treatment_std"],
                mean_diff=raw["mean_diff"],
                t_statistic=raw["t_statistic"],
                p_value=raw["p_value"],
                p_value_corrected=p_corr,
                cohens_d=raw["cohens_d"],
                effect_size_interpretation=interpret_effect_size(raw["cohens_d"]),
                significant=bool(math.isfinite(p_corr) and p_corr < alpha),
            )
        )

    return results


def compare_experiment_pair(
    baseline_values: Dict[str, np.ndarray],
    treatment_values: Dict[str, np.ndarray],
    metric_names: List[str],
    alpha: float = 0.05,
    correction_method: str = "benjamini-hochberg",
) -> List[ComparisonResult]:
    """Compare two experiments across multiple metrics with correction.

    Runs paired t-test and Cohen's d for each metric, then applies
    multiple comparison correction to the p-values.

    Args:
        baseline_values: Dict mapping metric name -> array of seed values.
        treatment_values: Dict mapping metric name -> array of seed values.
        metric_names: List of metric names to compare.
        alpha: Significance threshold (after correction). Must be in (0, 1).
        correction_method: P-value correction method.

    Returns:
        List of ComparisonResult, one per metric.

    Raises:
        ValueError: If alpha is not in (0, 1).
    """
    raw_results, raw_pvalues = compute_raw_comparisons(
        baseline_values, treatment_values, metric_names
    )
    return finalize_comparisons(raw_results, raw_pvalues, alpha, correction_method)
