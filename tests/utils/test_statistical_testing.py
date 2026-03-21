"""Tests for statistical testing utilities."""

import math

import numpy as np
import pytest
from scipy import stats

from src.utils.statistical_testing import (
    ComparisonResult,
    adjust_pvalues,
    cohens_d_paired,
    compare_experiment_pair,
    interpret_effect_size,
    paired_ttest,
)


class TestPairedTtest:
    """Tests for paired_ttest."""

    @pytest.mark.unit
    def test_known_values_from_reference(self) -> None:
        """Verify against the hand-calculated example from the reference doc.

        Baseline: [0.72, 0.68, 0.75, 0.70, 0.66]
        Synthetic: [0.78, 0.75, 0.77, 0.76, 0.71]
        Differences: [0.06, 0.07, 0.02, 0.06, 0.05]
        mean_d = 0.052, sd = 0.0192, t = 6.06, p ≈ 0.004
        """
        baseline = np.array([0.72, 0.68, 0.75, 0.70, 0.66])
        synthetic = np.array([0.78, 0.75, 0.77, 0.76, 0.71])

        t_stat, p_val = paired_ttest(baseline, synthetic)

        # scipy.stats.ttest_rel returns negative t when treatment > baseline
        # because it computes baseline - treatment
        assert abs(t_stat) == pytest.approx(6.06, abs=0.1)
        assert p_val == pytest.approx(0.004, abs=0.002)
        assert p_val < 0.05

    @pytest.mark.unit
    def test_no_difference(self) -> None:
        """Identical values should give non-significant result."""
        values = np.array([0.80, 0.82, 0.79, 0.81, 0.80])
        t_stat, p_val = paired_ttest(values, values.copy())
        assert math.isnan(t_stat)
        assert math.isnan(p_val)

    @pytest.mark.unit
    def test_too_few_samples(self) -> None:
        """n < 2 should return nan."""
        t_stat, p_val = paired_ttest(np.array([0.5]), np.array([0.6]))
        assert math.isnan(t_stat)
        assert math.isnan(p_val)

    @pytest.mark.unit
    def test_mismatched_lengths(self) -> None:
        """Different array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            paired_ttest(np.array([0.5, 0.6]), np.array([0.5, 0.6, 0.7]))

    @pytest.mark.unit
    def test_agrees_with_scipy(self) -> None:
        """Verify our wrapper matches scipy directly."""
        rng = np.random.default_rng(42)
        bl = rng.normal(0.7, 0.05, size=10)
        tr = bl + rng.normal(0.03, 0.01, size=10)

        t_stat, p_val = paired_ttest(bl, tr)
        # paired_ttest computes treatment - baseline, matching scipy(tr, bl)
        expected_t, expected_p = stats.ttest_rel(tr, bl)

        assert t_stat == pytest.approx(float(expected_t))
        assert p_val == pytest.approx(float(expected_p))

    @pytest.mark.unit
    def test_list_input(self) -> None:
        """Should accept list inputs (converted to arrays internally)."""
        bl = [0.72, 0.68, 0.75, 0.70, 0.66]
        tr = [0.78, 0.75, 0.77, 0.76, 0.71]
        t_stat, p_val = paired_ttest(bl, tr)  # type: ignore[arg-type]
        assert math.isfinite(t_stat)
        assert math.isfinite(p_val)


class TestCohensDPaired:
    """Tests for cohens_d_paired."""

    @pytest.mark.unit
    def test_known_large_effect(self) -> None:
        """Clear improvement should give large effect size."""
        baseline = np.array([0.72, 0.68, 0.75, 0.70, 0.66])
        synthetic = np.array([0.78, 0.75, 0.77, 0.76, 0.71])

        d = cohens_d_paired(baseline, synthetic)
        # d should be positive (treatment > baseline) and large
        assert d > 0
        assert interpret_effect_size(d) == "large"

    @pytest.mark.unit
    def test_hedges_correction_reduces_magnitude(self) -> None:
        """Hedges' g should be smaller in magnitude than raw Cohen's d for small n."""
        baseline = np.array([0.70, 0.72, 0.68, 0.71, 0.69])
        treatment = np.array([0.75, 0.77, 0.73, 0.76, 0.74])

        diffs = treatment - baseline
        raw_d = float(np.mean(diffs)) / float(np.std(diffs, ddof=1))
        hedges_g = cohens_d_paired(baseline, treatment)

        # Correction factor < 1 for small n, so |g| < |d|
        assert abs(hedges_g) < abs(raw_d)

    @pytest.mark.unit
    def test_zero_effect(self) -> None:
        """No difference should give d ≈ 0 (with some noise)."""
        rng = np.random.default_rng(42)
        values = rng.normal(0.75, 0.05, size=100)
        d = cohens_d_paired(values, values + rng.normal(0, 0.0001, size=100))
        assert abs(d) < 0.5  # negligible or small

    @pytest.mark.unit
    def test_identical_values_returns_nan(self) -> None:
        """Zero variance in differences should return nan."""
        values = np.array([0.80, 0.80, 0.80, 0.80, 0.80])
        d = cohens_d_paired(values, values)
        assert math.isnan(d)

    @pytest.mark.unit
    def test_too_few_samples(self) -> None:
        """n < 2 should return nan."""
        d = cohens_d_paired(np.array([0.5]), np.array([0.6]))
        assert math.isnan(d)

    @pytest.mark.unit
    def test_mismatched_lengths(self) -> None:
        """Different array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            cohens_d_paired(np.array([0.5, 0.6]), np.array([0.5]))

    @pytest.mark.unit
    def test_negative_effect(self) -> None:
        """Treatment worse than baseline should give negative d."""
        baseline = np.array([0.80, 0.82, 0.79, 0.81, 0.80])
        treatment = np.array([0.70, 0.72, 0.69, 0.71, 0.70])
        d = cohens_d_paired(baseline, treatment)
        assert d < 0

    @pytest.mark.unit
    def test_n2_returns_uncorrected(self) -> None:
        """n=2 should return uncorrected Cohen's d (correction not applicable)."""
        baseline = np.array([0.60, 0.70])
        treatment = np.array([0.80, 0.85])
        d = cohens_d_paired(baseline, treatment)
        # For n=2 (df=1), Hedges' correction is undefined, so raw d is returned.
        # diffs = [0.20, 0.15], raw d = mean/std = 0.175/0.0354 ≈ 4.95
        assert math.isfinite(d)
        assert abs(d) > 1.0  # should be large, not zeroed out

    @pytest.mark.unit
    def test_n3_uses_exact_hedges(self) -> None:
        """n≥3 should use exact Hedges' correction (smaller than raw d)."""
        baseline = np.array([0.70, 0.72, 0.68])
        treatment = np.array([0.80, 0.82, 0.78])
        d = cohens_d_paired(baseline, treatment)

        diffs = treatment - baseline
        raw_d = float(np.mean(diffs)) / float(np.std(diffs, ddof=1))
        # Exact correction for df=2 is ~0.564, so |g| < |raw_d|
        assert abs(d) < abs(raw_d)
        assert abs(d) > 0


class TestInterpretEffectSize:
    """Tests for interpret_effect_size."""

    @pytest.mark.unit
    def test_negligible(self) -> None:
        assert interpret_effect_size(0.0) == "negligible"
        assert interpret_effect_size(0.19) == "negligible"
        assert interpret_effect_size(-0.1) == "negligible"

    @pytest.mark.unit
    def test_small(self) -> None:
        assert interpret_effect_size(0.2) == "small"
        assert interpret_effect_size(0.49) == "small"
        assert interpret_effect_size(-0.3) == "small"

    @pytest.mark.unit
    def test_medium(self) -> None:
        assert interpret_effect_size(0.5) == "medium"
        assert interpret_effect_size(0.79) == "medium"
        assert interpret_effect_size(-0.6) == "medium"

    @pytest.mark.unit
    def test_large(self) -> None:
        assert interpret_effect_size(0.8) == "large"
        assert interpret_effect_size(2.0) == "large"
        assert interpret_effect_size(-1.5) == "large"

    @pytest.mark.unit
    def test_nan(self) -> None:
        assert interpret_effect_size(float("nan")) == "undefined"

    @pytest.mark.unit
    def test_inf(self) -> None:
        assert interpret_effect_size(float("inf")) == "undefined"


class TestAdjustPvalues:
    """Tests for adjust_pvalues."""

    @pytest.mark.unit
    def test_bonferroni(self) -> None:
        """Bonferroni multiplies by n and caps at 1.0."""
        pvals = [0.01, 0.04, 0.5]
        adjusted = adjust_pvalues(pvals, method="bonferroni")
        assert adjusted[0] == pytest.approx(0.03)
        assert adjusted[1] == pytest.approx(0.12)
        assert adjusted[2] == pytest.approx(1.0)  # capped at 1.0

    @pytest.mark.unit
    def test_benjamini_hochberg(self) -> None:
        """BH correction with known values."""
        pvals = [0.01, 0.04, 0.03, 0.20]
        adjusted = adjust_pvalues(pvals, method="benjamini-hochberg")

        # All adjusted values should be >= raw values
        for raw, adj in zip(pvals, adjusted):
            assert adj >= raw or adj == pytest.approx(raw)

        # Adjusted values should be <= 1.0
        for adj in adjusted:
            assert adj <= 1.0

    @pytest.mark.unit
    def test_bh_monotonicity(self) -> None:
        """BH-adjusted p-values should maintain rank order."""
        pvals = [0.001, 0.01, 0.02, 0.05, 0.10]
        adjusted = adjust_pvalues(pvals, method="benjamini-hochberg")
        sorted_raw = sorted(range(len(pvals)), key=lambda i: pvals[i])
        for i in range(len(sorted_raw) - 1):
            assert adjusted[sorted_raw[i]] <= adjusted[sorted_raw[i + 1]]

    @pytest.mark.unit
    def test_single_pvalue(self) -> None:
        """Single p-value should be unchanged."""
        adjusted = adjust_pvalues([0.05], method="benjamini-hochberg")
        assert adjusted[0] == pytest.approx(0.05)

    @pytest.mark.unit
    def test_empty_list(self) -> None:
        assert adjust_pvalues([], method="bonferroni") == []
        assert adjust_pvalues([], method="benjamini-hochberg") == []

    @pytest.mark.unit
    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown correction method"):
            adjust_pvalues([0.05], method="invalid")

    @pytest.mark.unit
    def test_bh_preserves_original_order(self) -> None:
        """Adjusted values should correspond to the same indices as input."""
        pvals = [0.05, 0.001, 0.10]  # Not sorted
        adjusted = adjust_pvalues(pvals, method="benjamini-hochberg")
        # The smallest raw p-value (index 1) should have the smallest adjusted
        assert adjusted[1] <= adjusted[0]
        assert adjusted[1] <= adjusted[2]


class TestCompareExperimentPair:
    """Tests for compare_experiment_pair."""

    @pytest.mark.unit
    def test_significant_improvement(self) -> None:
        """Clear improvement should be detected as significant."""
        baseline = {
            "recall_1": np.array([0.72, 0.68, 0.75, 0.70, 0.66]),
            "f1_1": np.array([0.70, 0.66, 0.73, 0.68, 0.64]),
        }
        treatment = {
            "recall_1": np.array([0.78, 0.75, 0.77, 0.76, 0.71]),
            "f1_1": np.array([0.76, 0.73, 0.75, 0.74, 0.69]),
        }

        results = compare_experiment_pair(
            baseline, treatment, ["recall_1", "f1_1"], alpha=0.05
        )

        assert len(results) == 2
        for r in results:
            assert isinstance(r, ComparisonResult)
            assert r.mean_diff > 0
            assert r.cohens_d > 0
            assert r.p_value < 0.05

    @pytest.mark.unit
    def test_no_improvement(self) -> None:
        """Random noise should not be significant."""
        rng = np.random.default_rng(42)
        values = rng.normal(0.75, 0.05, size=5)
        baseline = {"metric": values}
        treatment = {"metric": values + rng.normal(0, 0.001, size=5)}

        results = compare_experiment_pair(baseline, treatment, ["metric"], alpha=0.05)

        assert len(results) == 1
        assert not results[0].significant

    @pytest.mark.unit
    def test_missing_metric_skipped(self) -> None:
        """Missing metric in one experiment should be skipped."""
        baseline = {"recall_1": np.array([0.7, 0.8])}
        treatment = {"f1_1": np.array([0.7, 0.8])}

        results = compare_experiment_pair(
            baseline, treatment, ["recall_1", "f1_1"], alpha=0.05
        )
        assert len(results) == 0

    @pytest.mark.unit
    def test_result_fields(self) -> None:
        """Verify all ComparisonResult fields are populated correctly."""
        bl = np.array([0.70, 0.72, 0.68, 0.71, 0.69])
        tr = np.array([0.80, 0.82, 0.78, 0.81, 0.79])

        results = compare_experiment_pair({"m": bl}, {"m": tr}, ["m"], alpha=0.05)

        r = results[0]
        assert r.metric == "m"
        assert r.baseline_mean == pytest.approx(np.mean(bl))
        assert r.treatment_mean == pytest.approx(np.mean(tr))
        assert r.mean_diff == pytest.approx(np.mean(tr) - np.mean(bl))
        assert r.effect_size_interpretation in {
            "negligible",
            "small",
            "medium",
            "large",
        }
        assert isinstance(r.significant, bool)

    @pytest.mark.unit
    def test_correction_method_bonferroni(self) -> None:
        """Bonferroni correction should be more conservative than BH."""
        bl = {
            "m1": np.array([0.70, 0.72, 0.68, 0.71, 0.69]),
            "m2": np.array([0.60, 0.62, 0.58, 0.61, 0.59]),
        }
        tr = {
            "m1": np.array([0.75, 0.77, 0.73, 0.76, 0.74]),
            "m2": np.array([0.65, 0.67, 0.63, 0.66, 0.64]),
        }

        results_bh = compare_experiment_pair(
            bl, tr, ["m1", "m2"], correction_method="benjamini-hochberg"
        )
        results_bf = compare_experiment_pair(
            bl, tr, ["m1", "m2"], correction_method="bonferroni"
        )

        # Bonferroni corrected p-values should be >= BH corrected
        for bh, bf in zip(results_bh, results_bf):
            assert bf.p_value_corrected >= bh.p_value_corrected or (
                math.isnan(bf.p_value_corrected) and math.isnan(bh.p_value_corrected)
            )

    @pytest.mark.unit
    def test_nan_pvalues_handled(self) -> None:
        """Identical baseline/treatment should produce nan p-values gracefully."""
        values = np.array([0.80, 0.80, 0.80, 0.80, 0.80])
        results = compare_experiment_pair(
            {"m": values}, {"m": values.copy()}, ["m"], alpha=0.05
        )

        assert len(results) == 1
        assert math.isnan(results[0].p_value)
        assert not results[0].significant
