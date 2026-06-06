"""Tests for the hard-core direct-evaluation metric (pure functions)."""

import numpy as np
import pytest

from src.experiments.classifier import hard_core as hc


@pytest.mark.unit
class TestResolveContrastClass:
    def test_override_wins(self):
        assert hc.resolve_contrast_class(["abnormal", "suspicious"], override=0) == 0

    def test_name_lookup(self):
        names = ["red", "suspicious", "abnormal"]
        assert hc.resolve_contrast_class(names) == 1

    def test_custom_name(self):
        names = ["red", "borderline", "abnormal"]
        assert hc.resolve_contrast_class(names, name="borderline") == 1

    def test_missing_name_returns_none(self):
        assert hc.resolve_contrast_class(["red", "abnormal"]) is None

    def test_empty_class_names_returns_none(self):
        assert hc.resolve_contrast_class([]) is None
        assert hc.resolve_contrast_class(None) is None


@pytest.mark.unit
class TestComputeHardCoreMetrics:
    def test_renorm_outranks_raw(self):
        # 3 classes: abn=0, susp=1, other=2. Restricted set = targets in {0,1}.
        # Renormalizing by P(abn)+P(susp) perfectly separates abn from susp,
        # while the raw P(abn) ranking interleaves them.
        targets = np.array([0, 0, 1, 1])
        probs = np.array(
            [
                [0.5, 0.1, 0.4],  # abn, p_abn=0.5, renorm=0.833
                [0.2, 0.1, 0.7],  # abn, p_abn=0.2, renorm=0.667
                [0.4, 0.5, 0.1],  # susp, p_abn=0.4, renorm=0.444
                [0.1, 0.8, 0.1],  # susp, p_abn=0.1, renorm=0.111
            ]
        )
        m = hc.compute_hard_core_metrics(
            targets, probs, positive_class=0, contrast_class=1
        )
        assert m["hardcore_n"] == 4
        assert m["hardcore_pr_auc_renorm"] == pytest.approx(1.0)
        assert m["hardcore_roc_auc_renorm"] == pytest.approx(1.0)
        # raw ranking is [1,0,1,0] by score desc -> AP = (1 + 2/3) / 2.
        assert m["hardcore_pr_auc_raw"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
        assert m["hardcore_pr_auc_raw"] < m["hardcore_pr_auc_renorm"]

    def test_renorm_equals_raw_when_other_mass_zero(self):
        # With no probability mass on the third class, renorm == raw (the
        # denominator is 1), so both PR-AUCs match.
        targets = np.array([0, 0, 1, 1])
        probs = np.array(
            [
                [0.7, 0.3, 0.0],
                [0.6, 0.4, 0.0],
                [0.45, 0.55, 0.0],
                [0.2, 0.8, 0.0],
            ]
        )
        m = hc.compute_hard_core_metrics(targets, probs, 0, 1)
        assert m["hardcore_pr_auc_renorm"] == pytest.approx(m["hardcore_pr_auc_raw"])

    def test_leak_rate(self):
        # 3 true-abnormal samples; argmax lands on contrast (susp=1) for 2 of 3.
        targets = np.array([0, 0, 0, 1])
        probs = np.array(
            [
                [0.1, 0.8, 0.1],  # argmax susp -> leak
                [0.8, 0.1, 0.1],  # argmax abn
                [0.2, 0.7, 0.1],  # argmax susp -> leak
                [0.1, 0.8, 0.1],  # susp sample
            ]
        )
        m = hc.compute_hard_core_metrics(targets, probs, 0, 1)
        assert m["hardcore_leak_rate"] == pytest.approx(2.0 / 3.0)

    def test_single_class_in_restricted_set_is_nan(self):
        # No suspicious samples -> restricted set is one class -> AP undefined,
        # but n and leak_rate are still reported.
        targets = np.array([0, 0, 2])
        probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.1, 0.7], [0.1, 0.1, 0.8]])
        m = hc.compute_hard_core_metrics(targets, probs, 0, 1)
        assert m["hardcore_n"] == 2
        assert np.isnan(m["hardcore_pr_auc_renorm"])
        assert np.isnan(m["hardcore_pr_auc_raw"])
        assert not np.isnan(m["hardcore_leak_rate"])

    def test_empty_restricted_set_is_nan(self):
        targets = np.array([2, 2, 2])
        probs = np.array([[0.1, 0.1, 0.8]] * 3)
        m = hc.compute_hard_core_metrics(targets, probs, 0, 1)
        assert m["hardcore_n"] == 0
        assert np.isnan(m["hardcore_pr_auc_renorm"])

    def test_out_of_range_indices_are_nan(self):
        targets = np.array([0, 1])
        probs = np.array([[0.6, 0.4], [0.3, 0.7]])
        m = hc.compute_hard_core_metrics(targets, probs, 0, 5)
        assert np.isnan(m["hardcore_pr_auc_renorm"])

    def test_equal_positive_and_contrast_is_nan(self):
        targets = np.array([0, 1])
        probs = np.array([[0.6, 0.4], [0.3, 0.7]])
        m = hc.compute_hard_core_metrics(targets, probs, 1, 1)
        assert np.isnan(m["hardcore_pr_auc_renorm"])
