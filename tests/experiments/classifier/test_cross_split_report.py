"""Tests for the cross-split robustness report.

Verifies the split-as-unit aggregation: per-split seed-means, across-split
t-CIs, and paired-by-split treatment-vs-baseline comparisons. The key
statistical contract is that the experimental unit is the SPLIT (not the
(split, seed) pair) and that within-split seed std is never pooled.
"""

import json

import numpy as np
import pytest

from src.experiments.classifier.cross_split_report import (
    _parse_split_label,
    _select_baseline,
    _t_ci,
    build_across_split_summary,
    compute_cross_split_comparisons,
    compute_split_means,
    generate_cross_split_report,
    load_per_split_results,
)
from src.experiments.classifier.evaluation_report import key_metrics


@pytest.mark.unit
class TestPureHelpers:
    """Unit coverage for the pure (filesystem-free) building blocks."""

    def test_parse_split_label(self):
        assert _parse_split_label("outputs/multisplit/split7/binary-depth") == 7
        assert _parse_split_label("a/b/c") is None

    def test_t_ci_known(self):
        mean, lo, hi = _t_ci(np.array([0.70, 0.72, 0.74, 0.76]))
        assert mean == pytest.approx(0.73)
        assert lo < mean < hi

    def test_t_ci_single_value_nan_bounds(self):
        mean, lo, hi = _t_ci(np.array([0.7]))
        assert mean == 0.7
        assert np.isnan(lo) and np.isnan(hi)

    def test_compute_split_means_from_dicts(self):
        # Two experiments, two seeds each -> seed-averaged per metric.
        results = [
            {"experiment": "baseline__ws", "seed": 0, "recall_1": 0.70, "pr_auc": 0.85},
            {"experiment": "baseline__ws", "seed": 1, "recall_1": 0.72, "pr_auc": 0.87},
            {
                "experiment": "ft-mixed67__ws",
                "seed": 0,
                "recall_1": 0.80,
                "pr_auc": 0.90,
            },
            {
                "experiment": "ft-mixed67__ws",
                "seed": 1,
                "recall_1": 0.82,
                "pr_auc": 0.92,
            },
        ]
        means = compute_split_means(results, ["recall_1", "pr_auc"])
        assert means["baseline__ws"]["recall_1"] == pytest.approx(0.71)
        assert means["ft-mixed67__ws"]["pr_auc"] == pytest.approx(0.91)

    def test_compute_split_means_empty(self):
        assert compute_split_means([], ["recall_1"]) == {}

    def test_compute_split_means_nan_tolerant(self):
        # A NaN minority metric in ONE seed must NOT drop the whole split's
        # contribution -- the mean is taken over the finite seeds only. This is
        # the honest-CI contract: pr_auc / hardcore_pr_auc_* are legitimately
        # NaN on a degenerate restricted set for some seeds.
        results = [
            {"experiment": "baseline__ws", "seed": 0, "recall_1": 0.70, "pr_auc": 0.85},
            {
                "experiment": "baseline__ws",
                "seed": 1,
                "recall_1": 0.72,
                "pr_auc": float("nan"),
            },
        ]
        means = compute_split_means(results, ["recall_1", "pr_auc"])
        # recall has both seeds; pr_auc keeps the single finite seed (0.85),
        # rather than the split losing pr_auc entirely.
        assert means["baseline__ws"]["recall_1"] == pytest.approx(0.71)
        assert means["baseline__ws"]["pr_auc"] == pytest.approx(0.85)

    def test_select_baseline_explicit(self):
        split_means = {
            0: {"baseline__ws": {"recall_1": 0.7}, "baseline__us": {"recall_1": 0.8}}
        }
        assert (
            _select_baseline(split_means, "baseline__ws", ["recall_1"], 1)
            == "baseline__ws"
        )

    def test_select_baseline_auto_highest_recall(self):
        split_means = {
            0: {"baseline__ws": {"recall_1": 0.7}, "baseline__us": {"recall_1": 0.8}}
        }
        # No explicit name -> pick the baseline with the highest mean recall.
        assert _select_baseline(split_means, None, ["recall_1"], 1) == "baseline__us"

    def test_select_baseline_none_when_no_baseline(self):
        split_means = {0: {"ft-mixed67__ws": {"recall_1": 0.8}}}
        assert _select_baseline(split_means, None, ["recall_1"], 1) is None


def _write_eval(base, split, family, exp, seed, metrics):
    """Write one evaluation.json at the canonical split/family/exp/seed path."""
    reports = base / f"split{split}" / family / exp / f"seed{seed}" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / "evaluation.json", "w") as f:
        json.dump(metrics, f)


def _make_tree(base, family="binary-depth", n_splits=4, n_seeds=3):
    """Build a small multi-split tree with a baseline and one transfer variant.

    The transfer variant is offset above baseline by a fixed per-split amount so
    paired-by-split tests have a clean, detectable signal.
    """
    rng = np.random.default_rng(0)
    for s in range(n_splits):
        # Per-split level shifts so splits genuinely differ (data variance).
        bl_level = 0.70 + 0.01 * s
        tr_level = bl_level + 0.05  # transfer consistently better
        for seed in range(n_seeds):
            bl = {
                "recall_1": bl_level + rng.normal(0, 0.002),
                "pr_auc": 0.85 + 0.01 * s + rng.normal(0, 0.002),
                "balanced_accuracy": 0.75,
                "f1_1": 0.70,
                "roc_auc": 0.9,
                "accuracy": 80.0,
                "precision_1": 0.6,
                "loss": 0.3,
                "positive_class": 1,
                "num_classes": 2,
            }
            tr = dict(bl)
            tr["recall_1"] = tr_level + rng.normal(0, 0.002)
            tr["pr_auc"] = 0.90 + 0.01 * s + rng.normal(0, 0.002)
            _write_eval(base, s, family, "baseline__ws", seed, bl)
            _write_eval(base, s, family, "ft-mixed67__ws", seed, tr)


@pytest.mark.component
class TestLoadPerSplit:
    def test_groups_by_split(self, tmp_path):
        _make_tree(tmp_path, n_splits=4, n_seeds=3)
        per_split = load_per_split_results(str(tmp_path), "binary-depth")
        assert sorted(per_split) == [0, 1, 2, 3]
        # 2 experiments x 3 seeds per split
        assert len(per_split[0]) == 6
        assert all("seed" in r for r in per_split[0])

    def test_missing_family_returns_empty(self, tmp_path):
        _make_tree(tmp_path)
        assert load_per_split_results(str(tmp_path), "nonexistent") == {}


@pytest.mark.component
class TestComputeSplitMeans:
    def test_averages_over_seeds(self, tmp_path):
        _make_tree(tmp_path, n_splits=1, n_seeds=3)
        per_split = load_per_split_results(str(tmp_path), "binary-depth")
        means = compute_split_means(per_split[0], key_metrics(1))
        assert "baseline__ws" in means
        assert "ft-mixed67__ws" in means
        # transfer recall (~0.75) should exceed baseline recall (~0.70)
        assert means["ft-mixed67__ws"]["recall_1"] > means["baseline__ws"]["recall_1"]


@pytest.mark.unit
class TestAcrossSplitSummary:
    def test_ci_uses_split_count_as_n(self):
        # 4 splits, baseline recall increasing 0.70..0.73 -> mean 0.715
        split_means = {
            s: {"baseline__ws": {"recall_1": 0.70 + 0.01 * s}} for s in range(4)
        }
        summary = build_across_split_summary(split_means, ["recall_1"])
        row = summary[summary["metric"] == "recall_1"].iloc[0]
        assert row["n_splits"] == 4  # unit is the split, not (split, seed)
        assert row["mean"] == pytest.approx(0.715, abs=1e-9)
        assert row["ci_lower"] < row["mean"] < row["ci_upper"]

    def test_single_split_has_nan_ci(self):
        split_means = {0: {"baseline__ws": {"recall_1": 0.7}}}
        summary = build_across_split_summary(split_means, ["recall_1"])
        row = summary.iloc[0]
        assert row["n_splits"] == 1
        assert np.isnan(row["ci_lower"])


@pytest.mark.unit
class TestCrossSplitComparisons:
    def test_paired_by_split_detects_consistent_gain(self):
        # transfer beats baseline by ~+0.05 in every split (with slight,
        # realistic per-split variation) -> significant
        gains = [0.048, 0.052, 0.050, 0.055, 0.047, 0.051]
        split_means = {}
        for s in range(6):
            bl = 0.70 + 0.01 * s
            split_means[s] = {
                "baseline__ws": {"recall_1": bl},
                "ft-mixed67__ws": {"recall_1": bl + gains[s]},
            }
        comp = compute_cross_split_comparisons(
            split_means, "baseline__ws", ["recall_1"]
        )
        row = comp[comp["treatment"] == "ft-mixed67__ws"].iloc[0]
        assert row["n_splits"] == 6
        assert row["mean_diff"] == pytest.approx(np.mean(gains), abs=1e-9)
        assert row["significant"]  # paired-t BH-corrected
        assert row["p_value_wilcoxon"] < 0.05  # 6 pairs reaches Wilcoxon floor

    def test_constant_shift_is_significant_via_wilcoxon_fallback(self):
        # A perfectly constant non-zero improvement in every split has zero
        # variance, so paired-t is degenerate (NaN). That is the STRONGEST
        # evidence, not the weakest -- significance must fall back to the
        # BH-corrected Wilcoxon p rather than being reported as non-significant.
        split_means = {}
        for s in range(6):
            bl = 0.70 + 0.01 * s
            split_means[s] = {
                "baseline__ws": {"recall_1": bl},
                "ft-mixed67__ws": {"recall_1": bl + 0.05},  # exact constant shift
            }
        comp = compute_cross_split_comparisons(
            split_means, "baseline__ws", ["recall_1"]
        )
        row = comp[comp["treatment"] == "ft-mixed67__ws"].iloc[0]
        assert np.isnan(row["p_value_ttest"])  # zero-variance -> t undefined
        assert row["p_value_wilcoxon"] < 0.05  # 6 identical-sign pairs
        assert row["significant"]  # Wilcoxon fallback, not a false negative

    def test_no_treatments_returns_empty(self):
        split_means = {s: {"baseline__ws": {"recall_1": 0.7}} for s in range(3)}
        comp = compute_cross_split_comparisons(
            split_means, "baseline__ws", ["recall_1"]
        )
        assert comp.empty


@pytest.mark.component
class TestGenerateReport:
    def test_writes_markdown_and_csv(self, tmp_path):
        _make_tree(tmp_path, n_splits=4, n_seeds=3)
        out = tmp_path / "cross_split" / "binary-depth"
        generate_cross_split_report(
            base_dir=str(tmp_path),
            family="binary-depth",
            output_dir=str(out),
            baseline_name="baseline__ws",
        )
        assert (out / "cross_split_report.md").exists()
        assert (out / "cross_split_summary.csv").exists()
        assert (out / "cross_split_comparisons.csv").exists()
        text = (out / "cross_split_report.md").read_text()
        assert "Cross-Split Robustness Report" in text
        assert "Splits (experimental unit): 4" in text
        assert "baseline__ws" in text

    def test_no_results_is_graceful(self, tmp_path):
        out = tmp_path / "out"
        generate_cross_split_report(
            base_dir=str(tmp_path), family="missing", output_dir=str(out)
        )
        # Nothing written when there is no data.
        assert not (out / "cross_split_report.md").exists()
