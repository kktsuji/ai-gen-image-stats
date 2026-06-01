"""Tests for the post-hoc decision-threshold analysis."""

import numpy as np
import pytest

from src.experiments.classifier import threshold_analysis as ta


@pytest.mark.unit
class TestSelectThreshold:
    def test_max_f1_recovers_separating_threshold(self):
        # Class 0 probs cluster near 0.2, class 1 near 0.8; the optimal threshold
        # for F1 should fall between the clusters.
        rng = np.random.RandomState(0)
        probs = np.concatenate([rng.uniform(0.0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
        targets = np.array([0] * 50 + [1] * 50)
        tau = ta.select_threshold(probs, targets, criterion="max_f1_1")
        # The selected threshold lands in the separating gap and yields ~perfect F1.
        assert 0.3 <= tau <= 0.65
        assert ta.metrics_at_threshold(probs, targets, tau)["f1_1"] >= 0.95

    def test_precision_at_recall_respects_floor(self):
        rng = np.random.RandomState(1)
        probs = np.concatenate([rng.uniform(0.0, 0.5, 50), rng.uniform(0.5, 1.0, 50)])
        targets = np.array([0] * 50 + [1] * 50)
        tau = ta.select_threshold(
            probs, targets, criterion="precision_at_recall", target_recall=0.9
        )
        # At the chosen threshold, recall on the positive class must meet the floor.
        recall = ta.metrics_at_threshold(probs, targets, tau)["recall_1"]
        assert recall >= 0.9

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            ta.select_threshold(np.array([0.5]), np.array([1]), criterion="bogus")


@pytest.mark.unit
class TestMetricsAtThreshold:
    def test_hand_computed_confusion(self):
        # probs_pos: 0.9, 0.8, 0.2, 0.1 ; targets: 1,0,1,0 ; tau=0.5
        # preds: 1,1,0,0 -> TP=1 (idx0), FP=1 (idx1), FN=1 (idx2), TN=1 (idx3)
        probs = np.array([0.9, 0.8, 0.2, 0.1])
        targets = np.array([1, 0, 1, 0])
        m = ta.metrics_at_threshold(probs, targets, 0.5)
        assert m["precision_1"] == pytest.approx(0.5)  # 1 / (1 + 1)
        assert m["recall_1"] == pytest.approx(0.5)  # 1 / (1 + 1)
        assert m["f1_1"] == pytest.approx(0.5)
        assert m["balanced_accuracy"] == pytest.approx(0.5)

    def test_threshold_zero_predicts_all_positive(self):
        probs = np.array([0.1, 0.9])
        targets = np.array([0, 1])
        m = ta.metrics_at_threshold(probs, targets, 0.0)
        assert m["recall_1"] == pytest.approx(1.0)


def _write_predictions(reports_dir, split, probs_pos, targets):
    reports_dir.mkdir(parents=True, exist_ok=True)
    probs = np.stack([1.0 - probs_pos, probs_pos], axis=1)
    np.savez_compressed(
        reports_dir / f"predictions_{split}.npz",
        targets=np.asarray(targets),
        predictions=(probs_pos >= 0.5).astype(int),
        probs=probs,
    )


@pytest.mark.component
class TestGenerateReport:
    def test_val_to_test_end_to_end(self, tmp_path):
        base = tmp_path / "classifier"
        rng = np.random.RandomState(2)
        for seed in range(2):
            reports = base / "ft-full__ws" / f"seed{seed}" / "reports"
            val_pos = np.concatenate(
                [rng.uniform(0, 0.4, 20), rng.uniform(0.6, 1.0, 20)]
            )
            test_pos = np.concatenate(
                [rng.uniform(0, 0.4, 20), rng.uniform(0.6, 1.0, 20)]
            )
            labels = np.array([0] * 20 + [1] * 20)
            _write_predictions(reports, "val", val_pos, labels)
            _write_predictions(reports, "test", test_pos, labels)

        out = tmp_path / "threshold_out"
        ta.generate_report(base_dir=str(base), output_dir=str(out))

        md = (out / "threshold_analysis.md").read_text()
        assert "Decision-Threshold Analysis" in md
        assert "ft-full__ws" in md
        assert (out / "threshold_results.csv").exists()

    def test_in_sample_fallback_reads_legacy_predictions(self, tmp_path):
        # Archived runs save a single unsuffixed predictions.npz (test only). The
        # --in-sample mode must read it via the fallback.
        base = tmp_path / "classifier"
        reports = base / "ft-full__ws" / "seed0" / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        pos = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        probs = np.stack([1.0 - pos, pos], axis=1)
        np.savez_compressed(
            reports / "predictions.npz",
            targets=labels,
            predictions=(pos >= 0.5).astype(int),
            probs=probs,
        )

        out = tmp_path / "probe"
        ta.generate_report(base_dir=str(base), output_dir=str(out), in_sample=True)
        md = (out / "threshold_analysis.md").read_text()
        assert "IN-SAMPLE / LEAKY" in md
        assert "ft-full__ws" in md

    def test_val_to_test_skips_when_no_val(self, tmp_path):
        # Without val predictions, the leak-free protocol must skip the run (no rows).
        base = tmp_path / "classifier"
        reports = base / "ft-full__ws" / "seed0" / "reports"
        _write_predictions(reports, "test", np.array([0.1, 0.9]), np.array([0, 1]))

        rows = ta.load_prediction_rows(
            str(base),
            in_sample=False,
            criterion="max_f1_1",
            target_recall=0.9,
            grid_points=51,
        )
        assert rows == []
