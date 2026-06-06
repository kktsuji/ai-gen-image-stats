"""Tests for the hard-core direct-evaluation report (post-hoc)."""

import json

import numpy as np
import pytest

from src.experiments.classifier import hard_core_analysis as hca

CLASS_NAMES = ["abnormal", "suspicious", "red"]


def _write_run(base, exp, seed, rng, *, n_per_class=8, stamp_contrast=True):
    """Create one run's predictions_test.npz + evaluation.json (3-class)."""
    reports = base / exp / f"seed{seed}" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    targets = np.repeat([0, 1, 2], n_per_class)
    logits = rng.gamma(1.0, size=(targets.size, 3))
    # Bias each sample toward its true class so the metric is non-degenerate.
    logits[np.arange(targets.size), targets] += rng.uniform(0.5, 2.0, size=targets.size)
    probs = logits / logits.sum(axis=1, keepdims=True)

    np.savez_compressed(
        reports / "predictions_test.npz",
        targets=targets,
        predictions=probs.argmax(axis=1),
        probs=probs,
    )

    payload = {
        "split": "test",
        "num_classes": 3,
        "class_names": CLASS_NAMES,
        "positive_class": 0,
        "recall_0": 0.8,
    }
    if stamp_contrast:
        payload["contrast_class"] = 1
    with open(reports / "evaluation.json", "w") as f:
        json.dump(payload, f)


@pytest.mark.component
class TestGenerateReport:
    def test_end_to_end_with_stamped_contrast(self, tmp_path):
        base = tmp_path / "classifier"
        rng = np.random.RandomState(0)
        for exp in ("baseline__ws", "ft-mixed67__ws"):
            for seed in range(3):
                _write_run(base, exp, seed, rng)

        out = tmp_path / "hardcore_out"
        hca.generate_report(
            base_dir=str(base), output_dir=str(out), baseline_name="baseline__ws"
        )

        md = (out / "hard_core_analysis.md").read_text()
        assert "Hard-Core Direct Evaluation" in md
        assert "contrast class (suspicious) = 1" in md
        assert "0.915" in md  # binary oracle reference
        assert "ft-mixed67__ws" in md
        assert (out / "hard_core_analysis.csv").exists()

    def test_contrast_autodetect_from_class_names(self, tmp_path):
        base = tmp_path / "classifier"
        rng = np.random.RandomState(1)
        for seed in range(2):
            _write_run(base, "baseline__ws", seed, rng, stamp_contrast=False)

        out = tmp_path / "hardcore_out"
        hca.generate_report(base_dir=str(base), output_dir=str(out))
        md = (out / "hard_core_analysis.md").read_text()
        # "suspicious" resolved by name lookup -> index 1.
        assert "contrast class (suspicious) = 1" in md

    def test_significance_table_present(self, tmp_path):
        base = tmp_path / "classifier"
        rng = np.random.RandomState(2)
        for exp in ("baseline__ws", "ft-mixed67__ws"):
            for seed in range(4):
                _write_run(base, exp, seed, rng)

        out = tmp_path / "hardcore_out"
        hca.generate_report(
            base_dir=str(base), output_dir=str(out), baseline_name="baseline__ws"
        )
        md = (out / "hard_core_analysis.md").read_text()
        assert "Significance vs baseline" in md
        assert "Baseline: **baseline__ws**" in md

    def test_no_predictions_aborts_cleanly(self, tmp_path):
        base = tmp_path / "classifier"
        base.mkdir()
        out = tmp_path / "hardcore_out"
        # No runs at all -> resolve fails / no rows -> no files written, no raise.
        hca.generate_report(base_dir=str(base), output_dir=str(out))
        assert not (out / "hard_core_analysis.md").exists()

    def test_unresolvable_contrast_aborts(self, tmp_path):
        base = tmp_path / "classifier"
        # class_names without "suspicious" and no stamped contrast -> abort.
        reports = base / "baseline__ws" / "seed0" / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        targets = np.array([0, 0, 1, 1])
        probs = np.array([[0.6, 0.4], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]], dtype=float)
        np.savez_compressed(
            reports / "predictions_test.npz",
            targets=targets,
            predictions=probs.argmax(axis=1),
            probs=probs,
        )
        with open(reports / "evaluation.json", "w") as f:
            json.dump(
                {
                    "split": "test",
                    "num_classes": 2,
                    "class_names": ["normal", "abnormal"],
                    "positive_class": 1,
                },
                f,
            )

        out = tmp_path / "hardcore_out"
        hca.generate_report(base_dir=str(base), output_dir=str(out))
        assert not (out / "hard_core_analysis.md").exists()


@pytest.mark.unit
class TestHelpers:
    def test_load_npz_full_roundtrip(self, tmp_path):
        probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
        targets = np.array([0, 1])
        np.savez_compressed(
            tmp_path / "predictions_test.npz",
            targets=targets,
            predictions=probs.argmax(axis=1),
            probs=probs,
        )
        loaded = hca._load_npz_full(tmp_path / "predictions_test.npz")
        assert loaded is not None
        got_targets, got_probs = loaded
        np.testing.assert_array_equal(got_targets, targets)
        np.testing.assert_allclose(got_probs, probs)

    def test_detect_contrast_prefers_stamped(self):
        results = [{"contrast_class": 1, "class_names": CLASS_NAMES}]
        assert hca.detect_contrast_class(results, positive_class=0) == 1

    def test_detect_contrast_falls_back_to_name(self):
        results = [{"class_names": CLASS_NAMES}]
        assert hca.detect_contrast_class(results, positive_class=0) == 1

    def test_detect_contrast_override(self):
        results = [{"contrast_class": 1, "class_names": CLASS_NAMES}]
        assert hca.detect_contrast_class(results, 0, override=2) == 2

    def test_detect_contrast_unresolvable(self):
        results = [{"class_names": ["normal", "abnormal"]}]
        assert hca.detect_contrast_class(results, positive_class=1) is None
