"""Tests for Sample Selection Evaluator

Unit tests for the evaluate mode of sample selection, which computes
distribution-level metrics across real, generated, and selected image sets.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_features(n: int, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Create mock feature vectors."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float32)


def _make_evaluate_config(
    tmp_dir: Path,
    *,
    include_selected: bool = False,
) -> dict:
    """Create a valid evaluate config for testing."""
    config: dict = {
        "experiment": "sample_selection",
        "mode": "evaluate",
        "compute": {"device": "cpu", "seed": 42},
        "feature_extraction": {
            "model": "inceptionv3",
            "batch_size": 4,
            "image_size": 299,
            "num_workers": 0,
        },
        "data": {
            "real": {
                "source": "directory",
                "directory": str(tmp_dir / "real"),
            },
            "generated": {
                "directory": str(tmp_dir / "generated"),
            },
        },
        "evaluation": {"k": 3},
        "output": {
            "base_dir": str(tmp_dir / "output"),
            "subdirs": {
                "logs": "logs",
                "reports": "reports",
            },
        },
        "logging": {
            "console_level": "INFO",
            "file_level": "INFO",
        },
    }
    if include_selected:
        config["data"]["selected"] = {
            "split_file": str(tmp_dir / "accepted.json"),
            "split": "train",
        }
    return config


# ============================================================================
# Unit Tests
# ============================================================================


@pytest.mark.unit
class TestRunSampleSelectionEvaluate:
    """Test the evaluate mode pipeline."""

    @patch("src.experiments.sample_selection.evaluator.extract_features_from_loader")
    @patch("src.experiments.sample_selection.evaluator.create_feature_model")
    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_real_vs_generated_only(
        self,
        mock_simple_ds,
        mock_load_real,
        mock_create_model,
        mock_extract,
        tmp_output_dir,
    ):
        """Without data.selected, report has only real_vs_generated."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)

        # Mock datasets
        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=20)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=30)
        mock_simple_ds.return_value = mock_gen_ds

        mock_create_model.return_value = MagicMock()

        real_feats = _make_mock_features(20, seed=1)
        gen_feats = _make_mock_features(30, seed=2)
        mock_extract.side_effect = [
            (real_feats, [f"real_{i}.png" for i in range(20)]),
            (gen_feats, [f"gen_{i}.png" for i in range(30)]),
        ]

        reports_dir = run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

        report_path = reports_dir / "evaluation.json"
        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        assert report["mode"] == "evaluate"
        assert "real_vs_generated" in report["comparisons"]
        assert len(report["comparisons"]) == 1
        assert report["dataset_sizes"]["real"] == 20
        assert report["dataset_sizes"]["generated"] == 30
        assert "selected" not in report["dataset_sizes"]

    @patch("src.experiments.sample_selection.evaluator.extract_features_from_loader")
    @patch("src.experiments.sample_selection.evaluator.create_feature_model")
    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    @patch("src.experiments.sample_selection.evaluator.SplitFileDataset")
    def test_all_three_comparisons(
        self,
        mock_split_ds,
        mock_simple_ds,
        mock_load_real,
        mock_create_model,
        mock_extract,
        tmp_output_dir,
    ):
        """With data.selected, report has 3 comparisons."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir, include_selected=True)

        # Mock datasets
        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=20)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=30)
        mock_simple_ds.return_value = mock_gen_ds

        mock_sel_ds = MagicMock()
        mock_sel_ds.__len__ = MagicMock(return_value=15)
        mock_split_ds.return_value = mock_sel_ds

        mock_create_model.return_value = MagicMock()

        real_feats = _make_mock_features(20, seed=1)
        gen_feats = _make_mock_features(30, seed=2)
        sel_feats = _make_mock_features(15, seed=3)
        mock_extract.side_effect = [
            (real_feats, [f"real_{i}.png" for i in range(20)]),
            (gen_feats, [f"gen_{i}.png" for i in range(30)]),
            (sel_feats, [f"sel_{i}.png" for i in range(15)]),
        ]

        reports_dir = run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

        report_path = reports_dir / "evaluation.json"
        with open(report_path) as f:
            report = json.load(f)

        assert len(report["comparisons"]) == 3
        assert "real_vs_generated" in report["comparisons"]
        assert "real_vs_selected" in report["comparisons"]
        assert "generated_vs_selected" in report["comparisons"]
        assert report["dataset_sizes"]["selected"] == 15

    @patch("src.experiments.sample_selection.evaluator.extract_features_from_loader")
    @patch("src.experiments.sample_selection.evaluator.create_feature_model")
    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_report_metrics_structure(
        self,
        mock_simple_ds,
        mock_load_real,
        mock_create_model,
        mock_extract,
        tmp_output_dir,
    ):
        """Each comparison should contain FID, precision, recall, roc_auc, pr_auc."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=20)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=30)
        mock_simple_ds.return_value = mock_gen_ds

        mock_create_model.return_value = MagicMock()

        real_feats = _make_mock_features(20, seed=1)
        gen_feats = _make_mock_features(30, seed=2)
        mock_extract.side_effect = [
            (real_feats, [f"real_{i}.png" for i in range(20)]),
            (gen_feats, [f"gen_{i}.png" for i in range(30)]),
        ]

        reports_dir = run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

        with open(reports_dir / "evaluation.json") as f:
            report = json.load(f)

        metrics = report["comparisons"]["real_vs_generated"]
        assert "fid" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

    @patch("src.experiments.sample_selection.evaluator.extract_features_from_loader")
    @patch("src.experiments.sample_selection.evaluator.create_feature_model")
    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_report_written_to_correct_path(
        self,
        mock_simple_ds,
        mock_load_real,
        mock_create_model,
        mock_extract,
        tmp_output_dir,
    ):
        """Report should be written to reports/evaluation.json."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=10)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=10)
        mock_simple_ds.return_value = mock_gen_ds

        mock_create_model.return_value = MagicMock()

        mock_extract.side_effect = [
            (_make_mock_features(10, seed=1), [f"r_{i}.png" for i in range(10)]),
            (_make_mock_features(10, seed=2), [f"g_{i}.png" for i in range(10)]),
        ]

        reports_dir = run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

        expected = Path(config["output"]["base_dir"]) / "reports" / "evaluation.json"
        assert expected.exists()
        assert reports_dir == expected.parent

    @patch("src.experiments.sample_selection.evaluator.extract_features_from_loader")
    @patch("src.experiments.sample_selection.evaluator.create_feature_model")
    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_small_dataset_skips_auc_metrics(
        self,
        mock_simple_ds,
        mock_load_real,
        mock_create_model,
        mock_extract,
        tmp_output_dir,
    ):
        """With fewer samples than threshold, report sets roc_auc/pr_auc to null."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)

        # 5 samples is below _MIN_SAMPLES_FOR_AUC (7) but enough for k-NN (k=3)
        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=5)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=50)
        mock_simple_ds.return_value = mock_gen_ds

        mock_create_model.return_value = MagicMock()

        real_feats = _make_mock_features(5, seed=1)
        gen_feats = _make_mock_features(50, seed=2)
        mock_extract.side_effect = [
            (real_feats, [f"r_{i}.png" for i in range(5)]),
            (gen_feats, [f"g_{i}.png" for i in range(50)]),
        ]

        reports_dir = run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

        with open(reports_dir / "evaluation.json") as f:
            report = json.load(f)

        metrics = report["comparisons"]["real_vs_generated"]
        assert "fid" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert metrics["roc_auc"] is None
        assert metrics["pr_auc"] is None

    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_empty_real_dataset_raises(
        self,
        mock_simple_ds,
        mock_load_real,
        tmp_output_dir,
    ):
        """Empty real dataset should raise ValueError."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=0)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=10)
        mock_simple_ds.return_value = mock_gen_ds

        with pytest.raises(ValueError, match="Real dataset is empty"):
            run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    @patch("src.experiments.sample_selection.evaluator.SplitFileDataset")
    def test_empty_selected_dataset_raises(
        self,
        mock_split_ds,
        mock_simple_ds,
        mock_load_real,
        tmp_output_dir,
    ):
        """Empty selected dataset should raise ValueError."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir, include_selected=True)

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=10)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=10)
        mock_simple_ds.return_value = mock_gen_ds

        mock_sel_ds = MagicMock()
        mock_sel_ds.__len__ = MagicMock(return_value=0)
        mock_split_ds.return_value = mock_sel_ds

        with pytest.raises(ValueError, match="Selected dataset is empty"):
            run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_k_exceeds_dataset_size_raises(
        self,
        mock_simple_ds,
        mock_load_real,
        tmp_output_dir,
    ):
        """k >= smaller dataset size should raise ValueError before feature extraction."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)
        config["evaluation"]["k"] = 5  # k=5 but only 5 real samples → k >= min_size

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=5)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=50)
        mock_simple_ds.return_value = mock_gen_ds

        with pytest.raises(ValueError, match="evaluation.k"):
            run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

    @patch("src.experiments.sample_selection.evaluator.evaluate_generative_model")
    @patch("src.experiments.sample_selection.evaluator.extract_features_from_loader")
    @patch("src.experiments.sample_selection.evaluator.create_feature_model")
    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_k_at_boundary_succeeds(
        self,
        mock_simple_ds,
        mock_load_real,
        mock_create_model,
        mock_extract,
        mock_eval_gen,
        tmp_output_dir,
    ):
        """k < smallest dataset size should succeed."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)
        config["evaluation"]["k"] = 5  # k=5 with 6 real samples → should pass

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=6)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=50)
        mock_simple_ds.return_value = mock_gen_ds

        mock_create_model.return_value = MagicMock()

        mock_extract.side_effect = [
            (_make_mock_features(6, seed=1), [f"r_{i}.png" for i in range(6)]),
            (_make_mock_features(50, seed=2), [f"g_{i}.png" for i in range(50)]),
        ]

        mock_eval_gen.return_value = {
            "fid": 10.0,
            "precision": 0.8,
            "recall": 0.7,
            "roc_auc": 0.9,
            "pr_auc": 0.85,
        }

        # Should not raise
        reports_dir = run_sample_selection_evaluate(config, "cpu", tmp_output_dir)
        assert (reports_dir / "evaluation.json").exists()

    @patch("src.experiments.sample_selection.evaluator.calculate_precision_recall")
    @patch("src.experiments.sample_selection.evaluator.calculate_fid")
    @patch("src.experiments.sample_selection.evaluator.evaluate_generative_model")
    @patch("src.experiments.sample_selection.evaluator.extract_features_from_loader")
    @patch("src.experiments.sample_selection.evaluator.create_feature_model")
    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_auc_failure_falls_back_to_fid_precision_recall(
        self,
        mock_simple_ds,
        mock_load_real,
        mock_create_model,
        mock_extract,
        mock_eval_gen,
        mock_calc_fid,
        mock_calc_pr,
        tmp_output_dir,
    ):
        """When evaluate_generative_model raises ValueError, should fall back gracefully."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=10)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=10)
        mock_simple_ds.return_value = mock_gen_ds

        mock_create_model.return_value = MagicMock()

        mock_extract.side_effect = [
            (_make_mock_features(10, seed=1), [f"r_{i}.png" for i in range(10)]),
            (_make_mock_features(10, seed=2), [f"g_{i}.png" for i in range(10)]),
        ]

        # Simulate AUC computation failure (e.g. stratified split issue)
        mock_eval_gen.side_effect = ValueError("too few samples for stratify")
        mock_calc_fid.return_value = 15.0
        mock_calc_pr.return_value = (0.8, 0.7)

        reports_dir = run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

        with open(reports_dir / "evaluation.json") as f:
            report = json.load(f)

        metrics = report["comparisons"]["real_vs_generated"]
        assert metrics["fid"] == 15.0
        assert metrics["precision"] == 0.8
        assert metrics["recall"] == 0.7
        assert metrics["roc_auc"] is None
        assert metrics["pr_auc"] is None

    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    def test_real_dataset_load_error_has_friendly_message(
        self,
        mock_load_real,
        tmp_output_dir,
    ):
        """FileNotFoundError from load_real_dataset should include config context."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)
        mock_load_real.side_effect = FileNotFoundError("no such directory")

        with pytest.raises(FileNotFoundError, match="Failed to load real dataset"):
            run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_generated_dataset_load_error_has_friendly_message(
        self,
        mock_simple_ds,
        mock_load_real,
        tmp_output_dir,
    ):
        """FileNotFoundError from SimpleImageDataset should include config context."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=10)
        mock_load_real.return_value = mock_real_ds

        mock_simple_ds.side_effect = FileNotFoundError("no such directory")

        with pytest.raises(FileNotFoundError, match="Failed to load generated dataset"):
            run_sample_selection_evaluate(config, "cpu", tmp_output_dir)

    @patch("src.experiments.sample_selection.evaluator.load_real_dataset")
    @patch("src.experiments.sample_selection.evaluator.SimpleImageDataset")
    def test_empty_generated_dataset_raises(
        self,
        mock_simple_ds,
        mock_load_real,
        tmp_output_dir,
    ):
        """Empty generated dataset should raise ValueError."""
        from src.experiments.sample_selection.evaluator import (
            run_sample_selection_evaluate,
        )

        config = _make_evaluate_config(tmp_output_dir)

        mock_real_ds = MagicMock()
        mock_real_ds.__len__ = MagicMock(return_value=10)
        mock_load_real.return_value = mock_real_ds

        mock_gen_ds = MagicMock()
        mock_gen_ds.__len__ = MagicMock(return_value=0)
        mock_simple_ds.return_value = mock_gen_ds

        with pytest.raises(ValueError, match="Generated dataset is empty"):
            run_sample_selection_evaluate(config, "cpu", tmp_output_dir)
