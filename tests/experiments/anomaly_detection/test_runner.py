"""Tests for the anomaly-detection runner and report."""

import json
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.experiments.anomaly_detection.report import (
    build_score_table,
    summarize_by_subclass,
)
from src.experiments.anomaly_detection.runner import (
    compute_binary_metrics,
    load_extended_split,
    run_anomaly_detection,
    select_partitions,
)


def _split():
    def e(path, label, subclass):
        return {"path": path, "label": label, "subclass": subclass}

    return {
        "train": [e("s0", 0, "suspicious"), e("a0", 1, "abnormal")],
        "val": [e("s1", 0, "suspicious"), e("a1", 1, "abnormal")],
        "test": [e("s2", 0, "suspicious"), e("a2", 1, "abnormal")],
        "normal_extra_train": [e("r0", 0, "red")],
        "normal_extra_val": [e("r1", 0, "red")],
        "normal_extra_test": [e("r2", 0, "red")],
    }


@pytest.mark.unit
class TestSelectPartitions:
    def test_all_pool_never_trains_on_abnormal(self):
        parts = select_partitions(_split(), "all")
        assert [x["path"] for x in parts["train_normals"]] == ["s0", "r0"]
        assert [x["path"] for x in parts["val_normals"]] == ["s1", "r1"]
        assert [x["path"] for x in parts["test"]] == ["s2", "a2"]
        assert [x["path"] for x in parts["extra_test"]] == ["r2"]

    def test_suspicious_pool_excludes_extras(self):
        parts = select_partitions(_split(), "suspicious")
        assert [x["path"] for x in parts["train_normals"]] == ["s0"]
        assert [x["path"] for x in parts["val_normals"]] == ["s1"]

    def test_unknown_pool_raises(self):
        with pytest.raises(ValueError, match="normal_pool"):
            select_partitions(_split(), "red")


@pytest.mark.unit
class TestBinaryMetrics:
    def test_perfect_ranking(self):
        targets = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_binary_metrics(targets, scores, threshold=0.5)
        assert m["pr_auc"] == pytest.approx(1.0)
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["recall_1"] == pytest.approx(1.0)
        assert m["cm_1_1"] == 2.0 and m["cm_0_1"] == 0.0

    def test_threshold_is_strict(self):
        m = compute_binary_metrics(np.array([0, 1]), np.array([0.5, 0.6]), 0.5)
        assert m["cm_0_0"] == 1.0

    @pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
    def test_single_class_skips_auc(self):
        m = compute_binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]), 0.15)
        assert "pr_auc" not in m


@pytest.mark.unit
class TestLoadExtendedSplit:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_extended_split(str(tmp_path / "nope.json"))

    def test_binary_split_rejected(self, tmp_path):
        path = tmp_path / "binary.json"
        path.write_text(json.dumps({"train": [], "val": [], "test": []}))
        with pytest.raises(KeyError, match="anomaly_detection.splits"):
            load_extended_split(str(path))


@pytest.mark.unit
class TestReport:
    def test_summary_order_and_fraction(self):
        entries = [
            {"path": "a", "label": 1, "subclass": "abnormal"},
            {"path": "b", "label": 0, "subclass": "red"},
            {"path": "c", "label": 0, "subclass": "zzz"},
            {"path": "d", "label": 0, "subclass": "suspicious"},
        ]
        table = build_score_table(entries, np.array([3.0, 1.0, 2.0, 0.5]), "test")
        summary = summarize_by_subclass(table, threshold=1.5)
        assert summary["subclass"].tolist() == ["abnormal", "suspicious", "red", "zzz"]
        assert summary.set_index("subclass")["frac_above_threshold"].to_dict() == {
            "abnormal": 1.0,
            "suspicious": 0.0,
            "red": 0.0,
            "zzz": 1.0,
        }


def _config(split_file, tmp_path, method="knn"):
    return {
        "compute": {"device": "cpu", "seed": 0},
        "data": {"split_file": str(split_file), "normal_pool": "all"},
        "feature_extraction": {
            "model": "resnet50",
            "image_size": 16,
            "crop_size": 16,
            "batch_size": 4,
            "num_workers": 0,
            "cache_dir": str(tmp_path / "cache"),
        },
        "method": {
            "type": method,
            "knn": {"k": 2},
            "mahalanobis": {"shrinkage": "ledoit_wolf"},
            "patchcore": {
                "layers": ["layer2", "layer3"],
                "grid_size": 2,
                "feature_dim": 3,
                "coreset_ratio": 0.5,
                "projection_dim": 2,
            },
        },
        "threshold": {"normal_percentile": 95},
        "output": {
            "base_dir": str(tmp_path / "out"),
            "subdirs": {"logs": "logs", "reports": "reports"},
        },
    }


@pytest.mark.unit
class TestRunAnomalyDetectionStubbed:
    """Fast runner check: features and plotting are stubbed out."""

    def test_writes_reports(self, ad_split_file, tmp_path):
        from unittest.mock import patch

        def fake_features(paths, spec, **kwargs):
            # 1-D feature = 1 for abnormal images, 0 for normal ones.
            return np.array([[1.0] if "abnormal" in p else [0.0] for p in paths])

        config = _config(ad_split_file, tmp_path)
        config["method"]["knn"]["k"] = 1
        with (
            patch(
                "src.experiments.anomaly_detection.runner.get_features",
                side_effect=fake_features,
            ),
            patch(
                "src.experiments.anomaly_detection.runner.plot_subclass_scores"
            ) as mock_plot,
        ):
            reports = run_anomaly_detection(config, "cpu")

        evaluation = json.loads((reports / "evaluation.json").read_text())
        assert evaluation["pr_auc"] == pytest.approx(1.0)
        assert evaluation["recall_1"] == pytest.approx(1.0)
        assert evaluation["threshold"] == pytest.approx(0.0)
        assert evaluation["normal_pool"] == "all"
        assert (reports / "predictions_test.npz").exists()
        summary = pd.read_csv(reports / "subclass_summary.csv")
        assert summary["subclass"].tolist() == ["abnormal", "suspicious", "junk"]
        mock_plot.assert_called_once()

    def test_failure_after_metrics_leaves_no_done_marker(self, ad_split_file, tmp_path):
        """evaluation.json is the completion marker: it must be written last,
        and a stale one from an earlier run must not survive a failed rerun."""
        from unittest.mock import patch

        config = _config(ad_split_file, tmp_path)
        reports = tmp_path / "out" / "reports"
        reports.mkdir(parents=True)
        (reports / "evaluation.json").write_text('{"old": true}')

        with (
            patch(
                "src.experiments.anomaly_detection.runner.get_features",
                side_effect=lambda paths, spec, **kw: np.zeros((len(paths), 1)),
            ),
            patch(
                "src.experiments.anomaly_detection.runner.plot_subclass_scores",
                side_effect=RuntimeError("plot failed"),
            ),
            pytest.raises(RuntimeError, match="plot failed"),
        ):
            run_anomaly_detection(config, "cpu")

        assert not (reports / "evaluation.json").exists()
        assert not (reports / "evaluation.json.tmp").exists()
        assert (reports / "predictions_test.npz").exists()

    def test_no_abnormal_in_test_still_writes_reports(self, ad_split_file, tmp_path):
        from unittest.mock import patch

        split = json.loads(ad_split_file.read_text())
        split["test"] = [e for e in split["test"] if e["label"] == 0]
        path = tmp_path / "no_abnormal.json"
        path.write_text(json.dumps(split))

        with (
            patch(
                "src.experiments.anomaly_detection.runner.get_features",
                side_effect=lambda paths, spec, **kw: np.zeros((len(paths), 1)),
            ),
            patch("src.experiments.anomaly_detection.runner.plot_subclass_scores"),
        ):
            reports = run_anomaly_detection(_config(path, tmp_path), "cpu")

        evaluation = json.loads((reports / "evaluation.json").read_text())
        for key in (
            "pr_auc",
            "roc_auc",
            "pr_auc_vs_all_normals",
            "roc_auc_vs_all_normals",
        ):
            assert key not in evaluation
        assert (reports / "predictions_test.npz").exists()
        assert (reports / "subclass_summary.csv").exists()

    def test_no_normals_raises(self, tmp_path):
        split: Dict[str, Any] = {
            k: []
            for k in (
                "train",
                "val",
                "test",
                "normal_extra_train",
                "normal_extra_val",
                "normal_extra_test",
            )
        }
        split["metadata"] = {"classes": {"suspicious": 0, "abnormal": 1}}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(split))
        with pytest.raises(ValueError, match="No normal images"):
            run_anomaly_detection(_config(path, tmp_path), "cpu")


@pytest.mark.component
class TestRunAnomalyDetection:
    @pytest.mark.parametrize("method", ["knn", "mahalanobis", "patchcore"])
    def test_end_to_end(self, method, tiny_backbone, ad_split_file, tmp_path):
        reports = run_anomaly_detection(_config(ad_split_file, tmp_path, method), "cpu")

        evaluation = json.loads((reports / "evaluation.json").read_text())
        for key in (
            "pr_auc",
            "roc_auc",
            "recall_1",
            "f1_1",
            "pr_auc_vs_all_normals",
            "threshold",
        ):
            assert np.isfinite(evaluation[key])
        assert evaluation["split"] == "test"
        assert evaluation["positive_class"] == 1
        assert evaluation["class_names"] == ["suspicious", "abnormal"]
        assert evaluation["n_train_normals"] == 11  # 6 suspicious + 5 red
        assert evaluation["n_test"] == 7
        assert evaluation["n_test_vs_all_normals"] == 10  # 7 test + 3 extra normals

        preds = np.load(reports / "predictions_test.npz")
        assert preds["scores"].shape == (7,)
        assert preds["targets"].tolist() == [0, 0, 0, 0, 1, 1, 1]

        table = pd.read_csv(reports / "subclass_scores.csv")
        assert set(table["partition"]) == {"test", "normal_extra_test"}
        assert (reports / "subclass_summary.csv").exists()
        assert (reports / "subclass_scores.png").exists()

    def test_bright_abnormal_is_detected(self, tiny_backbone, ad_split_file, tmp_path):
        reports = run_anomaly_detection(_config(ad_split_file, tmp_path), "cpu")
        evaluation = json.loads((reports / "evaluation.json").read_text())
        assert evaluation["roc_auc"] == pytest.approx(1.0)

    def test_readable_by_cross_split_report(
        self, tiny_backbone, ad_split_file, tmp_path
    ):
        from src.experiments.classifier.cross_split_report import (
            load_per_split_results,
        )

        config = _config(ad_split_file, tmp_path)
        config["output"]["base_dir"] = str(
            tmp_path / "ad" / "split0" / "ad-frozen" / "ad-knn-rn50__all" / "seed0"
        )
        run_anomaly_detection(config, "cpu")
        per_split = load_per_split_results(str(tmp_path / "ad"), "ad-frozen")
        assert list(per_split) == [0]
        assert per_split[0][0]["experiment"] == "ad-knn-rn50__all"
        assert per_split[0][0]["seed"] == 0
