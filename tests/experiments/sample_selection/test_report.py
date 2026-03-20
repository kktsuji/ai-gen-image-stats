"""Tests for Sample Selection Report Generation

Unit tests for CSV quality report, accepted samples JSON, and summary JSON.
"""

import csv
import json

import numpy as np
import pytest

from src.experiments.sample_selection.report import (
    write_accepted_samples_json,
    write_evaluation_report,
    write_quality_report,
    write_summary,
)

# ============================================================================
# Unit Tests - write_quality_report
# ============================================================================


@pytest.mark.unit
class TestWriteQualityReport:
    """Test CSV quality report generation."""

    def test_creates_csv_with_correct_headers(self, tmp_output_dir):
        """CSV should have the expected column headers."""
        output = tmp_output_dir / "report.csv"
        write_quality_report(
            output_path=output,
            image_paths=["a.png", "b.png"],
            knn_scores=np.array([1.0, 2.0]),
            realism_flags=np.array([True, False]),
            selected_mask=np.array([True, False]),
        )
        with open(output, newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == ["path", "knn_score", "realism_flag", "selected"]

    def test_correct_number_of_rows(self, tmp_output_dir):
        """CSV should have one row per sample plus header."""
        output = tmp_output_dir / "report.csv"
        n = 5
        write_quality_report(
            output_path=output,
            image_paths=[f"img_{i}.png" for i in range(n)],
            knn_scores=np.random.rand(n),
            realism_flags=np.ones(n, dtype=bool),
            selected_mask=np.zeros(n, dtype=bool),
        )
        with open(output, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == n + 1  # header + data rows

    def test_none_realism_flags_produces_empty_column(self, tmp_output_dir):
        """When realism_flags is None, the column should be empty."""
        output = tmp_output_dir / "report.csv"
        write_quality_report(
            output_path=output,
            image_paths=["img.png"],
            knn_scores=np.array([1.5]),
            realism_flags=None,
            selected_mask=np.array([True]),
        )
        with open(output, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)
        assert row[2] == ""  # realism_flag column

    def test_creates_parent_directories(self, tmp_output_dir):
        """Should create parent directories if they don't exist."""
        output = tmp_output_dir / "nested" / "dir" / "report.csv"
        write_quality_report(
            output_path=output,
            image_paths=["img.png"],
            knn_scores=np.array([1.0]),
            realism_flags=None,
            selected_mask=np.array([True]),
        )
        assert output.exists()


# ============================================================================
# Unit Tests - write_accepted_samples_json
# ============================================================================


@pytest.mark.unit
class TestWriteAcceptedSamplesJson:
    """Test accepted samples JSON generation."""

    def test_output_has_train_key(self, tmp_output_dir):
        """JSON should have a 'train' key for SplitFileDataset compatibility."""
        output = tmp_output_dir / "accepted.json"
        write_accepted_samples_json(
            output_path=output,
            selected_paths=["img_0.png", "img_1.png"],
            label=0,
            class_name="normal",
            metadata={"selection_mode": "top_k", "total_candidates": 10},
        )
        with open(output) as f:
            data = json.load(f)
        assert "train" in data
        assert "metadata" in data

    def test_train_entries_have_correct_format(self, tmp_output_dir):
        """Each train entry should have 'path' and 'label' keys."""
        output = tmp_output_dir / "accepted.json"
        write_accepted_samples_json(
            output_path=output,
            selected_paths=["a.png", "b.png"],
            label=1,
            class_name="abnormal",
            metadata={},
        )
        with open(output) as f:
            data = json.load(f)
        for entry in data["train"]:
            assert "path" in entry
            assert "label" in entry
            assert entry["label"] == 1

    def test_correct_number_of_train_entries(self, tmp_output_dir):
        """Number of train entries should match selected_paths."""
        paths = [f"img_{i}.png" for i in range(7)]
        output = tmp_output_dir / "accepted.json"
        write_accepted_samples_json(
            output_path=output,
            selected_paths=paths,
            label=0,
            class_name="normal",
            metadata={},
        )
        with open(output) as f:
            data = json.load(f)
        assert len(data["train"]) == 7

    def test_metadata_contains_classes(self, tmp_output_dir):
        """Metadata should contain classes mapping."""
        output = tmp_output_dir / "accepted.json"
        write_accepted_samples_json(
            output_path=output,
            selected_paths=["img.png"],
            label=0,
            class_name="normal",
            metadata={},
        )
        with open(output) as f:
            data = json.load(f)
        assert data["metadata"]["classes"] == {"normal": 0}

    def test_metadata_contains_source_experiment(self, tmp_output_dir):
        """Metadata should identify source experiment."""
        output = tmp_output_dir / "accepted.json"
        write_accepted_samples_json(
            output_path=output,
            selected_paths=[],
            label=0,
            class_name="normal",
            metadata={},
        )
        with open(output) as f:
            data = json.load(f)
        assert data["metadata"]["source_experiment"] == "sample_selection"

    def test_metadata_includes_custom_fields(self, tmp_output_dir):
        """Custom metadata fields should be included."""
        output = tmp_output_dir / "accepted.json"
        write_accepted_samples_json(
            output_path=output,
            selected_paths=[],
            label=0,
            class_name="normal",
            metadata={"selection_mode": "percentile", "total_selected": 42},
        )
        with open(output) as f:
            data = json.load(f)
        assert data["metadata"]["selection_mode"] == "percentile"
        assert data["metadata"]["total_selected"] == 42


# ============================================================================
# Unit Tests - write_summary
# ============================================================================


@pytest.mark.unit
class TestWriteSummary:
    """Test summary JSON generation."""

    def test_summary_contains_selection_stats(self, tmp_output_dir):
        """Summary should contain selection statistics."""
        output = tmp_output_dir / "summary.json"
        stats = {"total_candidates": 100, "total_selected": 80}
        write_summary(
            output_path=output,
            dataset_metrics=None,
            selection_stats=stats,
            config={"scoring": {"k": 5}, "selection": {"mode": "top_k"}},
        )
        with open(output) as f:
            data = json.load(f)
        assert data["selection"]["total_candidates"] == 100
        assert data["selection"]["total_selected"] == 80

    def test_summary_includes_dataset_metrics_when_provided(self, tmp_output_dir):
        """Summary should include dataset metrics when provided."""
        output = tmp_output_dir / "summary.json"
        metrics = {"fid": 42.5, "precision": 0.85, "recall": 0.72}
        write_summary(
            output_path=output,
            dataset_metrics=metrics,
            selection_stats={"total_candidates": 10},
            config={"scoring": {"k": 5}},
        )
        with open(output) as f:
            data = json.load(f)
        assert "dataset_metrics" in data
        assert data["dataset_metrics"]["fid"] == 42.5

    def test_summary_omits_dataset_metrics_when_none(self, tmp_output_dir):
        """Summary should not include dataset_metrics key when None."""
        output = tmp_output_dir / "summary.json"
        write_summary(
            output_path=output,
            dataset_metrics=None,
            selection_stats={"total_candidates": 10},
            config={"scoring": {"k": 5}},
        )
        with open(output) as f:
            data = json.load(f)
        assert "dataset_metrics" not in data

    def test_summary_contains_config_snapshot(self, tmp_output_dir):
        """Summary should contain relevant config sections."""
        output = tmp_output_dir / "summary.json"
        config = {
            "feature_extraction": {"model": "resnet50"},
            "scoring": {"k": 5},
            "selection": {"mode": "percentile", "value": 80},
            "dataset_metrics": {"enabled": True},
        }
        write_summary(
            output_path=output,
            dataset_metrics=None,
            selection_stats={},
            config=config,
        )
        with open(output) as f:
            data = json.load(f)
        assert data["config"]["feature_extraction"]["model"] == "resnet50"
        assert data["config"]["scoring"]["k"] == 5

    def test_summary_has_created_at(self, tmp_output_dir):
        """Summary should have a created_at timestamp."""
        output = tmp_output_dir / "summary.json"
        write_summary(
            output_path=output,
            dataset_metrics=None,
            selection_stats={},
            config={},
        )
        with open(output) as f:
            data = json.load(f)
        assert "created_at" in data


# ============================================================================
# Unit Tests - write_evaluation_report
# ============================================================================


@pytest.mark.unit
class TestWriteEvaluationReport:
    """Test evaluation report JSON generation."""

    def test_output_structure_single_comparison(self, tmp_output_dir):
        """Report with one comparison should have correct structure."""
        output = tmp_output_dir / "evaluation.json"
        comparisons: dict[str, dict[str, float | None]] = {
            "real_vs_generated": {
                "fid": 12.3,
                "precision": 0.85,
                "recall": 0.72,
                "roc_auc": 0.65,
                "pr_auc": 0.68,
            },
        }
        write_evaluation_report(
            output_path=output,
            comparisons=comparisons,
            dataset_sizes={"real": 100, "generated": 500},
            config={"feature_extraction": {"model": "inceptionv3"}},
        )
        with open(output) as f:
            data = json.load(f)

        assert data["mode"] == "evaluate"
        assert "created_at" in data
        assert len(data["comparisons"]) == 1
        assert data["comparisons"]["real_vs_generated"]["fid"] == 12.3
        assert data["dataset_sizes"]["real"] == 100
        assert data["dataset_sizes"]["generated"] == 500

    def test_output_structure_three_comparisons(self, tmp_output_dir):
        """Report with three comparisons should include all pairs."""
        output = tmp_output_dir / "evaluation.json"
        comparisons: dict[str, dict[str, float | None]] = {
            "real_vs_generated": {"fid": 12.3, "precision": 0.85},
            "real_vs_selected": {"fid": 8.1, "precision": 0.92},
            "generated_vs_selected": {"fid": 3.2, "precision": 0.97},
        }
        write_evaluation_report(
            output_path=output,
            comparisons=comparisons,
            dataset_sizes={"real": 100, "generated": 500, "selected": 400},
            config={},
        )
        with open(output) as f:
            data = json.load(f)

        assert len(data["comparisons"]) == 3
        assert "real_vs_generated" in data["comparisons"]
        assert "real_vs_selected" in data["comparisons"]
        assert "generated_vs_selected" in data["comparisons"]
        assert data["dataset_sizes"]["selected"] == 400

    def test_config_snapshot_included(self, tmp_output_dir):
        """Report should include relevant config sections."""
        output = tmp_output_dir / "evaluation.json"
        config = {
            "feature_extraction": {"model": "resnet50", "batch_size": 32},
            "evaluation": {"k": 5},
            "data": {
                "real": {"source": "directory", "directory": "/tmp/real"},
                "generated": {"directory": "/tmp/gen"},
            },
        }
        write_evaluation_report(
            output_path=output,
            comparisons={"real_vs_generated": {"fid": 10.0, "roc_auc": None}},
            dataset_sizes={"real": 50, "generated": 100},
            config=config,
        )
        with open(output) as f:
            data = json.load(f)

        assert data["config"]["feature_extraction"]["model"] == "resnet50"
        assert data["config"]["evaluation"]["k"] == 5
        assert data["config"]["data"]["real"]["source"] == "directory"
        assert data["config"]["data"]["generated"]["directory"] == "/tmp/gen"

    def test_creates_parent_directories(self, tmp_output_dir):
        """Should create parent directories if they don't exist."""
        output = tmp_output_dir / "nested" / "dir" / "evaluation.json"
        write_evaluation_report(
            output_path=output,
            comparisons={"real_vs_generated": {"fid": 5.0, "roc_auc": None}},
            dataset_sizes={"real": 10, "generated": 20},
            config={},
        )
        assert output.exists()
