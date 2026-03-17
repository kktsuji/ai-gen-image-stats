"""Tests for Sample Selection Configuration

This module tests the sample selection configuration validation.
"""

import copy
from pathlib import Path

import pytest
import yaml

from src.experiments.sample_selection.config import validate_config

# Resolve project root so config file tests work regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


_config_path = _PROJECT_ROOT / "configs/sample-selection-example.yaml"
with open(_config_path) as _f:
    _BASE_CONFIG = yaml.safe_load(_f)


def _make_valid_config():
    """Create a valid config for testing.

    Returns a deep copy of the cached example config so each test gets
    an isolated copy (single source of truth).
    """
    return copy.deepcopy(_BASE_CONFIG)


# ============================================================================
# Unit Tests - Fast, Pure Logic
# ============================================================================


@pytest.mark.unit
class TestValidateConfig:
    """Test configuration validation."""

    def test_valid_config_passes(self):
        """Test that a valid config passes validation."""
        config = _make_valid_config()
        validate_config(config)  # Should not raise

    def test_wrong_experiment_type_raises(self):
        """Test that wrong experiment type raises ValueError."""
        config = _make_valid_config()
        config["experiment"] = "diffusion"
        with pytest.raises(ValueError, match="Expected 'sample_selection'"):
            validate_config(config)

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "train"
        with pytest.raises(ValueError, match="Valid modes"):
            validate_config(config)

    # -- feature_extraction section --

    def test_missing_feature_extraction_raises(self):
        """Test that missing feature_extraction raises KeyError."""
        config = _make_valid_config()
        del config["feature_extraction"]
        with pytest.raises(KeyError, match="feature_extraction"):
            validate_config(config)

    def test_invalid_feature_model_raises(self):
        """Test that invalid feature extraction model raises ValueError."""
        config = _make_valid_config()
        config["feature_extraction"]["model"] = "vgg16"
        with pytest.raises(ValueError, match="feature_extraction.model"):
            validate_config(config)

    def test_missing_feature_batch_size_raises(self):
        """Test that missing batch_size raises KeyError."""
        config = _make_valid_config()
        del config["feature_extraction"]["batch_size"]
        with pytest.raises(KeyError, match="batch_size"):
            validate_config(config)

    def test_invalid_feature_batch_size_raises(self):
        """Test that non-positive batch_size raises ValueError."""
        config = _make_valid_config()
        config["feature_extraction"]["batch_size"] = 0
        with pytest.raises(ValueError, match="batch_size"):
            validate_config(config)

    def test_boolean_feature_batch_size_raises(self):
        """Test that boolean batch_size raises ValueError."""
        config = _make_valid_config()
        config["feature_extraction"]["batch_size"] = True
        with pytest.raises(ValueError, match="batch_size"):
            validate_config(config)

    def test_missing_feature_image_size_raises(self):
        """Test that missing image_size raises KeyError."""
        config = _make_valid_config()
        del config["feature_extraction"]["image_size"]
        with pytest.raises(KeyError, match="image_size"):
            validate_config(config)

    def test_boolean_feature_image_size_raises(self):
        """Test that boolean image_size raises ValueError."""
        config = _make_valid_config()
        config["feature_extraction"]["image_size"] = True
        with pytest.raises(ValueError, match="image_size"):
            validate_config(config)

    def test_missing_feature_num_workers_raises(self):
        """Test that missing num_workers raises KeyError."""
        config = _make_valid_config()
        del config["feature_extraction"]["num_workers"]
        with pytest.raises(KeyError, match="num_workers"):
            validate_config(config)

    def test_negative_num_workers_raises(self):
        """Test that negative num_workers raises ValueError."""
        config = _make_valid_config()
        config["feature_extraction"]["num_workers"] = -1
        with pytest.raises(ValueError, match="num_workers"):
            validate_config(config)

    def test_boolean_num_workers_raises(self):
        """Test that boolean num_workers raises ValueError."""
        config = _make_valid_config()
        config["feature_extraction"]["num_workers"] = False
        with pytest.raises(ValueError, match="num_workers"):
            validate_config(config)

    # -- data section --

    def test_missing_data_raises(self):
        """Test that missing data section raises KeyError."""
        config = _make_valid_config()
        del config["data"]
        with pytest.raises(KeyError, match="data"):
            validate_config(config)

    def test_missing_data_real_raises(self):
        """Test that missing data.real raises KeyError."""
        config = _make_valid_config()
        del config["data"]["real"]
        with pytest.raises(KeyError, match="data.real"):
            validate_config(config)

    def test_invalid_real_source_raises(self):
        """Test that invalid data.real.source raises ValueError."""
        config = _make_valid_config()
        config["data"]["real"]["source"] = "database"
        with pytest.raises(ValueError, match="data.real.source"):
            validate_config(config)

    def test_split_file_source_missing_split_file_raises(self):
        """Test split_file source requires split_file field."""
        config = _make_valid_config()
        config["data"]["real"]["source"] = "split_file"
        del config["data"]["real"]["split_file"]
        with pytest.raises(KeyError, match="split_file"):
            validate_config(config)

    def test_split_file_source_missing_split_raises(self):
        """Test split_file source requires split field."""
        config = _make_valid_config()
        config["data"]["real"]["source"] = "split_file"
        del config["data"]["real"]["split"]
        with pytest.raises(KeyError, match="data.real.split"):
            validate_config(config)

    def test_split_file_source_invalid_split_raises(self):
        """Test that invalid split value raises ValueError."""
        config = _make_valid_config()
        config["data"]["real"]["split"] = "test"
        with pytest.raises(ValueError, match="data.real.split"):
            validate_config(config)

    def test_split_file_source_missing_class_label_raises(self):
        """Test split_file source requires class_label field."""
        config = _make_valid_config()
        del config["data"]["real"]["class_label"]
        with pytest.raises(KeyError, match="class_label"):
            validate_config(config)

    def test_class_label_string_raises(self):
        """Test that string class_label raises ValueError."""
        config = _make_valid_config()
        config["data"]["real"]["class_label"] = "zero"
        with pytest.raises(
            ValueError, match="class_label must be null or a non-negative"
        ):
            validate_config(config)

    def test_class_label_negative_raises(self):
        """Test that negative class_label raises ValueError."""
        config = _make_valid_config()
        config["data"]["real"]["class_label"] = -1
        with pytest.raises(
            ValueError, match="class_label must be null or a non-negative"
        ):
            validate_config(config)

    def test_class_label_null_valid(self):
        """Test that null class_label passes validation."""
        config = _make_valid_config()
        config["data"]["real"]["class_label"] = None
        validate_config(config)  # Should not raise

    def test_class_label_valid_int(self):
        """Test that valid non-negative integer class_label passes validation."""
        config = _make_valid_config()
        config["data"]["real"]["class_label"] = 0
        validate_config(config)  # Should not raise

    def test_directory_source_missing_directory_raises(self):
        """Test directory source requires directory field."""
        config = _make_valid_config()
        config["data"]["real"] = {"source": "directory"}
        with pytest.raises(KeyError, match="data.real.directory"):
            validate_config(config)

    def test_directory_source_valid(self):
        """Test that directory source with all fields passes."""
        config = _make_valid_config()
        config["data"]["real"] = {
            "source": "directory",
            "directory": "/some/path",
        }
        validate_config(config)

    def test_missing_generated_raises(self):
        """Test that missing data.generated raises KeyError."""
        config = _make_valid_config()
        del config["data"]["generated"]
        with pytest.raises(KeyError, match="data.generated"):
            validate_config(config)

    def test_missing_generated_directory_raises(self):
        """Test that missing data.generated.directory raises KeyError."""
        config = _make_valid_config()
        del config["data"]["generated"]["directory"]
        with pytest.raises(KeyError, match="data.generated.directory"):
            validate_config(config)

    def test_empty_generated_directory_raises(self):
        """Test that empty data.generated.directory raises ValueError."""
        config = _make_valid_config()
        config["data"]["generated"]["directory"] = ""
        with pytest.raises(ValueError, match="data.generated.directory"):
            validate_config(config)

    def test_missing_label_raises(self):
        """Test that missing data.label raises KeyError."""
        config = _make_valid_config()
        del config["data"]["label"]
        with pytest.raises(KeyError, match="data.label"):
            validate_config(config)

    def test_negative_label_raises(self):
        """Test that negative data.label raises ValueError."""
        config = _make_valid_config()
        config["data"]["label"] = -1
        with pytest.raises(ValueError, match="data.label"):
            validate_config(config)

    def test_boolean_label_raises(self):
        """Test that boolean data.label raises ValueError."""
        config = _make_valid_config()
        config["data"]["label"] = True
        with pytest.raises(ValueError, match="data.label"):
            validate_config(config)

    def test_missing_class_name_raises(self):
        """Test that missing data.class_name raises KeyError."""
        config = _make_valid_config()
        del config["data"]["class_name"]
        with pytest.raises(KeyError, match="data.class_name"):
            validate_config(config)

    def test_empty_class_name_raises(self):
        """Test that empty data.class_name raises ValueError."""
        config = _make_valid_config()
        config["data"]["class_name"] = ""
        with pytest.raises(ValueError, match="data.class_name"):
            validate_config(config)

    # -- scoring section --

    def test_missing_scoring_raises(self):
        """Test that missing scoring section raises KeyError."""
        config = _make_valid_config()
        del config["scoring"]
        with pytest.raises(KeyError, match="scoring"):
            validate_config(config)

    def test_missing_scoring_k_raises(self):
        """Test that missing scoring.k raises KeyError."""
        config = _make_valid_config()
        del config["scoring"]["k"]
        with pytest.raises(KeyError, match="scoring.k"):
            validate_config(config)

    def test_invalid_scoring_k_raises(self):
        """Test that non-positive scoring.k raises ValueError."""
        config = _make_valid_config()
        config["scoring"]["k"] = 0
        with pytest.raises(ValueError, match="scoring.k"):
            validate_config(config)

    def test_boolean_scoring_k_raises(self):
        """Test that boolean scoring.k raises ValueError."""
        config = _make_valid_config()
        config["scoring"]["k"] = True
        with pytest.raises(ValueError, match="scoring.k"):
            validate_config(config)

    def test_missing_require_realism_raises(self):
        """Test that missing scoring.require_realism raises KeyError."""
        config = _make_valid_config()
        del config["scoring"]["require_realism"]
        with pytest.raises(KeyError, match="require_realism"):
            validate_config(config)

    def test_invalid_require_realism_raises(self):
        """Test that non-boolean scoring.require_realism raises ValueError."""
        config = _make_valid_config()
        config["scoring"]["require_realism"] = "yes"
        with pytest.raises(ValueError, match="require_realism"):
            validate_config(config)

    # -- selection section --

    def test_missing_selection_raises(self):
        """Test that missing selection section raises KeyError."""
        config = _make_valid_config()
        del config["selection"]
        with pytest.raises(KeyError, match="selection"):
            validate_config(config)

    def test_invalid_selection_mode_raises(self):
        """Test that invalid selection.mode raises ValueError."""
        config = _make_valid_config()
        config["selection"]["mode"] = "random"
        with pytest.raises(ValueError, match="selection.mode"):
            validate_config(config)

    def test_missing_selection_value_raises(self):
        """Test that missing selection.value raises KeyError."""
        config = _make_valid_config()
        del config["selection"]["value"]
        with pytest.raises(KeyError, match="selection.value"):
            validate_config(config)

    def test_top_k_non_positive_value_raises(self):
        """Test that top_k mode with non-positive value raises ValueError."""
        config = _make_valid_config()
        config["selection"]["mode"] = "top_k"
        config["selection"]["value"] = 0
        with pytest.raises(ValueError, match="positive integer"):
            validate_config(config)

    def test_top_k_boolean_value_raises(self):
        """Test that top_k mode with boolean value raises ValueError."""
        config = _make_valid_config()
        config["selection"]["mode"] = "top_k"
        config["selection"]["value"] = True
        with pytest.raises(ValueError, match="positive integer"):
            validate_config(config)

    def test_percentile_boolean_value_raises(self):
        """Test that percentile mode with boolean value raises ValueError."""
        config = _make_valid_config()
        config["selection"]["mode"] = "percentile"
        config["selection"]["value"] = True
        with pytest.raises(ValueError, match="percentile"):
            validate_config(config)

    def test_threshold_boolean_value_raises(self):
        """Test that threshold mode with boolean value raises ValueError."""
        config = _make_valid_config()
        config["selection"]["mode"] = "threshold"
        config["selection"]["value"] = True
        with pytest.raises(ValueError, match="threshold"):
            validate_config(config)

    def test_top_k_valid(self):
        """Test that top_k mode with positive integer passes."""
        config = _make_valid_config()
        config["selection"]["mode"] = "top_k"
        config["selection"]["value"] = 100
        validate_config(config)

    def test_percentile_out_of_range_raises(self):
        """Test that percentile mode with value > 100 raises ValueError."""
        config = _make_valid_config()
        config["selection"]["mode"] = "percentile"
        config["selection"]["value"] = 101
        with pytest.raises(ValueError, match="percentile"):
            validate_config(config)

    def test_percentile_zero_raises(self):
        """Test that percentile mode with value = 0 raises ValueError."""
        config = _make_valid_config()
        config["selection"]["mode"] = "percentile"
        config["selection"]["value"] = 0
        with pytest.raises(ValueError, match="percentile"):
            validate_config(config)

    def test_threshold_negative_raises(self):
        """Test that threshold mode with negative value raises ValueError."""
        config = _make_valid_config()
        config["selection"]["mode"] = "threshold"
        config["selection"]["value"] = -1.0
        with pytest.raises(ValueError, match="positive number"):
            validate_config(config)

    def test_threshold_valid(self):
        """Test that threshold mode with positive value passes."""
        config = _make_valid_config()
        config["selection"]["mode"] = "threshold"
        config["selection"]["value"] = 5.0
        validate_config(config)

    # -- dataset_metrics section --

    def test_missing_dataset_metrics_raises(self):
        """Test that missing dataset_metrics raises KeyError."""
        config = _make_valid_config()
        del config["dataset_metrics"]
        with pytest.raises(KeyError, match="dataset_metrics"):
            validate_config(config)

    def test_missing_dataset_metrics_enabled_raises(self):
        """Test that missing dataset_metrics.enabled raises KeyError."""
        config = _make_valid_config()
        del config["dataset_metrics"]["enabled"]
        with pytest.raises(KeyError, match="dataset_metrics.enabled"):
            validate_config(config)

    def test_invalid_dataset_metrics_enabled_raises(self):
        """Test that non-boolean dataset_metrics.enabled raises ValueError."""
        config = _make_valid_config()
        config["dataset_metrics"]["enabled"] = "yes"
        with pytest.raises(ValueError, match="dataset_metrics.enabled"):
            validate_config(config)

    # -- logging section --

    def test_missing_logging_raises(self):
        """Test that missing logging section raises KeyError."""
        config = _make_valid_config()
        del config["logging"]
        with pytest.raises(KeyError, match="logging"):
            validate_config(config)

    def test_missing_console_level_raises(self):
        """Test that missing logging.console_level raises KeyError."""
        config = _make_valid_config()
        del config["logging"]["console_level"]
        with pytest.raises(KeyError, match="console_level"):
            validate_config(config)

    def test_missing_file_level_raises(self):
        """Test that missing logging.file_level raises KeyError."""
        config = _make_valid_config()
        del config["logging"]["file_level"]
        with pytest.raises(KeyError, match="file_level"):
            validate_config(config)

    def test_invalid_console_level_raises(self):
        """Test that invalid logging.console_level raises ValueError."""
        config = _make_valid_config()
        config["logging"]["console_level"] = "VERBOSE"
        with pytest.raises(ValueError, match="console_level"):
            validate_config(config)

    def test_invalid_file_level_raises(self):
        """Test that invalid logging.file_level raises ValueError."""
        config = _make_valid_config()
        config["logging"]["file_level"] = "TRACE"
        with pytest.raises(ValueError, match="file_level"):
            validate_config(config)

    # -- output section --

    def test_missing_output_raises(self):
        """Test that missing output section raises KeyError."""
        config = _make_valid_config()
        del config["output"]
        with pytest.raises(KeyError, match="output"):
            validate_config(config)

    def test_missing_output_subdirs_reports_raises(self):
        """Test that missing reports subdir raises ValueError."""
        config = _make_valid_config()
        del config["output"]["subdirs"]["reports"]
        with pytest.raises(ValueError, match="reports"):
            validate_config(config)

    def test_missing_output_subdirs_selected_raises(self):
        """Test that missing selected subdir raises ValueError."""
        config = _make_valid_config()
        del config["output"]["subdirs"]["selected"]
        with pytest.raises(ValueError, match="selected"):
            validate_config(config)
