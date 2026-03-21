"""Tests for Data Preparation Configuration

This module tests the data preparation configuration validation.
"""

import copy
from pathlib import Path

import pytest
import yaml

from src.experiments.data_preparation.config import validate_config

# Resolve project root so config file tests work regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _make_valid_config():
    """Create a valid config for testing.

    Loads from configs/examples/data-preparation.yaml to stay in sync with the
    canonical example config (single source of truth).
    """
    config_path = _PROJECT_ROOT / "configs/examples/data-preparation.yaml"
    with open(config_path) as f:
        return copy.deepcopy(yaml.safe_load(f))


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
        with pytest.raises(ValueError, match="Must be 'data_preparation'"):
            validate_config(config)

    def test_missing_classes_raises(self):
        """Test that missing classes raises KeyError."""
        config = _make_valid_config()
        del config["classes"]
        with pytest.raises(KeyError, match="classes"):
            validate_config(config)

    def test_empty_classes_raises(self):
        """Test that empty classes dict raises ValueError."""
        config = _make_valid_config()
        config["classes"] = {}
        with pytest.raises(ValueError, match="non-empty"):
            validate_config(config)

    def test_missing_split_raises(self):
        """Test that missing split section raises KeyError."""
        config = _make_valid_config()
        del config["split"]
        with pytest.raises(KeyError, match="split"):
            validate_config(config)

    def test_invalid_train_ratio_zero_raises(self):
        """Test that train_ratio=0 raises ValueError."""
        config = _make_valid_config()
        config["split"]["train_ratio"] = 0.0
        with pytest.raises(ValueError, match="train_ratio"):
            validate_config(config)

    def test_invalid_train_ratio_one_raises(self):
        """Test that train_ratio=1 raises ValueError."""
        config = _make_valid_config()
        config["split"]["train_ratio"] = 1.0
        with pytest.raises(ValueError, match="train_ratio"):
            validate_config(config)

    def test_invalid_train_ratio_negative_raises(self):
        """Test that negative train_ratio raises ValueError."""
        config = _make_valid_config()
        config["split"]["train_ratio"] = -0.5
        with pytest.raises(ValueError, match="train_ratio"):
            validate_config(config)

    def test_invalid_train_ratio_greater_than_one_raises(self):
        """Test that train_ratio > 1 raises ValueError."""
        config = _make_valid_config()
        config["split"]["train_ratio"] = 1.5
        with pytest.raises(ValueError, match="train_ratio"):
            validate_config(config)

    def test_missing_split_file_raises(self):
        """Test that missing split_file raises KeyError."""
        config = _make_valid_config()
        del config["split"]["split_file"]
        with pytest.raises(KeyError, match="split_file"):
            validate_config(config)

    def test_split_file_not_json_raises(self):
        """Test that split_file not ending in .json raises ValueError."""
        config = _make_valid_config()
        config["split"]["split_file"] = "split.yaml"
        with pytest.raises(ValueError, match=".json"):
            validate_config(config)

    def test_missing_train_ratio_raises(self):
        """Test that missing train_ratio raises KeyError."""
        config = _make_valid_config()
        del config["split"]["train_ratio"]
        with pytest.raises(KeyError, match="train_ratio"):
            validate_config(config)

    def test_missing_save_dir_raises(self):
        """Test that missing save_dir raises KeyError."""
        config = _make_valid_config()
        del config["split"]["save_dir"]
        with pytest.raises(KeyError, match="save_dir"):
            validate_config(config)

    def test_null_seed_is_valid(self):
        """Test that null seed passes validation."""
        config = _make_valid_config()
        config["split"]["seed"] = None
        validate_config(config)  # Should not raise

    def test_invalid_force_type_raises(self):
        """Test that non-boolean force raises ValueError."""
        config = _make_valid_config()
        config["split"]["force"] = "yes"
        with pytest.raises(ValueError, match="force"):
            validate_config(config)

    def test_class_with_empty_path_raises(self):
        """Test that class with empty string path raises ValueError."""
        config = _make_valid_config()
        config["classes"]["normal"] = {"path": "", "label": 0}
        with pytest.raises(ValueError, match="non-empty string"):
            validate_config(config)

    def test_class_not_dict_raises(self):
        """Test that class with string value (old format) raises ValueError."""
        config = _make_valid_config()
        config["classes"]["normal"] = "data/path"
        with pytest.raises(ValueError, match="dict"):
            validate_config(config)

    def test_class_missing_path_raises(self):
        """Test that class without path key raises KeyError."""
        config = _make_valid_config()
        config["classes"]["normal"] = {"label": 0}
        with pytest.raises(KeyError, match="path"):
            validate_config(config)

    def test_class_missing_label_raises(self):
        """Test that class without label key raises KeyError."""
        config = _make_valid_config()
        config["classes"]["normal"] = {"path": "data/x", "label": 0}
        config["classes"]["abnormal"] = {"path": "data/y"}
        with pytest.raises(KeyError, match="label"):
            validate_config(config)

    def test_class_negative_label_raises(self):
        """Test that negative label raises ValueError."""
        config = _make_valid_config()
        config["classes"]["normal"] = {"path": "data/x", "label": -1}
        with pytest.raises(ValueError, match="non-negative"):
            validate_config(config)

    def test_class_non_int_label_raises(self):
        """Test that non-integer label raises ValueError."""
        config = _make_valid_config()
        config["classes"]["normal"] = {"path": "data/x", "label": "0"}
        with pytest.raises(ValueError, match="non-negative integer"):
            validate_config(config)

    def test_bool_label_raises(self):
        """Test that boolean label raises ValueError (bool is subclass of int)."""
        config = _make_valid_config()
        config["classes"]["normal"] = {"path": "data/x", "label": True}
        config["classes"]["abnormal"] = {"path": "data/y", "label": False}
        with pytest.raises(ValueError, match="non-negative integer"):
            validate_config(config)

    def test_duplicate_labels_raises(self):
        """Test that duplicate labels raise ValueError."""
        config = _make_valid_config()
        config["classes"]["normal"] = {"path": "data/x", "label": 0}
        config["classes"]["abnormal"] = {"path": "data/y", "label": 0}
        with pytest.raises(ValueError, match="unique"):
            validate_config(config)

    def test_non_contiguous_labels_raises(self):
        """Test that non-contiguous labels raise ValueError."""
        config = _make_valid_config()
        config["classes"]["normal"] = {"path": "data/x", "label": 0}
        config["classes"]["abnormal"] = {"path": "data/y", "label": 2}
        with pytest.raises(ValueError, match="contiguous"):
            validate_config(config)

    def test_valid_three_classes(self):
        """Test that three classes with contiguous labels passes."""
        config = _make_valid_config()
        config["classes"] = {
            "a": {"path": "data/a", "label": 0},
            "b": {"path": "data/b", "label": 1},
            "c": {"path": "data/c", "label": 2},
        }
        validate_config(config)  # Should not raise
