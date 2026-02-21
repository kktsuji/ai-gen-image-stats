"""Tests for Data Preparation Configuration

This module tests the data preparation configuration validation.
"""

import pytest

from src.experiments.data_preparation.config import get_default_config, validate_config

# ============================================================================
# Unit Tests - Fast, Pure Logic
# ============================================================================


@pytest.mark.unit
class TestGetDefaultConfig:
    """Test default configuration generation."""

    def test_returns_dict(self):
        """Test that get_default_config returns a dictionary."""
        config = get_default_config()
        assert isinstance(config, dict)

    def test_has_required_keys(self):
        """Test that default config has all required top-level keys."""
        config = get_default_config()
        required_keys = ["experiment", "classes", "split"]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_experiment_type(self):
        """Test that experiment type is 'data_preparation'."""
        config = get_default_config()
        assert config["experiment"] == "data_preparation"

    def test_classes_defaults(self):
        """Test classes configuration defaults."""
        config = get_default_config()
        classes = config["classes"]
        assert isinstance(classes, dict)
        assert len(classes) > 0
        assert "normal" in classes
        assert "abnormal" in classes

    def test_split_defaults(self):
        """Test split configuration defaults."""
        config = get_default_config()
        split = config["split"]
        assert isinstance(split, dict)
        assert "seed" in split
        assert "train_ratio" in split
        assert "save_dir" in split
        assert "split_file" in split
        assert split["train_ratio"] == 0.8
        assert split["split_file"].endswith(".json")


@pytest.mark.unit
class TestValidateConfig:
    """Test configuration validation."""

    def _make_valid_config(self):
        """Create a valid config for testing."""
        return {
            "experiment": "data_preparation",
            "classes": {
                "normal": "data/0.Normal",
                "abnormal": "data/1.Abnormal",
            },
            "split": {
                "seed": 42,
                "train_ratio": 0.8,
                "save_dir": "outputs/splits",
                "split_file": "train_val_split.json",
                "force": False,
            },
        }

    def test_valid_config_passes(self):
        """Test that a valid config passes validation."""
        config = self._make_valid_config()
        validate_config(config)  # Should not raise

    def test_wrong_experiment_type_raises(self):
        """Test that wrong experiment type raises ValueError."""
        config = self._make_valid_config()
        config["experiment"] = "diffusion"
        with pytest.raises(ValueError, match="Must be 'data_preparation'"):
            validate_config(config)

    def test_missing_classes_raises(self):
        """Test that missing classes raises KeyError."""
        config = self._make_valid_config()
        del config["classes"]
        with pytest.raises(KeyError, match="classes"):
            validate_config(config)

    def test_empty_classes_raises(self):
        """Test that empty classes dict raises ValueError."""
        config = self._make_valid_config()
        config["classes"] = {}
        with pytest.raises(ValueError, match="non-empty"):
            validate_config(config)

    def test_missing_split_raises(self):
        """Test that missing split section raises KeyError."""
        config = self._make_valid_config()
        del config["split"]
        with pytest.raises(KeyError, match="split"):
            validate_config(config)

    def test_invalid_train_ratio_zero_raises(self):
        """Test that train_ratio=0 raises ValueError."""
        config = self._make_valid_config()
        config["split"]["train_ratio"] = 0.0
        with pytest.raises(ValueError, match="train_ratio"):
            validate_config(config)

    def test_invalid_train_ratio_one_raises(self):
        """Test that train_ratio=1 raises ValueError."""
        config = self._make_valid_config()
        config["split"]["train_ratio"] = 1.0
        with pytest.raises(ValueError, match="train_ratio"):
            validate_config(config)

    def test_invalid_train_ratio_negative_raises(self):
        """Test that negative train_ratio raises ValueError."""
        config = self._make_valid_config()
        config["split"]["train_ratio"] = -0.5
        with pytest.raises(ValueError, match="train_ratio"):
            validate_config(config)

    def test_invalid_train_ratio_greater_than_one_raises(self):
        """Test that train_ratio > 1 raises ValueError."""
        config = self._make_valid_config()
        config["split"]["train_ratio"] = 1.5
        with pytest.raises(ValueError, match="train_ratio"):
            validate_config(config)

    def test_missing_split_file_raises(self):
        """Test that missing split_file raises KeyError."""
        config = self._make_valid_config()
        del config["split"]["split_file"]
        with pytest.raises(KeyError, match="split_file"):
            validate_config(config)

    def test_split_file_not_json_raises(self):
        """Test that split_file not ending in .json raises ValueError."""
        config = self._make_valid_config()
        config["split"]["split_file"] = "split.yaml"
        with pytest.raises(ValueError, match=".json"):
            validate_config(config)

    def test_missing_train_ratio_raises(self):
        """Test that missing train_ratio raises KeyError."""
        config = self._make_valid_config()
        del config["split"]["train_ratio"]
        with pytest.raises(KeyError, match="train_ratio"):
            validate_config(config)

    def test_missing_save_dir_raises(self):
        """Test that missing save_dir raises KeyError."""
        config = self._make_valid_config()
        del config["split"]["save_dir"]
        with pytest.raises(KeyError, match="save_dir"):
            validate_config(config)

    def test_null_seed_is_valid(self):
        """Test that null seed passes validation."""
        config = self._make_valid_config()
        config["split"]["seed"] = None
        validate_config(config)  # Should not raise

    def test_invalid_force_type_raises(self):
        """Test that non-boolean force raises ValueError."""
        config = self._make_valid_config()
        config["split"]["force"] = "yes"
        with pytest.raises(ValueError, match="force"):
            validate_config(config)

    def test_class_with_empty_path_raises(self):
        """Test that class with empty string path raises ValueError."""
        config = self._make_valid_config()
        config["classes"]["normal"] = ""
        with pytest.raises(ValueError, match="non-empty string"):
            validate_config(config)
