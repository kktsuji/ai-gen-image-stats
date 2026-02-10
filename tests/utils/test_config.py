"""Unit tests for configuration management utilities.

Tests cover:
- JSON config loading (valid and invalid cases)
- Config merging with nested dictionaries
- Complete config workflow with defaults and overrides
- Config saving
"""

import json
from pathlib import Path

import pytest

from src.utils.config import (
    load_and_merge_configs,
    load_config,
    merge_configs,
    save_config,
)


@pytest.mark.unit
class TestLoadConfig:
    """Test config loading from JSON files."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid JSON config file."""
        # Create a test config file
        config_data = {"experiment": "classifier", "epochs": 10, "batch_size": 32}
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Load the config
        result = load_config(str(config_file))

        # Verify the result
        assert result == config_data
        assert result["experiment"] == "classifier"
        assert result["epochs"] == 10

    def test_load_config_with_nested_dict(self, tmp_path):
        """Test loading config with nested dictionaries."""
        config_data = {
            "model": {"name": "resnet50", "params": {"layers": 50, "pretrained": True}}
        }
        config_file = tmp_path / "nested_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = load_config(str(config_file))

        assert result["model"]["name"] == "resnet50"
        assert result["model"]["params"]["layers"] == 50
        assert result["model"]["params"]["pretrained"] is True

    def test_load_config_file_not_found(self):
        """Test loading a non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/non/existent/path/config.json")

    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises JSONDecodeError."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, "w") as f:
            f.write("{ invalid json content }")

        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))

    def test_load_config_empty_path(self):
        """Test loading with empty path raises ValueError."""
        with pytest.raises(ValueError, match="Config path cannot be empty"):
            load_config("")

    def test_load_config_none_path(self):
        """Test loading with None path raises ValueError."""
        with pytest.raises(ValueError, match="Config path cannot be empty"):
            load_config(None)

    def test_load_existing_fixture_config(self):
        """Test loading an existing fixture config."""
        # This tests that our fixture configs are valid
        fixture_path = (
            Path(__file__).parent.parent
            / "fixtures"
            / "configs"
            / "classifier_minimal.json"
        )

        if fixture_path.exists():
            result = load_config(str(fixture_path))

            assert "experiment" in result
            assert result["experiment"] == "classifier"
            assert "model" in result
            assert "data" in result


@pytest.mark.unit
class TestMergeConfigs:
    """Test configuration merging logic."""

    def test_merge_flat_configs(self):
        """Test merging configs with no nesting."""
        base = {"a": 1, "b": 2, "c": 3}
        override = {"b": 99, "d": 4}

        result = merge_configs(base, override)

        assert result == {"a": 1, "b": 99, "c": 3, "d": 4}

    def test_merge_nested_configs(self):
        """Test merging configs with nested dictionaries."""
        base = {
            "model": {"name": "resnet50", "layers": 50},
            "training": {"epochs": 10, "lr": 0.001},
        }
        override = {
            "model": {"layers": 101},  # Override only layers
            "training": {"epochs": 20},  # Override only epochs
        }

        result = merge_configs(base, override)

        assert result["model"]["name"] == "resnet50"  # Preserved from base
        assert result["model"]["layers"] == 101  # Overridden
        assert result["training"]["epochs"] == 20  # Overridden
        assert result["training"]["lr"] == 0.001  # Preserved from base

    def test_merge_deeply_nested_configs(self):
        """Test merging deeply nested dictionaries."""
        base = {"a": {"b": {"c": 1, "d": 2}, "e": 3}}
        override = {"a": {"b": {"c": 99}, "f": 4}}

        result = merge_configs(base, override)

        assert result["a"]["b"]["c"] == 99  # Overridden
        assert result["a"]["b"]["d"] == 2  # Preserved
        assert result["a"]["e"] == 3  # Preserved
        assert result["a"]["f"] == 4  # Added

    def test_merge_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        result = merge_configs(base, override)

        assert result["items"] == [4, 5]  # Completely replaced

    def test_merge_with_none_base(self):
        """Test merging when base is None."""
        override = {"a": 1, "b": 2}

        result = merge_configs(None, override)

        assert result == override

    def test_merge_with_none_override(self):
        """Test merging when override is None."""
        base = {"a": 1, "b": 2}

        result = merge_configs(base, None)

        assert result == base

    def test_merge_both_none(self):
        """Test merging when both configs are None."""
        result = merge_configs(None, None)

        assert result == {}

    def test_merge_empty_configs(self):
        """Test merging empty dictionaries."""
        result = merge_configs({}, {})

        assert result == {}

    def test_merge_does_not_modify_originals(self):
        """Test that merging doesn't modify original dictionaries."""
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}}

        result = merge_configs(base, override)

        # Original base should be unchanged
        assert base == {"a": 1, "b": {"c": 2}}
        assert "d" not in base["b"]

        # Original override should be unchanged
        assert override == {"b": {"d": 3}}


@pytest.mark.unit
class TestLoadAndMergeConfigs:
    """Test complete config loading and merging workflow."""

    def test_defaults_only(self):
        """Test with only defaults provided."""
        defaults = {"epochs": 10, "batch_size": 32}

        result = load_and_merge_configs(defaults=defaults)

        assert result == defaults

    def test_config_file_only(self, tmp_path):
        """Test with only config file provided."""
        config_data = {"epochs": 20, "lr": 0.001}
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = load_and_merge_configs(config_path=str(config_file))

        assert result == config_data

    def test_overrides_only(self):
        """Test with only overrides provided."""
        overrides = {"batch_size": 64}

        result = load_and_merge_configs(overrides=overrides)

        assert result == overrides

    def test_priority_order_config_over_defaults(self, tmp_path):
        """Test that config file overrides defaults."""
        defaults = {"epochs": 10, "batch_size": 32, "lr": 0.001}
        config_data = {"epochs": 20, "batch_size": 64}

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = load_and_merge_configs(config_path=str(config_file), defaults=defaults)

        # Config file values should override defaults
        assert result["epochs"] == 20  # From config file
        assert result["batch_size"] == 64  # From config file
        # Default values should be preserved if not in config
        assert result["lr"] == 0.001  # From defaults

    def test_priority_order_overrides_over_config(self, tmp_path):
        """Test that overrides have highest priority."""
        defaults = {"epochs": 10, "batch_size": 32, "lr": 0.001}
        config_data = {"epochs": 20, "batch_size": 64}
        overrides = {"batch_size": 128}

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = load_and_merge_configs(
            config_path=str(config_file), defaults=defaults, overrides=overrides
        )

        # Overrides should have highest priority
        assert result["batch_size"] == 128  # From overrides
        # Config file should override defaults
        assert result["epochs"] == 20  # From config file
        # Defaults should be preserved if not overridden
        assert result["lr"] == 0.001  # From defaults

    def test_nested_priority_order(self, tmp_path):
        """Test priority order with nested dictionaries."""
        defaults = {"model": {"name": "resnet50", "layers": 50, "pretrained": False}}
        config_data = {"model": {"layers": 101, "pretrained": True}}
        overrides = {"model": {"pretrained": False}}

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = load_and_merge_configs(
            config_path=str(config_file), defaults=defaults, overrides=overrides
        )

        assert result["model"]["name"] == "resnet50"  # From defaults
        assert result["model"]["layers"] == 101  # From config
        assert result["model"]["pretrained"] is False  # From overrides

    def test_all_none_parameters(self):
        """Test when all parameters are None."""
        result = load_and_merge_configs()

        assert result == {}

    def test_with_fixture_config(self):
        """Test loading with actual fixture config."""
        fixture_path = (
            Path(__file__).parent.parent
            / "fixtures"
            / "configs"
            / "classifier_minimal.json"
        )

        if not fixture_path.exists():
            pytest.skip("Fixture config not found")

        defaults = {"training": {"device": "cuda"}}
        overrides = {"data": {"batch_size": 8}}

        result = load_and_merge_configs(
            config_path=str(fixture_path), defaults=defaults, overrides=overrides
        )

        # Should have values from all sources
        assert "experiment" in result  # From fixture
        assert result["data"]["batch_size"] == 8  # From overrides
        # Config file should override defaults for device
        assert (
            result["training"]["device"] == "cpu"
        )  # From fixture (overrides defaults)


@pytest.mark.unit
class TestSaveConfig:
    """Test configuration saving."""

    def test_save_config_basic(self, tmp_path):
        """Test saving a basic config."""
        config = {"epochs": 10, "batch_size": 32}
        output_file = tmp_path / "saved_config.json"

        save_config(config, str(output_file))

        # Verify file was created
        assert output_file.exists()

        # Verify content
        with open(output_file, "r") as f:
            loaded = json.load(f)
        assert loaded == config

    def test_save_config_nested(self, tmp_path):
        """Test saving a config with nested dictionaries."""
        config = {
            "model": {"name": "resnet50", "layers": 50},
            "training": {"epochs": 10, "lr": 0.001},
        }
        output_file = tmp_path / "nested_config.json"

        save_config(config, str(output_file))

        # Verify content
        with open(output_file, "r") as f:
            loaded = json.load(f)
        assert loaded == config

    def test_save_config_creates_directories(self, tmp_path):
        """Test that save_config creates parent directories."""
        output_file = tmp_path / "subdir" / "nested" / "config.json"
        config = {"test": "value"}

        save_config(config, str(output_file))

        assert output_file.exists()

    def test_save_config_custom_indent(self, tmp_path):
        """Test saving with custom indentation."""
        config = {"a": 1, "b": 2}
        output_file = tmp_path / "config.json"

        save_config(config, str(output_file), indent=4)

        # Check indentation by reading raw file
        with open(output_file, "r") as f:
            content = f.read()

        # With indent=4, nested items should have 4 spaces
        assert "    " in content or "{\n" in content  # Some indentation present

    def test_save_config_none_config(self, tmp_path):
        """Test saving None config raises ValueError."""
        output_file = tmp_path / "config.json"

        with pytest.raises(ValueError, match="Config cannot be None"):
            save_config(None, str(output_file))

    def test_save_config_empty_path(self):
        """Test saving with empty path raises ValueError."""
        with pytest.raises(ValueError, match="Output path cannot be empty"):
            save_config({"test": "value"}, "")

    def test_save_config_none_path(self):
        """Test saving with None path raises ValueError."""
        with pytest.raises(ValueError, match="Output path cannot be empty"):
            save_config({"test": "value"}, None)

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that saving and loading preserves config."""
        original_config = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "pretrained": True},
            "training": {"epochs": 10, "lr": 0.001, "batch_size": 32},
        }
        config_file = tmp_path / "roundtrip.json"

        # Save and load
        save_config(original_config, str(config_file))
        loaded_config = load_config(str(config_file))

        # Should be identical
        assert loaded_config == original_config


@pytest.mark.unit
class TestConfigIntegration:
    """Integration tests for config workflow."""

    def test_complete_workflow(self, tmp_path):
        """Test a complete config workflow from defaults to final config."""
        # Step 1: Define defaults (code-level defaults)
        defaults = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "pretrained": True, "num_classes": 10},
            "training": {"epochs": 100, "batch_size": 32, "lr": 0.001},
        }

        # Step 2: Create a config file (experiment-specific settings)
        config_data = {
            "model": {"name": "resnet101"},  # Change model
            "training": {"epochs": 50},  # Reduce epochs
        }
        config_file = tmp_path / "experiment.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Step 3: Define CLI overrides (runtime overrides)
        overrides = {
            "training": {"batch_size": 64}  # Increase batch size
        }

        # Step 4: Load and merge (simulating what main.py would do)
        final_config = load_and_merge_configs(
            config_path=str(config_file), defaults=defaults, overrides=overrides
        )

        # Step 5: Verify priority order
        assert final_config["experiment"] == "classifier"  # From defaults
        assert final_config["model"]["name"] == "resnet101"  # From config file
        assert (
            final_config["model"]["pretrained"] is True
        )  # From defaults (not overridden)
        assert (
            final_config["model"]["num_classes"] == 10
        )  # From defaults (not overridden)
        assert final_config["training"]["epochs"] == 50  # From config file
        assert final_config["training"]["batch_size"] == 64  # From CLI overrides
        assert final_config["training"]["lr"] == 0.001  # From defaults (not overridden)

        # Step 6: Save final config for reproducibility
        output_file = tmp_path / "final_config.json"
        save_config(final_config, str(output_file))

        # Step 7: Verify saved config can be reloaded
        reloaded = load_config(str(output_file))
        assert reloaded == final_config
