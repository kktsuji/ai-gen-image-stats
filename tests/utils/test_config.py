"""Unit tests for configuration management utilities.

Tests cover:
- YAML config loading (valid and invalid cases)
- Config merging with nested dictionaries
- Complete config workflow with defaults and overrides
- Config saving
- Common validation functions
- Default config loading from modules
"""

from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    get_default_config_from_module,
    load_and_merge_configs,
    load_config,
    merge_configs,
    save_config,
    validate_compute_section,
    validate_data_loading_section,
    validate_optimizer_section,
    validate_output_section,
    validate_scheduler_section,
)


@pytest.mark.unit
class TestLoadConfig:
    """Test config loading from YAML files."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid YAML config file."""
        # Create a test config file
        config_data = {"experiment": "classifier", "epochs": 10, "batch_size": 32}
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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
        config_file = tmp_path / "nested_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        result = load_config(str(config_file))

        assert result["model"]["name"] == "resnet50"
        assert result["model"]["params"]["layers"] == 50
        assert result["model"]["params"]["pretrained"] is True

    def test_load_config_file_not_found(self):
        """Test loading a non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/non/existent/path/config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises YAMLError."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [unclosed")

        with pytest.raises(yaml.YAMLError):
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
            / "classifier_minimal.yaml"
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
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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
            / "classifier_minimal.yaml"
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
        output_file = tmp_path / "saved_config.yaml"

        save_config(config, str(output_file))

        # Verify file was created
        assert output_file.exists()

        # Verify content
        with open(output_file, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded == config

    def test_save_config_nested(self, tmp_path):
        """Test saving a config with nested dictionaries."""
        config = {
            "model": {"name": "resnet50", "layers": 50},
            "training": {"epochs": 10, "lr": 0.001},
        }
        output_file = tmp_path / "nested_config.yaml"

        save_config(config, str(output_file))

        # Verify content
        with open(output_file, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded == config

    def test_save_config_creates_directories(self, tmp_path):
        """Test that save_config creates parent directories."""
        output_file = tmp_path / "subdir" / "nested" / "config.yaml"
        config = {"test": "value"}

        save_config(config, str(output_file))

        assert output_file.exists()

    def test_save_config_custom_indent(self, tmp_path):
        """Test saving with custom indentation."""
        config = {"a": 1, "b": 2}
        output_file = tmp_path / "config.yaml"

        save_config(config, str(output_file), indent=4)

        # Check indentation by reading raw file
        with open(output_file, "r") as f:
            content = f.read()

        # YAML output should have line breaks for keys
        assert "\n" in content

    def test_save_config_none_config(self, tmp_path):
        """Test saving None config raises ValueError."""
        output_file = tmp_path / "config.yaml"

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
        config_file = tmp_path / "roundtrip.yaml"

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
        config_file = tmp_path / "experiment.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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
        output_file = tmp_path / "final_config.yaml"
        save_config(final_config, str(output_file))

        # Step 7: Verify saved config can be reloaded
        reloaded = load_config(str(output_file))
        assert reloaded == final_config


@pytest.mark.unit
class TestGetDefaultConfigFromModule:
    """Test get_default_config_from_module function."""

    def test_load_from_module_with_default_yaml(self, tmp_path):
        """Test loading default.yaml from module directory."""
        # Create a mock module directory with default.yaml
        module_dir = tmp_path / "mock_module"
        module_dir.mkdir()

        default_yaml = module_dir / "default.yaml"
        config_data = {
            "experiment": "test",
            "epochs": 100,
            "model": {"name": "test_model"},
        }
        with open(default_yaml, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Create a mock module file
        module_file = module_dir / "config.py"
        module_file.write_text("# Mock module file")

        # Load config
        result = get_default_config_from_module(str(module_file))

        assert result == config_data
        assert result["experiment"] == "test"
        assert result["epochs"] == 100

    def test_load_from_module_file_not_found(self, tmp_path):
        """Test error when default.yaml is missing."""
        # Create a module directory without default.yaml
        module_dir = tmp_path / "mock_module"
        module_dir.mkdir()

        module_file = module_dir / "config.py"
        module_file.write_text("# Mock module file")

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Default config not found"):
            get_default_config_from_module(str(module_file))

    def test_load_from_real_diffusion_module(self):
        """Test loading from actual diffusion module."""
        from src.experiments.diffusion import config

        result = get_default_config_from_module(config.__file__)

        # Verify it has expected structure
        assert "experiment" in result
        assert result["experiment"] == "diffusion"
        assert "model" in result
        assert "training" in result

    def test_load_from_real_classifier_module(self):
        """Test loading from actual classifier module."""
        from src.experiments.classifier import config

        result = get_default_config_from_module(config.__file__)

        # Verify it has expected structure
        assert "experiment" in result
        assert result["experiment"] == "classifier"
        assert "model" in result
        assert "training" in result


@pytest.mark.unit
class TestValidateComputeSection:
    """Test validate_compute_section function."""

    def test_valid_compute_config_cuda(self):
        """Test valid compute config with cuda device."""
        config = {"compute": {"device": "cuda", "seed": 42}}

        # Should not raise
        validate_compute_section(config)

    def test_valid_compute_config_cpu(self):
        """Test valid compute config with cpu device."""
        config = {"compute": {"device": "cpu", "seed": 0}}

        # Should not raise
        validate_compute_section(config)

    def test_valid_compute_config_auto(self):
        """Test valid compute config with auto device."""
        config = {"compute": {"device": "auto", "seed": None}}

        # Should not raise
        validate_compute_section(config)

    def test_valid_compute_config_no_seed(self):
        """Test valid compute config without seed."""
        config = {"compute": {"device": "cuda"}}

        # Should not raise
        validate_compute_section(config)

    def test_missing_compute_section(self):
        """Test error when compute section is missing."""
        config = {"model": {}}

        with pytest.raises(KeyError, match="Missing required config key: compute"):
            validate_compute_section(config)

    def test_invalid_device(self):
        """Test error with invalid device."""
        config = {"compute": {"device": "invalid_device"}}

        with pytest.raises(ValueError, match="Invalid device"):
            validate_compute_section(config)

    def test_negative_seed(self):
        """Test error with negative seed."""
        config = {"compute": {"device": "cuda", "seed": -1}}

        with pytest.raises(
            ValueError, match="seed must be None or a non-negative integer"
        ):
            validate_compute_section(config)

    def test_non_integer_seed(self):
        """Test error with non-integer seed."""
        config = {"compute": {"device": "cuda", "seed": "42"}}

        with pytest.raises(
            ValueError, match="seed must be None or a non-negative integer"
        ):
            validate_compute_section(config)

    def test_custom_valid_devices(self):
        """Test with custom valid devices list."""
        config = {"compute": {"device": "tpu"}}

        # Should work with custom valid_devices
        validate_compute_section(config, valid_devices=["cpu", "cuda", "tpu"])

        # Should fail with default valid_devices
        with pytest.raises(ValueError, match="Invalid device"):
            validate_compute_section(config)


@pytest.mark.unit
class TestValidateOutputSection:
    """Test validate_output_section function."""

    def test_valid_output_config(self):
        """Test valid output configuration."""
        config = {
            "output": {
                "base_dir": "outputs",
                "subdirs": {"logs": "logs", "checkpoints": "checkpoints"},
            }
        }

        # Should not raise
        validate_output_section(config)

    def test_valid_output_with_all_subdirs(self):
        """Test valid output with all common subdirs."""
        config = {
            "output": {
                "base_dir": "outputs",
                "subdirs": {
                    "logs": "logs",
                    "checkpoints": "checkpoints",
                    "samples": "samples",
                    "generated": "generated",
                },
            }
        }

        # Should not raise
        validate_output_section(config)

    def test_missing_output_section(self):
        """Test error when output section is missing."""
        config = {"model": {}}

        with pytest.raises(KeyError, match="Missing required config key: output"):
            validate_output_section(config)

    def test_missing_base_dir(self):
        """Test error when base_dir is missing."""
        config = {"output": {"subdirs": {"logs": "logs"}}}

        with pytest.raises(ValueError, match="output.base_dir is required"):
            validate_output_section(config)

    def test_none_base_dir(self):
        """Test error when base_dir is None."""
        config = {"output": {"base_dir": None, "subdirs": {"logs": "logs"}}}

        with pytest.raises(ValueError, match="output.base_dir is required"):
            validate_output_section(config)

    def test_missing_subdirs_section(self):
        """Test error when subdirs section is missing."""
        config = {"output": {"base_dir": "outputs"}}

        with pytest.raises(
            KeyError, match="Missing required config key: output.subdirs"
        ):
            validate_output_section(config)

    def test_missing_required_subdir(self):
        """Test error when required subdir is missing."""
        config = {
            "output": {
                "base_dir": "outputs",
                "subdirs": {
                    "logs": "logs"
                    # Missing checkpoints
                },
            }
        }

        with pytest.raises(ValueError, match="output.subdirs.checkpoints is required"):
            validate_output_section(config)

    def test_none_required_subdir(self):
        """Test error when required subdir is None."""
        config = {
            "output": {
                "base_dir": "outputs",
                "subdirs": {"logs": "logs", "checkpoints": None},
            }
        }

        with pytest.raises(ValueError, match="output.subdirs.checkpoints is required"):
            validate_output_section(config)

    def test_custom_required_subdirs(self):
        """Test with custom required subdirs."""
        config = {
            "output": {
                "base_dir": "outputs",
                "subdirs": {"models": "models", "results": "results"},
            }
        }

        # Should work with custom required subdirs
        validate_output_section(config, required_subdirs=["models", "results"])

        # Should fail with default required subdirs
        with pytest.raises(ValueError, match="output.subdirs.logs is required"):
            validate_output_section(config)


@pytest.mark.unit
class TestValidateDataLoadingSection:
    """Test validate_data_loading_section function."""

    def test_valid_data_loading_config(self):
        """Test valid data loading configuration."""
        data_config = {"loading": {"batch_size": 32, "num_workers": 4}}

        # Should not raise
        validate_data_loading_section(data_config)

    def test_valid_with_zero_workers(self):
        """Test valid config with zero workers."""
        data_config = {"loading": {"batch_size": 64, "num_workers": 0}}

        # Should not raise
        validate_data_loading_section(data_config)

    def test_missing_loading_section(self):
        """Test error when loading section is missing."""
        data_config = {"paths": {}}

        with pytest.raises(KeyError, match="Missing required config key: data.loading"):
            validate_data_loading_section(data_config)

    def test_missing_batch_size(self):
        """Test error when batch_size is missing."""
        data_config = {"loading": {"num_workers": 4}}

        with pytest.raises(
            KeyError, match="Missing required field: data.loading.batch_size"
        ):
            validate_data_loading_section(data_config)

    def test_missing_num_workers(self):
        """Test error when num_workers is missing."""
        data_config = {"loading": {"batch_size": 32}}

        with pytest.raises(
            KeyError, match="Missing required field: data.loading.num_workers"
        ):
            validate_data_loading_section(data_config)

    def test_none_batch_size(self):
        """Test error when batch_size is None."""
        data_config = {"loading": {"batch_size": None, "num_workers": 4}}

        with pytest.raises(ValueError, match="data.loading.batch_size cannot be None"):
            validate_data_loading_section(data_config)

    def test_none_num_workers(self):
        """Test error when num_workers is None."""
        data_config = {"loading": {"batch_size": 32, "num_workers": None}}

        with pytest.raises(ValueError, match="data.loading.num_workers cannot be None"):
            validate_data_loading_section(data_config)

    def test_zero_batch_size(self):
        """Test error with batch_size = 0."""
        data_config = {"loading": {"batch_size": 0, "num_workers": 4}}

        with pytest.raises(
            ValueError, match="data.loading.batch_size must be a positive integer"
        ):
            validate_data_loading_section(data_config)

    def test_negative_batch_size(self):
        """Test error with negative batch_size."""
        data_config = {"loading": {"batch_size": -32, "num_workers": 4}}

        with pytest.raises(
            ValueError, match="data.loading.batch_size must be a positive integer"
        ):
            validate_data_loading_section(data_config)

    def test_negative_num_workers(self):
        """Test error with negative num_workers."""
        data_config = {"loading": {"batch_size": 32, "num_workers": -1}}

        with pytest.raises(
            ValueError, match="data.loading.num_workers must be a non-negative integer"
        ):
            validate_data_loading_section(data_config)

    def test_non_integer_batch_size(self):
        """Test error with non-integer batch_size."""
        data_config = {"loading": {"batch_size": "32", "num_workers": 4}}

        with pytest.raises(
            ValueError, match="data.loading.batch_size must be a positive integer"
        ):
            validate_data_loading_section(data_config)

    def test_non_integer_num_workers(self):
        """Test error with non-integer num_workers."""
        data_config = {"loading": {"batch_size": 32, "num_workers": 4.5}}

        with pytest.raises(
            ValueError, match="data.loading.num_workers must be a non-negative integer"
        ):
            validate_data_loading_section(data_config)


@pytest.mark.unit
class TestValidateOptimizerSection:
    """Test validate_optimizer_section function."""

    def test_valid_optimizer_adam(self):
        """Test valid optimizer config with adam."""
        optimizer_config = {"type": "adam", "learning_rate": 0.001}

        # Should not raise
        validate_optimizer_section(optimizer_config)

    def test_valid_optimizer_adamw(self):
        """Test valid optimizer config with adamw."""
        optimizer_config = {"type": "adamw", "learning_rate": 0.0001}

        # Should not raise
        validate_optimizer_section(optimizer_config)

    def test_valid_optimizer_sgd(self):
        """Test valid optimizer config with sgd."""
        optimizer_config = {"type": "sgd", "learning_rate": 0.01}

        # Should not raise
        validate_optimizer_section(optimizer_config)

    def test_valid_with_gradient_clip(self):
        """Test valid config with gradient clipping."""
        optimizer_config = {
            "type": "adam",
            "learning_rate": 0.001,
            "gradient_clip_norm": 1.0,
        }

        # Should not raise
        validate_optimizer_section(optimizer_config)

    def test_valid_with_none_gradient_clip(self):
        """Test valid config with None gradient clipping."""
        optimizer_config = {
            "type": "adam",
            "learning_rate": 0.001,
            "gradient_clip_norm": None,
        }

        # Should not raise
        validate_optimizer_section(optimizer_config)

    def test_missing_type(self):
        """Test error when optimizer type is missing."""
        optimizer_config = {"learning_rate": 0.001}

        with pytest.raises(KeyError, match="Missing required field: optimizer.type"):
            validate_optimizer_section(optimizer_config)

    def test_missing_learning_rate(self):
        """Test error when learning_rate is missing."""
        optimizer_config = {"type": "adam"}

        with pytest.raises(
            KeyError, match="Missing required field: optimizer.learning_rate"
        ):
            validate_optimizer_section(optimizer_config)

    def test_invalid_optimizer_type(self):
        """Test error with invalid optimizer type."""
        optimizer_config = {"type": "invalid_optimizer", "learning_rate": 0.001}

        with pytest.raises(ValueError, match="Invalid optimizer type"):
            validate_optimizer_section(optimizer_config)

    def test_zero_learning_rate(self):
        """Test error with learning_rate = 0."""
        optimizer_config = {"type": "adam", "learning_rate": 0}

        with pytest.raises(
            ValueError, match="optimizer.learning_rate must be a positive number"
        ):
            validate_optimizer_section(optimizer_config)

    def test_negative_learning_rate(self):
        """Test error with negative learning_rate."""
        optimizer_config = {"type": "adam", "learning_rate": -0.001}

        with pytest.raises(
            ValueError, match="optimizer.learning_rate must be a positive number"
        ):
            validate_optimizer_section(optimizer_config)

    def test_zero_gradient_clip(self):
        """Test error with gradient_clip_norm = 0."""
        optimizer_config = {
            "type": "adam",
            "learning_rate": 0.001,
            "gradient_clip_norm": 0,
        }

        with pytest.raises(
            ValueError,
            match="optimizer.gradient_clip_norm must be a positive number or None",
        ):
            validate_optimizer_section(optimizer_config)

    def test_negative_gradient_clip(self):
        """Test error with negative gradient_clip_norm."""
        optimizer_config = {
            "type": "adam",
            "learning_rate": 0.001,
            "gradient_clip_norm": -1.0,
        }

        with pytest.raises(
            ValueError,
            match="optimizer.gradient_clip_norm must be a positive number or None",
        ):
            validate_optimizer_section(optimizer_config)

    def test_custom_valid_types(self):
        """Test with custom valid optimizer types."""
        optimizer_config = {"type": "rmsprop", "learning_rate": 0.001}

        # Should work with custom valid_types
        validate_optimizer_section(optimizer_config, valid_types=["adam", "rmsprop"])

        # Should fail with default valid_types
        with pytest.raises(ValueError, match="Invalid optimizer type"):
            validate_optimizer_section(optimizer_config)


@pytest.mark.unit
class TestValidateSchedulerSection:
    """Test validate_scheduler_section function."""

    def test_valid_scheduler_cosine(self):
        """Test valid scheduler config with cosine."""
        scheduler_config = {"type": "cosine"}

        # Should not raise
        validate_scheduler_section(scheduler_config)

    def test_valid_scheduler_step(self):
        """Test valid scheduler config with step."""
        scheduler_config = {"type": "step"}

        # Should not raise
        validate_scheduler_section(scheduler_config)

    def test_valid_scheduler_plateau(self):
        """Test valid scheduler config with plateau."""
        scheduler_config = {"type": "plateau"}

        # Should not raise
        validate_scheduler_section(scheduler_config)

    def test_valid_scheduler_none(self):
        """Test valid scheduler config with None."""
        scheduler_config = {"type": None}

        # Should not raise
        validate_scheduler_section(scheduler_config)

    def test_missing_type(self):
        """Test error when scheduler type is missing."""
        scheduler_config = {}

        with pytest.raises(KeyError, match="Missing required field: scheduler.type"):
            validate_scheduler_section(scheduler_config)

    def test_invalid_scheduler_type(self):
        """Test error with invalid scheduler type."""
        scheduler_config = {"type": "invalid_scheduler"}

        with pytest.raises(ValueError, match="Invalid scheduler type"):
            validate_scheduler_section(scheduler_config)

    def test_custom_valid_types(self):
        """Test with custom valid scheduler types."""
        scheduler_config = {"type": "exponential"}

        # Should work with custom valid_types
        validate_scheduler_section(
            scheduler_config, valid_types=["cosine", "exponential"]
        )

        # Should fail with default valid_types
        with pytest.raises(ValueError, match="Invalid scheduler type"):
            validate_scheduler_section(scheduler_config)

    def test_custom_valid_types_without_none(self):
        """Test with custom valid types that don't include None."""
        scheduler_config = {"type": None}

        # Should work with default valid_types (includes None)
        validate_scheduler_section(scheduler_config)

        # Should fail with custom valid_types that don't include None
        with pytest.raises(ValueError, match="Invalid scheduler type"):
            validate_scheduler_section(scheduler_config, valid_types=["cosine", "step"])
