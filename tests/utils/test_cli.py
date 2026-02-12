"""Unit tests for CLI utilities.

Tests cover:
- Argument parser creation and configuration (config-only mode)
- CLI argument parsing with config file
- Config validation
- Error handling for missing/invalid configs
"""

import json
from pathlib import Path

import pytest

from src.utils.cli import create_parser, parse_args, validate_config


@pytest.mark.unit
class TestCreateParser:
    """Test argument parser creation for config-only mode."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.description is not None

    def test_parser_has_positional_config_argument(self):
        """Test that parser has positional config_path argument."""
        parser = create_parser()
        # Parse with config path
        args = parser.parse_args(["configs/test.json"])
        assert args.config_path == "configs/test.json"

    def test_config_path_is_required(self):
        """Test that config_path argument is required."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # No arguments should fail

    def test_parser_accepts_verbose_flag(self):
        """Test that parser accepts optional --verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["configs/test.json", "--verbose"])
        assert args.config_path == "configs/test.json"
        assert args.verbose is True

    def test_parser_rejects_unknown_arguments(self):
        """Test that parser rejects unknown CLI arguments."""
        parser = create_parser()
        # Should fail on unknown arguments like --epochs
        with pytest.raises(SystemExit):
            parser.parse_args(["configs/test.json", "--epochs", "10"])


@pytest.mark.unit
class TestParseArgs:
    """Test the main parse_args function with config-only mode."""

    def test_parse_args_with_config_file(self, tmp_path):
        """Test parsing with a valid config file."""
        # Create config file
        config_data = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "pretrained": True, "num_classes": 2},
            "training": {
                "epochs": 20,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "data": {
                "train_path": "data/train",
                "batch_size": 32,
                "num_workers": 4,
                "image_size": 256,
                "crop_size": 224,
            },
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = parse_args([str(config_file)])

        assert config["experiment"] == "classifier"
        assert config["model"]["name"] == "resnet50"
        assert config["training"]["epochs"] == 20
        assert config["training"]["learning_rate"] == 0.001
        assert config["data"]["batch_size"] == 32

    def test_parse_args_requires_config(self):
        """Test that config file is required."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_parse_args_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            parse_args(["nonexistent_config.json"])

    def test_parse_args_missing_experiment_field(self, tmp_path):
        """Test that config without experiment field raises error."""
        config_data = {"training": {"epochs": 10}}
        config_file = tmp_path / "no_experiment.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="Missing required 'experiment' field"):
            parse_args([str(config_file)])

    def test_parse_args_invalid_experiment_type(self, tmp_path):
        """Test that invalid experiment type raises error."""
        config_data = {"experiment": "invalid_type"}
        config_file = tmp_path / "invalid_exp.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="Invalid experiment type"):
            parse_args([str(config_file)])

    def test_parse_args_reads_experiment_from_config(self, tmp_path):
        """Test that experiment type is read from config, not CLI."""
        config_data = {
            "experiment": "diffusion",
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 64,
                "num_timesteps": 1000,
                "beta_schedule": "linear",
            },
            "training": {
                "epochs": 10,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "data": {
                "train_path": "data/train",
                "batch_size": 16,
                "num_workers": 2,
                "image_size": 32,
            },
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
        }
        config_file = tmp_path / "diffusion_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = parse_args([str(config_file)])
        assert config["experiment"] == "diffusion"

    def test_parse_args_with_verbose_flag(self, tmp_path):
        """Test that --verbose flag is added to config."""
        config_data = {"experiment": "classifier", "model": {"name": "resnet50"}}
        config_file = tmp_path / "test.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = parse_args([str(config_file), "--verbose"])
        assert config["verbose"] is True


@pytest.mark.unit
class TestValidateConfig:
    """Test configuration validation."""

    def test_validate_valid_config(self):
        """Test that valid config passes validation."""
        config = {
            "experiment": "classifier",
            "training": {"epochs": 10, "learning_rate": 0.001},
            "data": {"batch_size": 32},
        }
        # Should not raise
        validate_config(config)

    def test_validate_missing_experiment(self):
        """Test that missing experiment field raises error."""
        config = {"training": {"epochs": 10}}

        with pytest.raises(ValueError, match="'experiment' field is required"):
            validate_config(config)

    def test_validate_invalid_experiment_type(self):
        """Test that invalid experiment type raises error."""
        config = {"experiment": "invalid_type"}

        with pytest.raises(ValueError, match="Invalid experiment type"):
            validate_config(config)

    def test_validate_valid_experiment_types(self):
        """Test that all valid experiment types pass validation."""
        for exp_type in ["classifier", "diffusion", "gan"]:
            config = {"experiment": exp_type}
            validate_config(config)  # Should not raise


@pytest.mark.unit
class TestConfigOnlyMode:
    """Test that config-only mode is enforced."""

    def test_requires_config_file(self):
        """Test that config file is mandatory."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_no_cli_overrides_allowed(self):
        """Test that CLI parameter overrides are not accepted."""
        parser = create_parser()
        # Should reject any parameter-style arguments
        with pytest.raises(SystemExit):
            parser.parse_args(["config.json", "--epochs", "10"])
        with pytest.raises(SystemExit):
            parser.parse_args(["config.json", "--batch-size", "64"])
        with pytest.raises(SystemExit):
            parser.parse_args(["config.json", "--model", "resnet50"])

    def test_loads_config_only(self, tmp_path):
        """Test that all settings come from config file only."""
        config_data = {
            "experiment": "classifier",
            "model": {"name": "inceptionv3", "pretrained": True, "num_classes": 10},
            "training": {
                "epochs": 50,
                "learning_rate": 0.0002,
                "optimizer": "adamw",
                "device": "cpu",
            },
            "data": {
                "train_path": "data/train",
                "batch_size": 64,
                "num_workers": 8,
                "image_size": 256,
                "crop_size": 224,
            },
            "output": {"checkpoint_dir": "outputs/ckpts", "log_dir": "outputs/logs"},
        }
        config_file = tmp_path / "full_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = parse_args([str(config_file)])

        # All values should match config file exactly
        assert config["experiment"] == "classifier"
        assert config["model"]["name"] == "inceptionv3"
        assert config["model"]["pretrained"] is True
        assert config["model"]["num_classes"] == 10
        assert config["training"]["epochs"] == 50
        assert config["training"]["learning_rate"] == 0.0002
        assert config["training"]["optimizer"] == "adamw"
        assert config["data"]["batch_size"] == 64
        assert config["data"]["num_workers"] == 8
        assert config["output"]["checkpoint_dir"] == "outputs/ckpts"

    def test_experiment_from_config(self, tmp_path):
        """Test that experiment type comes from config, not CLI."""
        for exp_type in ["classifier", "diffusion", "gan"]:
            config_data = {"experiment": exp_type}
            config_file = tmp_path / f"{exp_type}_config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            config = parse_args([str(config_file)])
            assert config["experiment"] == exp_type


@pytest.mark.unit
class TestEndToEndCLIWorkflow:
    """Test complete CLI workflows with config-only mode."""

    def test_classifier_training_workflow(self, tmp_path):
        """Test a classifier training workflow with config file."""
        config_data = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "pretrained": True, "num_classes": 2},
            "training": {
                "epochs": 10,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "data": {
                "train_path": "data/train",
                "batch_size": 32,
                "num_workers": 4,
                "image_size": 256,
                "crop_size": 224,
            },
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
        }
        config_file = tmp_path / "classifier.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = parse_args([str(config_file)])
        validate_config(config)

        assert config["experiment"] == "classifier"
        assert config["model"]["name"] == "resnet50"
        assert config["training"]["epochs"] == 10
        assert config["data"]["batch_size"] == 32

    def test_diffusion_training_workflow(self, tmp_path):
        """Test a diffusion training workflow with config file."""
        config_data = {
            "experiment": "diffusion",
            "model": {
                "image_size": 64,
                "in_channels": 3,
                "model_channels": 128,
                "num_timesteps": 1000,
                "beta_schedule": "linear",
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "data": {
                "train_path": "data/train",
                "batch_size": 16,
                "num_workers": 4,
                "image_size": 64,
            },
            "output": {
                "checkpoint_dir": "outputs/diffusion/checkpoints",
                "log_dir": "outputs/diffusion/logs",
            },
        }
        config_file = tmp_path / "diffusion.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = parse_args([str(config_file)])
        validate_config(config)

        assert config["experiment"] == "diffusion"
        assert config["model"]["image_size"] == 64
        assert config["training"]["epochs"] == 100

    def test_config_file_required_no_defaults(self):
        """Test that there are no defaults - config file is mandatory."""
        # Without config file, should fail
        with pytest.raises(SystemExit):
            parse_args([])

    def test_error_on_missing_config_fields(self, tmp_path):
        """Test that missing required fields in config cause errors during experiment-specific validation."""
        # This test just verifies basic CLI validation passes,
        # but experiment-specific validators (in classifier/config.py, diffusion/config.py)
        # will catch missing required fields
        config_data = {
            "experiment": "classifier",
            # Missing model, training, data, output sections
        }
        config_file = tmp_path / "incomplete.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Basic CLI validation should pass (only checks experiment field)
        config = parse_args([str(config_file)])
        validate_config(config)  # Should pass basic validation

        # Experiment-specific validation would fail (tested in experiment config tests)
        assert config["experiment"] == "classifier"
        assert "model" not in config  # Missing required section
