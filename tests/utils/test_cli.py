"""Unit tests for CLI utilities.

Tests cover:
- Argument parser creation and configuration
- CLI argument parsing with various options
- Converting arguments to nested config dictionaries
- Config merging with CLI overrides
- Config validation
"""

import argparse
import json
from pathlib import Path

import pytest

from src.utils.cli import args_to_dict, create_parser, parse_args, validate_config


@pytest.mark.unit
class TestCreateParser:
    """Test argument parser creation."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description is not None

    def test_parser_has_experiment_argument(self):
        """Test that parser has required experiment argument."""
        parser = create_parser()
        # Parse with experiment argument
        args = parser.parse_args(["--experiment", "classifier"])
        assert args.experiment == "classifier"

    def test_experiment_choices(self):
        """Test that experiment argument only accepts valid choices."""
        parser = create_parser()

        # Valid choices
        for exp in ["classifier", "diffusion", "gan"]:
            args = parser.parse_args(["--experiment", exp])
            assert args.experiment == exp

        # Invalid choice should raise error
        with pytest.raises(SystemExit):
            parser.parse_args(["--experiment", "invalid"])

    def test_experiment_is_required(self):
        """Test that experiment argument is required."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # No arguments


@pytest.mark.unit
class TestParseTrainingArguments:
    """Test parsing of training-related arguments."""

    def test_parse_epochs(self):
        """Test parsing epochs argument."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "classifier", "--epochs", "100"])
        assert args.epochs == 100

    def test_parse_batch_size(self):
        """Test parsing batch-size argument."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "gan", "--batch-size", "64"])
        assert args.batch_size == 64

    def test_parse_learning_rate(self):
        """Test parsing learning rate argument."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "diffusion", "--lr", "0.001"])
        assert args.lr == 0.001
        assert isinstance(args.lr, float)

    def test_parse_optimizer(self):
        """Test parsing optimizer argument."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "classifier", "--optimizer", "adam"])
        assert args.optimizer == "adam"

    def test_optimizer_choices(self):
        """Test that optimizer only accepts valid choices."""
        parser = create_parser()

        # Valid choices
        for opt in ["adam", "sgd", "adamw"]:
            args = parser.parse_args(["--experiment", "classifier", "--optimizer", opt])
            assert args.optimizer == opt

        # Invalid choice should raise error
        with pytest.raises(SystemExit):
            parser.parse_args(["--experiment", "classifier", "--optimizer", "invalid"])


@pytest.mark.unit
class TestParseModelArguments:
    """Test parsing of model-related arguments."""

    def test_parse_model_name(self):
        """Test parsing model name argument."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "classifier", "--model", "resnet50"])
        assert args.model == "resnet50"

    def test_parse_pretrained_flag(self):
        """Test parsing pretrained flag."""
        parser = create_parser()

        # With flag
        args = parser.parse_args(["--experiment", "classifier", "--pretrained"])
        assert args.pretrained is True

        # Without flag
        args = parser.parse_args(["--experiment", "classifier"])
        assert args.pretrained is None

    def test_parse_num_classes(self):
        """Test parsing num-classes argument."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "classifier", "--num-classes", "10"])
        assert args.num_classes == 10


@pytest.mark.unit
class TestParseDataArguments:
    """Test parsing of data-related arguments."""

    def test_parse_data_paths(self):
        """Test parsing train and validation paths."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--experiment",
                "classifier",
                "--train-path",
                "data/train",
                "--val-path",
                "data/val",
            ]
        )
        assert args.train_path == "data/train"
        assert args.val_path == "data/val"

    def test_parse_num_workers(self):
        """Test parsing num-workers argument."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "gan", "--num-workers", "4"])
        assert args.num_workers == 4


@pytest.mark.unit
class TestParseOutputArguments:
    """Test parsing of output-related arguments."""

    def test_parse_output_directories(self):
        """Test parsing output directory arguments."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--experiment",
                "diffusion",
                "--output-dir",
                "outputs",
                "--checkpoint-dir",
                "outputs/checkpoints",
                "--log-dir",
                "outputs/logs",
            ]
        )
        assert args.output_dir == "outputs"
        assert args.checkpoint_dir == "outputs/checkpoints"
        assert args.log_dir == "outputs/logs"

    def test_parse_checkpoint_path(self):
        """Test parsing checkpoint path argument."""
        parser = create_parser()
        args = parser.parse_args(
            ["--experiment", "gan", "--checkpoint", "checkpoints/model.pth"]
        )
        assert args.checkpoint == "checkpoints/model.pth"


@pytest.mark.unit
class TestParseModeArguments:
    """Test parsing of mode and generation arguments."""

    def test_parse_mode_default(self):
        """Test that mode defaults to train."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "classifier"])
        assert args.mode == "train"

    def test_parse_mode_choices(self):
        """Test parsing different mode options."""
        parser = create_parser()

        for mode in ["train", "generate", "evaluate"]:
            args = parser.parse_args(["--experiment", "diffusion", "--mode", mode])
            assert args.mode == mode

        # Invalid mode
        with pytest.raises(SystemExit):
            parser.parse_args(["--experiment", "diffusion", "--mode", "invalid"])

    def test_parse_num_samples(self):
        """Test parsing num-samples argument."""
        parser = create_parser()
        args = parser.parse_args(
            ["--experiment", "diffusion", "--mode", "generate", "--num-samples", "1000"]
        )
        assert args.num_samples == 1000


@pytest.mark.unit
class TestParseDeviceArguments:
    """Test parsing of device-related arguments."""

    def test_parse_device(self):
        """Test parsing device argument."""
        parser = create_parser()

        for device in ["cpu", "cuda", "auto"]:
            args = parser.parse_args(["--experiment", "classifier", "--device", device])
            assert args.device == device

    def test_parse_seed(self):
        """Test parsing random seed argument."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "gan", "--seed", "42"])
        assert args.seed == 42


@pytest.mark.unit
class TestParseConfigFile:
    """Test parsing config file argument."""

    def test_parse_config_path(self):
        """Test parsing config file path."""
        parser = create_parser()
        args = parser.parse_args(
            ["--experiment", "classifier", "--config", "configs/experiment.json"]
        )
        assert args.config == "configs/experiment.json"

    def test_config_file_optional(self):
        """Test that config file is optional."""
        parser = create_parser()
        args = parser.parse_args(["--experiment", "diffusion"])
        assert args.config is None


@pytest.mark.unit
class TestArgsToDict:
    """Test converting argparse Namespace to nested config dict."""

    def test_simple_args_to_dict(self):
        """Test converting simple arguments."""
        args = argparse.Namespace(experiment="classifier", mode="train")
        result = args_to_dict(args)

        assert result["experiment"] == "classifier"
        assert result["mode"] == "train"

    def test_nested_training_args(self):
        """Test that training args are nested correctly."""
        args = argparse.Namespace(
            experiment="classifier", epochs=100, lr=0.001, optimizer="adam"
        )
        result = args_to_dict(args)

        assert result["experiment"] == "classifier"
        assert result["training"]["epochs"] == 100
        assert result["training"]["learning_rate"] == 0.001
        assert result["training"]["optimizer"] == "adam"

    def test_nested_model_args(self):
        """Test that model args are nested correctly."""
        args = argparse.Namespace(
            experiment="classifier", model="resnet50", pretrained=True, num_classes=10
        )
        result = args_to_dict(args)

        assert result["model"]["name"] == "resnet50"
        assert result["model"]["pretrained"] is True
        assert result["model"]["num_classes"] == 10

    def test_nested_data_args(self):
        """Test that data args are nested correctly."""
        args = argparse.Namespace(
            experiment="gan",
            train_path="data/train",
            val_path="data/val",
            batch_size=64,
            num_workers=4,
        )
        result = args_to_dict(args)

        assert result["data"]["train_path"] == "data/train"
        assert result["data"]["val_path"] == "data/val"
        assert result["data"]["batch_size"] == 64
        assert result["data"]["num_workers"] == 4

    def test_nested_output_args(self):
        """Test that output args are nested correctly."""
        args = argparse.Namespace(
            experiment="diffusion",
            output_dir="outputs",
            checkpoint_dir="outputs/checkpoints",
            log_dir="outputs/logs",
            checkpoint="model.pth",
        )
        result = args_to_dict(args)

        assert result["output"]["output_dir"] == "outputs"
        assert result["output"]["checkpoint_dir"] == "outputs/checkpoints"
        assert result["output"]["log_dir"] == "outputs/logs"
        assert result["output"]["checkpoint"] == "model.pth"

    def test_exclude_none_values(self):
        """Test that None values are excluded by default."""
        args = argparse.Namespace(
            experiment="classifier", epochs=100, batch_size=None, lr=None
        )
        result = args_to_dict(args, exclude_none=True)

        assert "experiment" in result
        assert "training" in result
        assert "epochs" in result["training"]
        # None values should be excluded
        assert "data" not in result
        assert "learning_rate" not in result.get("training", {})

    def test_include_none_values(self):
        """Test that None values can be included."""
        args = argparse.Namespace(experiment="classifier", epochs=100, batch_size=None)
        result = args_to_dict(args, exclude_none=False)

        assert "experiment" in result
        assert "epochs" in result["training"]
        assert "batch_size" in result["data"]
        assert result["data"]["batch_size"] is None

    def test_hyphen_to_underscore_conversion(self):
        """Test that hyphens in arg names are converted to underscores."""
        args = argparse.Namespace(
            experiment="gan",
            batch_size=32,  # Stored as batch_size internally
        )
        result = args_to_dict(args)

        assert result["data"]["batch_size"] == 32

    def test_config_path_excluded(self):
        """Test that config path is excluded from result."""
        args = argparse.Namespace(
            experiment="classifier", config="configs/test.json", epochs=10
        )
        result = args_to_dict(args)

        assert "config" not in result
        assert result["experiment"] == "classifier"


@pytest.mark.unit
class TestParseArgs:
    """Test the main parse_args function with config merging."""

    def test_parse_args_cli_only(self):
        """Test parsing with CLI arguments only."""
        config = parse_args(["--experiment", "classifier", "--epochs", "50"])

        assert config["experiment"] == "classifier"
        assert config["training"]["epochs"] == 50

    def test_parse_args_with_defaults(self):
        """Test parsing with default values."""
        defaults = {
            "training": {"epochs": 10, "batch_size": 32},
            "data": {"num_workers": 4},
        }

        config = parse_args(["--experiment", "gan"], defaults=defaults)

        assert config["experiment"] == "gan"
        assert config["training"]["epochs"] == 10
        assert config["training"]["batch_size"] == 32
        assert config["data"]["num_workers"] == 4

    def test_parse_args_cli_overrides_defaults(self):
        """Test that CLI arguments override defaults."""
        defaults = {"training": {"epochs": 10, "batch_size": 32}}

        config = parse_args(
            ["--experiment", "classifier", "--epochs", "100"], defaults=defaults
        )

        # CLI override
        assert config["training"]["epochs"] == 100
        # Default preserved
        assert config["training"]["batch_size"] == 32

    def test_parse_args_with_config_file(self, tmp_path):
        """Test parsing with config file."""
        # Create config file
        config_data = {
            "experiment": "classifier",
            "training": {"epochs": 20, "learning_rate": 0.001},
            "model": {"name": "resnet50"},
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = parse_args(
            ["--experiment", "classifier", "--config", str(config_file)]
        )

        assert config["training"]["epochs"] == 20
        assert config["training"]["learning_rate"] == 0.001
        assert config["model"]["name"] == "resnet50"

    def test_parse_args_priority_order(self, tmp_path):
        """Test priority: CLI > Config file > Defaults."""
        # Create config file
        config_data = {"training": {"epochs": 20, "learning_rate": 0.001}}
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        defaults = {"training": {"epochs": 10, "batch_size": 32, "optimizer": "sgd"}}

        config = parse_args(
            [
                "--experiment",
                "diffusion",
                "--config",
                str(config_file),
                "--epochs",
                "100",
            ],
            defaults=defaults,
        )

        # CLI has highest priority
        assert config["training"]["epochs"] == 100
        # Config file overrides defaults
        assert config["training"]["learning_rate"] == 0.001
        # Defaults fill in missing values
        assert config["training"]["batch_size"] == 32
        assert config["training"]["optimizer"] == "sgd"

    def test_parse_args_missing_config_file(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            parse_args(
                ["--experiment", "classifier", "--config", "nonexistent_config.json"]
            )

    def test_parse_args_multiple_arguments(self):
        """Test parsing multiple arguments together."""
        config = parse_args(
            [
                "--experiment",
                "classifier",
                "--model",
                "inceptionv3",
                "--epochs",
                "50",
                "--batch-size",
                "64",
                "--lr",
                "0.0002",
                "--optimizer",
                "adam",
                "--train-path",
                "data/train",
                "--val-path",
                "data/val",
                "--num-workers",
                "8",
            ]
        )

        assert config["experiment"] == "classifier"
        assert config["model"]["name"] == "inceptionv3"
        assert config["training"]["epochs"] == 50
        assert config["training"]["learning_rate"] == 0.0002
        assert config["training"]["optimizer"] == "adam"
        assert config["data"]["batch_size"] == 64
        assert config["data"]["train_path"] == "data/train"
        assert config["data"]["val_path"] == "data/val"
        assert config["data"]["num_workers"] == 8


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

    def test_validate_negative_epochs(self):
        """Test that negative epochs raises error."""
        config = {"experiment": "classifier", "training": {"epochs": -5}}

        with pytest.raises(ValueError, match="epochs must be positive"):
            validate_config(config)

    def test_validate_zero_epochs(self):
        """Test that zero epochs raises error."""
        config = {"experiment": "gan", "training": {"epochs": 0}}

        with pytest.raises(ValueError, match="epochs must be positive"):
            validate_config(config)

    def test_validate_negative_learning_rate(self):
        """Test that negative learning rate raises error."""
        config = {"experiment": "diffusion", "training": {"learning_rate": -0.01}}

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_config(config)

    def test_validate_negative_batch_size(self):
        """Test that negative batch size raises error."""
        config = {"experiment": "classifier", "data": {"batch_size": -32}}

        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_config(config)

    def test_validate_negative_num_workers(self):
        """Test that negative num_workers raises error."""
        config = {"experiment": "gan", "data": {"num_workers": -2}}

        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            validate_config(config)

    def test_validate_zero_num_workers_allowed(self):
        """Test that zero num_workers is allowed."""
        config = {"experiment": "classifier", "data": {"num_workers": 0}}
        # Should not raise
        validate_config(config)

    def test_validate_missing_optional_fields(self):
        """Test that missing optional fields don't cause errors."""
        config = {"experiment": "diffusion"}
        # Should not raise
        validate_config(config)

    def test_validate_none_values_in_numeric_fields(self):
        """Test that None values in numeric fields don't cause errors."""
        config = {
            "experiment": "classifier",
            "training": {"epochs": None, "learning_rate": None},
            "data": {"batch_size": None},
        }
        # Should not raise
        validate_config(config)


@pytest.mark.unit
class TestEndToEndCLIWorkflow:
    """Test complete CLI workflows."""

    def test_simple_training_workflow(self):
        """Test a simple training command workflow."""
        config = parse_args(
            [
                "--experiment",
                "classifier",
                "--model",
                "resnet50",
                "--epochs",
                "10",
                "--batch-size",
                "32",
            ]
        )

        validate_config(config)

        assert config["experiment"] == "classifier"
        assert config["model"]["name"] == "resnet50"
        assert config["training"]["epochs"] == 10
        assert config["data"]["batch_size"] == 32

    def test_generation_workflow(self):
        """Test a generation command workflow."""
        config = parse_args(
            ["--experiment", "diffusion", "--mode", "generate", "--num-samples", "1000"]
        )

        validate_config(config)

        assert config["experiment"] == "diffusion"
        assert config["mode"] == "generate"
        assert config["generation"]["num_samples"] == 1000

    def test_config_file_with_overrides_workflow(self, tmp_path):
        """Test workflow with config file and CLI overrides."""
        # Create config file
        config_data = {
            "experiment": "classifier",
            "model": {"name": "inceptionv3", "pretrained": True},
            "training": {"epochs": 20, "learning_rate": 0.001},
            "data": {"batch_size": 32, "num_workers": 4},
        }
        config_file = tmp_path / "experiment.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Parse with overrides
        config = parse_args(
            [
                "--experiment",
                "classifier",
                "--config",
                str(config_file),
                "--batch-size",
                "64",
                "--epochs",
                "50",
            ]
        )

        validate_config(config)

        # CLI overrides
        assert config["data"]["batch_size"] == 64
        assert config["training"]["epochs"] == 50
        # Config file values preserved
        assert config["model"]["name"] == "inceptionv3"
        assert config["model"]["pretrained"] is True
        assert config["training"]["learning_rate"] == 0.001
        assert config["data"]["num_workers"] == 4
