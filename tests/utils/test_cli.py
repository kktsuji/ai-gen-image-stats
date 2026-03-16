"""Unit tests for CLI utilities.

Tests cover:
- Argument parser creation and configuration
- CLI argument parsing with config file
- Config validation
- Error handling for missing/invalid configs
- Dot-notation config overrides (type inference, parsing, merging)
"""

import pytest
import yaml

from src.utils.cli import (
    create_parser,
    dot_notation_to_dict,
    infer_type,
    parse_args,
    parse_override_args,
    validate_config,
    validate_override_keys,
)


@pytest.mark.unit
class TestCreateParser:
    """Test argument parser creation."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.description is not None

    def test_parser_has_positional_config_argument(self):
        """Test that parser has positional config_path argument."""
        parser = create_parser()
        # Parse with config path (use parse_known_args to allow unknown args)
        args, _ = parser.parse_known_args(["configs/test.yaml"])
        assert args.config_path == "configs/test.yaml"

    def test_config_path_is_required(self):
        """Test that config_path argument is required."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_known_args([])  # No arguments should fail

    def test_parser_accepts_verbose_flag(self):
        """Test that parser accepts optional --verbose flag."""
        parser = create_parser()
        args, _ = parser.parse_known_args(["configs/test.yaml", "--verbose"])
        assert args.config_path == "configs/test.yaml"
        assert args.verbose is True

    def test_parser_passes_dot_notation_as_remaining(self):
        """Test that dot-notation args are returned as remaining."""
        parser = create_parser()
        args, remaining = parser.parse_known_args(
            ["configs/test.yaml", "--model.image_size", "64"]
        )
        assert args.config_path == "configs/test.yaml"
        assert remaining == ["--model.image_size", "64"]


@pytest.mark.unit
class TestInferType:
    """Test type inference for CLI string values."""

    def test_infer_type_integer(self):
        assert infer_type("42") == 42
        assert isinstance(infer_type("42"), int)

    def test_infer_type_negative_integer(self):
        assert infer_type("-5") == -5
        assert isinstance(infer_type("-5"), int)

    def test_infer_type_zero(self):
        assert infer_type("0") == 0
        assert isinstance(infer_type("0"), int)

    def test_infer_type_float(self):
        assert infer_type("3.14") == 3.14
        assert isinstance(infer_type("3.14"), float)

    def test_infer_type_negative_float(self):
        assert infer_type("-0.001") == -0.001
        assert isinstance(infer_type("-0.001"), float)

    def test_infer_type_scientific_notation(self):
        assert infer_type("1e-4") == 0.0001
        assert isinstance(infer_type("1e-4"), float)

    def test_infer_type_bool_true(self):
        assert infer_type("true") is True
        assert infer_type("True") is True
        assert infer_type("TRUE") is True

    def test_infer_type_bool_false(self):
        assert infer_type("false") is False
        assert infer_type("False") is False
        assert infer_type("FALSE") is False

    def test_infer_type_none(self):
        assert infer_type("null") is None
        assert infer_type("none") is None
        assert infer_type("None") is None

    def test_infer_type_string(self):
        assert infer_type("hello") == "hello"
        assert infer_type("some/path") == "some/path"

    def test_infer_type_string_with_dots(self):
        """IP-like strings should remain as strings."""
        assert infer_type("192.168.1.1") == "192.168.1.1"
        assert isinstance(infer_type("192.168.1.1"), str)

    def test_infer_type_quoted_single_forces_string(self):
        """Single-quoted values are always treated as strings."""
        assert infer_type("'42'") == "42"
        assert isinstance(infer_type("'42'"), str)
        assert infer_type("'true'") == "true"
        assert infer_type("'none'") == "none"
        assert infer_type("'3.14'") == "3.14"

    def test_infer_type_quoted_double_forces_string(self):
        """Double-quoted values are always treated as strings."""
        assert infer_type('"42"') == "42"
        assert isinstance(infer_type('"42"'), str)
        assert infer_type('"false"') == "false"
        assert infer_type('"null"') == "null"

    def test_infer_type_quoted_empty_string(self):
        """Quoted empty string returns empty string."""
        assert infer_type("''") == ""
        assert infer_type('""') == ""

    def test_infer_type_mismatched_quotes_not_stripped(self):
        """Mismatched quotes are not treated as quoting."""
        assert infer_type("'42\"") == "'42\""
        assert infer_type("\"42'") == "\"42'"

    def test_infer_type_single_quote_not_stripped(self):
        """A single quote character is not treated as quoting."""
        assert infer_type("'") == "'"

    def test_infer_type_empty_string(self):
        assert infer_type("") == ""
        assert isinstance(infer_type(""), str)


@pytest.mark.unit
class TestDotNotationToDict:
    """Test dot-notation key to nested dict conversion."""

    def test_single_level(self):
        assert dot_notation_to_dict("key", 1) == {"key": 1}

    def test_two_levels(self):
        assert dot_notation_to_dict("a.b", 1) == {"a": {"b": 1}}

    def test_three_levels(self):
        assert dot_notation_to_dict("a.b.c", 1) == {"a": {"b": {"c": 1}}}

    def test_preserves_value_type(self):
        assert dot_notation_to_dict("a.b", "hello") == {"a": {"b": "hello"}}
        assert dot_notation_to_dict("a.b", None) == {"a": {"b": None}}
        assert dot_notation_to_dict("a.b", True) == {"a": {"b": True}}
        assert dot_notation_to_dict("a.b", 3.14) == {"a": {"b": 3.14}}


@pytest.mark.unit
class TestParseOverrideArgs:
    """Test parsing of dot-notation CLI override arguments."""

    def test_single_override(self):
        result = parse_override_args(["--model.image_size", "64"])
        assert result == {"model": {"image_size": 64}}

    def test_multiple_overrides(self):
        result = parse_override_args(
            ["--model.image_size", "64", "--training.epochs", "50"]
        )
        assert result == {"model": {"image_size": 64}, "training": {"epochs": 50}}

    def test_nested_override(self):
        result = parse_override_args(["--model.architecture.image_size", "60"])
        assert result == {"model": {"architecture": {"image_size": 60}}}

    def test_type_inference_in_overrides(self):
        result = parse_override_args(
            [
                "--a.int_val",
                "42",
                "--a.float_val",
                "3.14",
                "--a.bool_val",
                "true",
                "--a.none_val",
                "none",
                "--a.str_val",
                "hello",
            ]
        )
        assert result["a"]["int_val"] == 42
        assert result["a"]["float_val"] == 3.14
        assert result["a"]["bool_val"] is True
        assert result["a"]["none_val"] is None
        assert result["a"]["str_val"] == "hello"

    def test_empty_remaining_args(self):
        assert parse_override_args([]) == {}

    def test_missing_value_raises_error(self):
        with pytest.raises(ValueError, match="Missing value"):
            parse_override_args(["--model.image_size"])

    def test_missing_value_followed_by_key_raises_error(self):
        with pytest.raises(ValueError, match="Missing value"):
            parse_override_args(["--model.image_size", "--training.epochs", "10"])

    def test_non_dot_notation_raises_error(self):
        with pytest.raises(ValueError, match="dot-notation"):
            parse_override_args(["--epochs", "10"])

    def test_no_leading_dashes_raises_error(self):
        with pytest.raises(ValueError, match="Unexpected argument"):
            parse_override_args(["model.image_size", "64"])

    def test_overlapping_keys_deep_merged(self):
        result = parse_override_args(["--model.a", "1", "--model.b", "2"])
        assert result == {"model": {"a": 1, "b": 2}}


@pytest.mark.unit
class TestValidateOverrideKeys:
    """Test validation of override keys against base config."""

    def test_valid_key_passes(self):
        config = {"model": {"architecture": {"image_size": 32}}}
        overrides = {"model": {"architecture": {"image_size": 64}}}
        validate_override_keys(config, overrides)  # Should not raise

    def test_unknown_top_level_key_raises(self):
        config = {"model": {"name": "resnet"}}
        overrides = {"modle": {"name": "vgg"}}
        with pytest.raises(ValueError, match="Unknown config key: 'modle'"):
            validate_override_keys(config, overrides)

    def test_unknown_nested_key_raises(self):
        config = {"model": {"architecture": {"image_size": 32}}}
        overrides = {"model": {"architectur": {"image_size": 64}}}
        with pytest.raises(ValueError, match="Unknown config key: 'model.architectur'"):
            validate_override_keys(config, overrides)

    def test_unknown_leaf_key_raises(self):
        config = {"model": {"architecture": {"image_size": 32}}}
        overrides = {"model": {"architecture": {"imag_size": 64}}}
        with pytest.raises(
            ValueError, match="Unknown config key: 'model.architecture.imag_size'"
        ):
            validate_override_keys(config, overrides)

    def test_error_message_shows_available_keys(self):
        config = {"model": {"architecture": {"image_size": 32, "channels": 3}}}
        overrides = {"model": {"architecture": {"imag_size": 64}}}
        with pytest.raises(ValueError, match="Available keys.*channels.*image_size"):
            validate_override_keys(config, overrides)

    def test_multiple_valid_keys_pass(self):
        config = {"model": {"image_size": 32}, "training": {"epochs": 10}}
        overrides = {"model": {"image_size": 64}, "training": {"epochs": 50}}
        validate_override_keys(config, overrides)  # Should not raise

    def test_override_non_dict_with_dict_skips_recursion(self):
        """Override a scalar with a dict — validation only checks key existence."""
        config = {"model": {"image_size": 32}}
        overrides = {"model": {"image_size": {"width": 64}}}
        # Key exists, so validation passes (type mismatch is a downstream concern)
        validate_override_keys(config, overrides)

    def test_override_dict_with_scalar_skips_recursion(self):
        """Override a dict with a scalar — validation only checks key existence."""
        config = {"model": {"architecture": {"image_size": 32}}}
        overrides = {"model": {"architecture": "simple"}}
        validate_override_keys(config, overrides)  # Should not raise


@pytest.mark.unit
class TestParseArgs:
    """Test the main parse_args function."""

    def test_parse_args_with_config_file(self, tmp_path):
        """Test parsing with a valid config file."""
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
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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
            parse_args(["nonexistent_config.yaml"])

    def test_parse_args_missing_experiment_field(self, tmp_path):
        """Test that config without experiment field raises error."""
        config_data = {"training": {"epochs": 10}}
        config_file = tmp_path / "no_experiment.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with pytest.raises(ValueError, match="Missing required 'experiment' field"):
            parse_args([str(config_file)])

    def test_parse_args_invalid_experiment_type(self, tmp_path):
        """Test that invalid experiment type raises error."""
        config_data = {"experiment": "invalid_type"}
        config_file = tmp_path / "invalid_exp.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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
        config_file = tmp_path / "diffusion_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        config = parse_args([str(config_file)])
        assert config["experiment"] == "diffusion"

    def test_parse_args_with_verbose_flag(self, tmp_path):
        """Test that --verbose flag is added to config."""
        config_data = {"experiment": "classifier", "model": {"name": "resnet50"}}
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        config = parse_args([str(config_file), "--verbose"])
        assert config["verbose"] is True


@pytest.mark.unit
class TestParseArgsWithOverrides:
    """Test parse_args with dot-notation CLI overrides."""

    def _write_config(self, tmp_path, data):
        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        return str(config_file)

    def test_override_scalar_value(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "classifier",
                "model": {"architecture": {"image_size": 32}},
            },
        )
        config = parse_args([path, "--model.architecture.image_size", "60"])
        assert config["model"]["architecture"]["image_size"] == 60

    def test_override_nested_value(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "diffusion",
                "training": {"optimizer": {"learning_rate": 0.001}},
            },
        )
        config = parse_args([path, "--training.optimizer.learning_rate", "1e-4"])
        assert config["training"]["optimizer"]["learning_rate"] == 0.0001

    def test_override_preserves_unrelated_config(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "classifier",
                "model": {"name": "resnet50", "architecture": {"image_size": 32}},
                "data": {"batch_size": 16},
            },
        )
        config = parse_args([path, "--model.architecture.image_size", "64"])
        assert config["model"]["architecture"]["image_size"] == 64
        assert config["model"]["name"] == "resnet50"
        assert config["data"]["batch_size"] == 16

    def test_override_unknown_key_raises_error(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "classifier",
                "model": {"architecture": {}},
            },
        )
        with pytest.raises(ValueError, match="Unknown config key"):
            parse_args([path, "--model.architecture.new_param", "42"])

    def test_override_typo_key_raises_error(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "classifier",
                "model": {"architecture": {"image_size": 32}},
            },
        )
        with pytest.raises(ValueError, match="Unknown config key"):
            parse_args([path, "--model.architectur.image_size", "60"])

    def test_override_with_verbose(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "classifier",
                "model": {"architecture": {"image_size": 32}},
            },
        )
        config = parse_args(
            [path, "--verbose", "--model.architecture.image_size", "64"]
        )
        assert config["verbose"] is True
        assert config["model"]["architecture"]["image_size"] == 64

    def test_override_type_bool(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "diffusion",
                "training": {"performance": {"use_amp": False}},
            },
        )
        config = parse_args([path, "--training.performance.use_amp", "true"])
        assert config["training"]["performance"]["use_amp"] is True

    def test_override_type_none(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "diffusion",
                "compute": {"seed": 42},
            },
        )
        config = parse_args([path, "--compute.seed", "none"])
        assert config["compute"]["seed"] is None

    def test_override_type_string(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "classifier",
                "data": {"split_file": "old.json"},
            },
        )
        config = parse_args([path, "--data.split_file", "new.json"])
        assert config["data"]["split_file"] == "new.json"

    def test_multiple_overrides(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "diffusion",
                "model": {"architecture": {"image_size": 32}},
                "training": {"epochs": 10},
            },
        )
        config = parse_args(
            [
                path,
                "--model.architecture.image_size",
                "64",
                "--training.epochs",
                "50",
            ]
        )
        assert config["model"]["architecture"]["image_size"] == 64
        assert config["training"]["epochs"] == 50

    def test_override_quoted_value_forces_string(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "experiment": "diffusion",
                "data": {"label": "old"},
            },
        )
        config = parse_args([path, "--data.label", "'0'"])
        assert config["data"]["label"] == "0"
        assert isinstance(config["data"]["label"], str)

    def test_invalid_override_no_dot_raises_error(self, tmp_path):
        path = self._write_config(tmp_path, {"experiment": "classifier"})
        with pytest.raises(ValueError, match="dot-notation"):
            parse_args([path, "--epochs", "10"])


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
    """Test that config file is still required as the base."""

    def test_requires_config_file(self):
        """Test that config file is mandatory."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_loads_config_only(self, tmp_path):
        """Test that all settings come from config file when no overrides."""
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
        config_file = tmp_path / "full_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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
            config_file = tmp_path / f"{exp_type}_config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)

            config = parse_args([str(config_file)])
            assert config["experiment"] == exp_type


@pytest.mark.unit
class TestEndToEndCLIWorkflow:
    """Test complete CLI workflows."""

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
        config_file = tmp_path / "classifier.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

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
        config_file = tmp_path / "diffusion.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        config = parse_args([str(config_file)])
        validate_config(config)

        assert config["experiment"] == "diffusion"
        assert config["model"]["image_size"] == 64
        assert config["training"]["epochs"] == 100

    def test_workflow_with_overrides(self, tmp_path):
        """Test workflow with config file plus CLI overrides."""
        config_data = {
            "experiment": "diffusion",
            "model": {"architecture": {"image_size": 32}},
            "training": {"epochs": 100},
        }
        config_file = tmp_path / "diffusion.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        config = parse_args(
            [
                str(config_file),
                "--model.architecture.image_size",
                "64",
                "--training.epochs",
                "200",
            ]
        )
        validate_config(config)

        assert config["experiment"] == "diffusion"
        assert config["model"]["architecture"]["image_size"] == 64
        assert config["training"]["epochs"] == 200

    def test_config_file_required_no_defaults(self):
        """Test that there are no defaults - config file is mandatory."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_error_on_missing_config_fields(self, tmp_path):
        """Test that missing required fields in config cause errors during experiment-specific validation."""
        config_data = {
            "experiment": "classifier",
        }
        config_file = tmp_path / "incomplete.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        config = parse_args([str(config_file)])
        validate_config(config)

        assert config["experiment"] == "classifier"
        assert "model" not in config
