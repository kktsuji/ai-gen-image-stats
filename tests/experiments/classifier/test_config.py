"""Tests for Classifier Configuration

This module tests the classifier configuration management, including:
- Configuration validation
- Model-specific configuration overrides
"""

from pathlib import Path

import pytest
import yaml

from src.experiments.classifier.config import (
    get_model_specific_config,
    validate_config,
)

# Resolve project root so config file tests work regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ============================================================================
# Unit Tests - Fast, Pure Logic
# ============================================================================


@pytest.mark.unit
class TestGetModelSpecificConfig:
    """Test model-specific configuration overrides."""

    def test_resnet50_config(self):
        """Test ResNet50 specific configuration."""
        config = get_model_specific_config("resnet50")

        assert isinstance(config, dict)
        assert "data" in config
        assert config["data"]["image_size"] == 256
        assert config["data"]["crop_size"] == 224
        assert config["data"]["normalize"] == "imagenet"

    def test_resnet101_config(self):
        """Test ResNet101 specific configuration."""
        config = get_model_specific_config("resnet101")

        assert isinstance(config, dict)
        assert "data" in config
        assert config["data"]["image_size"] == 256
        assert config["data"]["crop_size"] == 224

    def test_resnet152_config(self):
        """Test ResNet152 specific configuration."""
        config = get_model_specific_config("resnet152")

        assert isinstance(config, dict)
        assert "data" in config
        assert config["data"]["image_size"] == 256
        assert config["data"]["crop_size"] == 224

    def test_inceptionv3_config(self):
        """Test InceptionV3 specific configuration."""
        config = get_model_specific_config("inceptionv3")

        assert isinstance(config, dict)
        assert "data" in config
        assert "model" in config

        # InceptionV3 requires 299x299 input
        assert config["data"]["image_size"] == 320
        assert config["data"]["crop_size"] == 299
        assert config["data"]["normalize"] == "imagenet"

        # InceptionV3 has dropout parameter
        assert "dropout" in config["model"]
        assert config["model"]["dropout"] == 0.5

    def test_invalid_model_name(self):
        """Test error with invalid model name."""
        with pytest.raises(ValueError, match="Unsupported model"):
            get_model_specific_config("invalid_model")

    def test_all_supported_models(self):
        """Test that all supported models have configurations."""
        supported_models = ["resnet50", "resnet101", "resnet152", "inceptionv3"]

        for model in supported_models:
            config = get_model_specific_config(model)
            assert isinstance(config, dict)


# ============================================================================
# Component Tests - With File I/O
# ============================================================================


def get_v2_default_config():
    """Helper function to get a valid V2 config for testing.

    Loads from configs/examples/classifier.yaml to stay in sync with the
    canonical example config (single source of truth).
    """
    import copy

    config_path = _PROJECT_ROOT / "configs/examples/classifier.yaml"
    with open(config_path) as f:
        return copy.deepcopy(yaml.safe_load(f))


def _make_evaluate_config() -> dict:
    """Create a valid evaluate-mode config (no filesystem dependency)."""
    config = get_v2_default_config()
    config["mode"] = "evaluate"
    config["evaluation"] = {
        "checkpoint": "path/to/checkpoint.pth",
        "bootstrap": {
            "enabled": False,
            "n_bootstrap": 1000,
            "confidence_level": 0.95,
            "save_predictions": False,
        },
    }
    config["output"]["subdirs"]["reports"] = "reports"
    return config


@pytest.mark.unit
class TestValidateConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = get_v2_default_config()
        # Should not raise
        validate_config(config)

    def test_missing_mode_key(self):
        """Test validation fails with missing mode key."""
        config = get_v2_default_config()
        del config["mode"]

        with pytest.raises(KeyError, match="Missing required config key: mode"):
            validate_config(config)

    def test_missing_compute_key(self):
        """Test validation fails with missing compute key."""
        config = get_v2_default_config()
        del config["compute"]

        with pytest.raises(KeyError, match="Missing required config key: compute"):
            validate_config(config)

    def test_invalid_mode(self):
        """Test validation fails with invalid mode."""
        config = get_v2_default_config()
        config["mode"] = "invalid"

        with pytest.raises(ValueError, match="Invalid mode"):
            validate_config(config)

    def test_valid_modes(self):
        """Test validation succeeds with all valid modes."""
        # Train mode
        config = get_v2_default_config()
        config["mode"] = "train"
        validate_config(config)

        # Evaluate mode
        eval_config = _make_evaluate_config()
        validate_config(eval_config)

    def test_invalid_device(self):
        """Test validation fails with invalid device."""
        config = get_v2_default_config()
        config["compute"]["device"] = "tpu"

        with pytest.raises(ValueError, match="Invalid device"):
            validate_config(config)

    def test_valid_devices(self):
        """Test validation succeeds with all valid devices."""
        config = get_v2_default_config()

        for device in ["cuda", "cpu", "auto"]:
            config["compute"]["device"] = device
            validate_config(config)

    def test_missing_architecture_section(self):
        """Test validation fails with missing architecture section."""
        config = get_v2_default_config()
        del config["model"]["architecture"]

        with pytest.raises(
            KeyError, match="Missing required field: model.architecture"
        ):
            validate_config(config)

    def test_invalid_model_name(self):
        """Test validation fails with invalid model name."""
        config = get_v2_default_config()
        config["model"]["architecture"]["name"] = "invalid_model"

        with pytest.raises(ValueError, match="Invalid model name"):
            validate_config(config)

    def test_missing_data_sections(self):
        """Test validation fails with missing data sections."""
        for section in ["split_file", "loading", "preprocessing", "augmentation"]:
            config_copy = get_v2_default_config()
            del config_copy["data"][section]

            with pytest.raises(
                KeyError, match=f"Missing required field: data.{section}"
            ):
                validate_config(config_copy)

    def test_missing_output_subdirs(self):
        """Test validation fails with missing output subdirs."""
        config = get_v2_default_config()
        del config["output"]["subdirs"]

        with pytest.raises(
            KeyError, match="Missing required config key: output.subdirs"
        ):
            validate_config(config)

    def test_missing_required_subdirs(self):
        """Test validation fails with missing required subdirs."""
        for subdir in ["logs", "checkpoints"]:
            config_copy = get_v2_default_config()
            del config_copy["output"]["subdirs"][subdir]

            with pytest.raises(
                ValueError, match=f"output.subdirs.{subdir} is required"
            ):
                validate_config(config_copy)

    def test_invalid_optimizer_type(self):
        """Test validation fails with invalid optimizer type."""
        config = get_v2_default_config()
        config["training"]["optimizer"]["type"] = "invalid"

        with pytest.raises(ValueError, match="Invalid optimizer"):
            validate_config(config)

    def test_invalid_scheduler_type(self):
        """Test validation fails with invalid scheduler type."""
        config = get_v2_default_config()
        config["training"]["scheduler"]["type"] = "invalid"

        with pytest.raises(ValueError, match="Invalid scheduler"):
            validate_config(config)

    def test_evaluate_mode_requires_checkpoint(self):
        """Test validation fails in evaluate mode without checkpoint."""
        config = _make_evaluate_config()
        config["evaluation"]["checkpoint"] = None  # Invalid: should be a path

        with pytest.raises(
            ValueError, match="evaluation.checkpoint is required for evaluate mode"
        ):
            validate_config(config)

    def test_evaluate_mode_requires_reports_subdir(self):
        """Test validation fails in evaluate mode without reports subdir."""
        config = _make_evaluate_config()
        config["output"]["subdirs"].pop("reports", None)

        with pytest.raises(
            ValueError, match="output.subdirs.reports is required for evaluate mode"
        ):
            validate_config(config)

    def test_save_latest_valid_true(self):
        """save_latest: true is accepted."""
        config = get_v2_default_config()
        config["training"]["checkpointing"]["save_latest"] = True
        validate_config(config)  # should not raise

    def test_save_latest_valid_false(self):
        """save_latest: false is accepted."""
        config = get_v2_default_config()
        config["training"]["checkpointing"]["save_latest"] = False
        validate_config(config)  # should not raise

    def test_save_latest_missing_is_optional(self):
        """save_latest absent from config is accepted (optional field)."""
        config = get_v2_default_config()
        config["training"]["checkpointing"].pop("save_latest", None)
        validate_config(config)  # should not raise

    def test_save_latest_invalid_type(self):
        """save_latest with non-bool raises ValueError."""
        config = get_v2_default_config()
        config["training"]["checkpointing"]["save_latest"] = "yes"
        with pytest.raises(ValueError, match="save_latest must be a boolean"):
            validate_config(config)


@pytest.mark.unit
class TestValidateConfigErrorPaths:
    """Test untested validation error paths in validate_config."""

    def test_invalid_experiment_type(self):
        """experiment != 'classifier' raises ValueError."""
        config = get_v2_default_config()
        config["experiment"] = "diffusion"
        with pytest.raises(ValueError, match="Invalid experiment type"):
            validate_config(config)

    def test_architecture_field_is_none(self):
        """architecture field set to None raises ValueError."""
        config = get_v2_default_config()
        config["model"]["architecture"]["name"] = None
        with pytest.raises(ValueError, match="model.architecture.name cannot be None"):
            validate_config(config)

    def test_num_classes_not_positive(self):
        """num_classes = 0 or -1 raises ValueError."""
        config = get_v2_default_config()
        config["model"]["architecture"]["num_classes"] = 0
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            validate_config(config)

        config = get_v2_default_config()
        config["model"]["architecture"]["num_classes"] = -1
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            validate_config(config)

    def test_pretrained_not_bool(self):
        """pretrained = 'yes' raises ValueError."""
        config = get_v2_default_config()
        config["model"]["initialization"]["pretrained"] = "yes"
        with pytest.raises(
            ValueError, match="initialization.pretrained must be a boolean"
        ):
            validate_config(config)

    def test_freeze_backbone_not_bool(self):
        """freeze_backbone = 'yes' raises ValueError."""
        config = get_v2_default_config()
        config["model"]["initialization"]["freeze_backbone"] = "yes"
        with pytest.raises(
            ValueError, match="initialization.freeze_backbone must be a boolean"
        ):
            validate_config(config)

    def test_split_file_empty_string(self):
        """split_file = '' raises ValueError."""
        config = get_v2_default_config()
        config["data"]["split_file"] = ""
        with pytest.raises(
            ValueError, match="data.split_file must be a non-empty string"
        ):
            validate_config(config)

    def test_image_size_not_positive(self):
        """image_size = 0 raises ValueError."""
        config = get_v2_default_config()
        config["data"]["preprocessing"]["image_size"] = 0
        with pytest.raises(ValueError, match="image_size must be a positive integer"):
            validate_config(config)

    def test_crop_size_not_positive(self):
        """crop_size = 0 raises ValueError."""
        config = get_v2_default_config()
        config["data"]["preprocessing"]["crop_size"] = 0
        with pytest.raises(ValueError, match="crop_size must be a positive integer"):
            validate_config(config)

    def test_invalid_normalize_option(self):
        """normalize = 'invalid' raises ValueError."""
        config = get_v2_default_config()
        config["data"]["preprocessing"]["normalize"] = "invalid"
        with pytest.raises(ValueError, match="Invalid normalize option"):
            validate_config(config)

    def test_train_mode_missing_training(self):
        """train mode without training section raises KeyError."""
        config = get_v2_default_config()
        config["mode"] = "train"
        del config["training"]
        with pytest.raises(
            KeyError,
            match="Missing required section: training",
        ):
            validate_config(config)

    def test_train_mode_invalid_epochs(self):
        """epochs = 0 raises ValueError."""
        config = get_v2_default_config()
        config["training"]["epochs"] = 0
        with pytest.raises(ValueError, match="epochs must be a positive integer"):
            validate_config(config)

    def test_train_mode_missing_optimizer_type(self):
        """missing optimizer.type raises KeyError."""
        config = get_v2_default_config()
        del config["training"]["optimizer"]["type"]
        with pytest.raises(
            KeyError, match="Missing required field: training.optimizer.type"
        ):
            validate_config(config)

    def test_train_mode_missing_learning_rate(self):
        """missing optimizer.learning_rate raises KeyError."""
        config = get_v2_default_config()
        del config["training"]["optimizer"]["learning_rate"]
        with pytest.raises(
            KeyError, match="Missing required field: training.optimizer.learning_rate"
        ):
            validate_config(config)

    def test_evaluate_mode_missing_evaluation(self):
        """evaluate mode without evaluation key raises KeyError."""
        config = get_v2_default_config()
        config["mode"] = "evaluate"
        config.pop("evaluation", None)  # Remove evaluation section
        with pytest.raises(
            KeyError,
            match="Missing required section: evaluation",
        ):
            validate_config(config)


@pytest.mark.unit
class TestValidateSyntheticAugmentation:
    """Test synthetic_augmentation config validation."""

    def test_missing_section_passes(self):
        """synthetic_augmentation absent from config is accepted (optional)."""
        config = get_v2_default_config()
        config["data"].pop("synthetic_augmentation", None)
        validate_config(config)  # should not raise

    def test_disabled_passes(self):
        """synthetic_augmentation with enabled: false passes validation."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": False,
            "split_file": None,
            "limit": {"mode": None, "max_ratio": None, "max_samples": None},
        }
        validate_config(config)  # should not raise

    def test_enabled_with_valid_split_file(self):
        """enabled: true with valid split_file passes."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": None, "max_ratio": None, "max_samples": None},
        }
        validate_config(config)  # should not raise

    def test_enabled_missing_split_file(self):
        """enabled: true with missing split_file raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": None,
            "limit": {"mode": None},
        }
        with pytest.raises(ValueError, match="split_file must be a non-empty string"):
            validate_config(config)

    def test_enabled_empty_split_file(self):
        """enabled: true with empty string split_file raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "",
            "limit": {"mode": None},
        }
        with pytest.raises(ValueError, match="split_file must be a non-empty string"):
            validate_config(config)

    def test_invalid_limit_mode(self):
        """Invalid limit.mode raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "invalid"},
        }
        with pytest.raises(
            ValueError, match="Invalid synthetic_augmentation limit.mode"
        ):
            validate_config(config)

    def test_max_ratio_valid(self):
        """limit.mode = max_ratio with positive float passes."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_ratio", "max_ratio": 0.5, "max_samples": None},
        }
        validate_config(config)  # should not raise

    def test_max_ratio_missing_key(self):
        """limit.mode = max_ratio without max_ratio key raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_ratio"},
        }
        with pytest.raises(ValueError, match="max_ratio is required"):
            validate_config(config)

    def test_max_ratio_not_positive(self):
        """limit.mode = max_ratio with non-positive value raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_ratio", "max_ratio": 0},
        }
        with pytest.raises(ValueError, match="max_ratio must be a positive number"):
            validate_config(config)

    def test_max_ratio_none(self):
        """limit.mode = max_ratio with None raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_ratio", "max_ratio": None},
        }
        with pytest.raises(ValueError, match="max_ratio must be a positive number"):
            validate_config(config)

    def test_max_samples_valid(self):
        """limit.mode = max_samples with positive int passes."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_samples", "max_ratio": None, "max_samples": 100},
        }
        validate_config(config)  # should not raise

    def test_max_samples_missing_key(self):
        """limit.mode = max_samples without max_samples key raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_samples"},
        }
        with pytest.raises(ValueError, match="max_samples is required"):
            validate_config(config)

    def test_max_samples_not_positive(self):
        """limit.mode = max_samples with non-positive value raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_samples", "max_samples": 0},
        }
        with pytest.raises(ValueError, match="max_samples must be a positive integer"):
            validate_config(config)

    def test_max_samples_not_int(self):
        """limit.mode = max_samples with float raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_samples", "max_samples": 10.5},
        }
        with pytest.raises(ValueError, match="max_samples must be a positive integer"):
            validate_config(config)

    def test_max_ratio_bool_rejected(self):
        """limit.mode = max_ratio with bool value raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_ratio", "max_ratio": True},
        }
        with pytest.raises(ValueError, match="max_ratio must be a positive number"):
            validate_config(config)

    def test_max_samples_bool_rejected(self):
        """limit.mode = max_samples with bool value raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": "max_samples", "max_samples": True},
        }
        with pytest.raises(ValueError, match="max_samples must be a positive integer"):
            validate_config(config)

    def test_balancing_and_augmentation_mutual_exclusion(self):
        """enabled augmentation + enabled balancing raises ValueError."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": None, "max_ratio": None, "max_samples": None},
        }
        config["data"]["balancing"] = {
            "weighted_sampler": {"enabled": True, "method": "inverse"},
            "downsampling": {"enabled": False},
            "upsampling": {"enabled": False},
        }
        with pytest.raises(ValueError, match="Cannot use both data.balancing and"):
            validate_config(config)

    def test_balancing_disabled_with_augmentation_passes(self):
        """enabled augmentation + all balancing disabled passes."""
        config = get_v2_default_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": "outputs/selected.json",
            "limit": {"mode": None, "max_ratio": None, "max_samples": None},
        }
        config["data"]["balancing"] = {
            "weighted_sampler": {"enabled": False},
            "downsampling": {"enabled": False},
            "upsampling": {"enabled": False},
        }
        validate_config(config)  # should not raise

    def test_not_validated_in_evaluate_mode(self):
        """synthetic_augmentation is not validated in evaluate mode."""
        config = _make_evaluate_config()
        config["data"]["synthetic_augmentation"] = {
            "enabled": True,
            "split_file": None,  # would fail in train mode
            "limit": {"mode": None},
        }
        validate_config(config)  # should not raise

    def test_evaluate_mode_not_broken_by_augmentation_section(self):
        """evaluate mode validation still works with synthetic_augmentation present."""
        config = get_v2_default_config()
        config["mode"] = "evaluate"
        config.pop("evaluation", None)  # Remove evaluation section
        # Don't add evaluation section — should raise KeyError
        with pytest.raises(KeyError, match="Missing required section: evaluation"):
            validate_config(config)


@pytest.mark.unit
class TestValidateBootstrapConfig:
    """Test evaluation.bootstrap config validation."""

    def test_evaluate_mode_with_valid_bootstrap(self):
        """Valid bootstrap config passes validation."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": True,
            "n_bootstrap": 10000,
            "confidence_level": 0.95,
            "save_predictions": True,
        }
        validate_config(config)  # should not raise

    def test_evaluate_mode_with_disabled_bootstrap(self):
        """Disabled bootstrap config with all fields passes validation."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": False,
            "n_bootstrap": 1000,
            "confidence_level": 0.95,
            "save_predictions": False,
        }
        validate_config(config)  # should not raise

    def test_evaluate_mode_without_bootstrap_section(self):
        """Missing bootstrap section raises KeyError in evaluate mode."""
        config = _make_evaluate_config()
        config["evaluation"].pop("bootstrap", None)
        with pytest.raises(
            KeyError, match="Missing required field: evaluation.bootstrap"
        ):
            validate_config(config)

    def test_bootstrap_missing_enabled(self):
        """bootstrap without enabled key raises KeyError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "n_bootstrap": 10000,
            "confidence_level": 0.95,
            "save_predictions": True,
        }
        with pytest.raises(
            KeyError, match="Missing required field: evaluation.bootstrap.enabled"
        ):
            validate_config(config)

    def test_bootstrap_null_raises(self):
        """bootstrap: null raises ValueError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = None
        with pytest.raises(ValueError, match="evaluation.bootstrap must be a mapping"):
            validate_config(config)

    def test_bootstrap_list_raises(self):
        """bootstrap as a list raises ValueError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = ["enabled", True]
        with pytest.raises(ValueError, match="evaluation.bootstrap must be a mapping"):
            validate_config(config)

    def test_bootstrap_empty_dict_raises(self):
        """Empty bootstrap dict raises KeyError for missing fields."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {}
        with pytest.raises(
            KeyError, match="Missing required field: evaluation.bootstrap.enabled"
        ):
            validate_config(config)

    def test_bootstrap_enabled_not_bool(self):
        """bootstrap.enabled = 'yes' raises ValueError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": "yes",
            "n_bootstrap": 1000,
            "confidence_level": 0.95,
            "save_predictions": False,
        }
        with pytest.raises(
            ValueError, match="evaluation.bootstrap.enabled must be a boolean"
        ):
            validate_config(config)

    def test_bootstrap_missing_n_bootstrap(self):
        """bootstrap without n_bootstrap raises KeyError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": True,
            "confidence_level": 0.95,
            "save_predictions": True,
        }
        with pytest.raises(
            KeyError, match="Missing required field: evaluation.bootstrap.n_bootstrap"
        ):
            validate_config(config)

    def test_bootstrap_disabled_missing_field_raises(self):
        """Disabled bootstrap still requires all fields."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": False,
            "n_bootstrap": 1000,
            "confidence_level": 0.95,
            # save_predictions missing
        }
        with pytest.raises(
            KeyError,
            match="Missing required field: evaluation.bootstrap.save_predictions",
        ):
            validate_config(config)

    def test_bootstrap_n_bootstrap_not_positive(self):
        """n_bootstrap = 0 raises ValueError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": True,
            "n_bootstrap": 0,
            "confidence_level": 0.95,
            "save_predictions": True,
        }
        with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
            validate_config(config)

    def test_bootstrap_n_bootstrap_bool_rejected(self):
        """n_bootstrap = True (bool) raises ValueError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": True,
            "n_bootstrap": True,
            "confidence_level": 0.95,
            "save_predictions": True,
        }
        with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
            validate_config(config)

    def test_bootstrap_missing_confidence_level(self):
        """enabled bootstrap without confidence_level raises KeyError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": True,
            "n_bootstrap": 1000,
            "save_predictions": True,
        }
        with pytest.raises(
            KeyError,
            match="Missing required field: evaluation.bootstrap.confidence_level",
        ):
            validate_config(config)

    def test_bootstrap_confidence_level_out_of_range(self):
        """confidence_level = 0 or 1 raises ValueError."""
        for invalid in [0, 1, -0.1, 1.5]:
            config = _make_evaluate_config()
            config["evaluation"]["bootstrap"] = {
                "enabled": True,
                "n_bootstrap": 1000,
                "confidence_level": invalid,
                "save_predictions": True,
            }
            with pytest.raises(
                ValueError, match="confidence_level must be a number in"
            ):
                validate_config(config)

    def test_bootstrap_confidence_level_bool_rejected(self):
        """confidence_level = True (bool) raises ValueError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": True,
            "n_bootstrap": 1000,
            "confidence_level": True,
            "save_predictions": True,
        }
        with pytest.raises(ValueError, match="confidence_level must be a number in"):
            validate_config(config)

    def test_bootstrap_missing_save_predictions(self):
        """enabled bootstrap without save_predictions raises KeyError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": True,
            "n_bootstrap": 1000,
            "confidence_level": 0.95,
        }
        with pytest.raises(
            KeyError,
            match="Missing required field: evaluation.bootstrap.save_predictions",
        ):
            validate_config(config)

    def test_bootstrap_save_predictions_not_bool(self):
        """save_predictions = 'yes' raises ValueError."""
        config = _make_evaluate_config()
        config["evaluation"]["bootstrap"] = {
            "enabled": True,
            "n_bootstrap": 1000,
            "confidence_level": 0.95,
            "save_predictions": "yes",
        }
        with pytest.raises(
            ValueError,
            match="evaluation.bootstrap.save_predictions must be a boolean",
        ):
            validate_config(config)


@pytest.mark.component
class TestConfigFiles:
    """Test actual config files (configs/examples/classifier.yaml)."""

    def test_example_config_file(self):
        """Test that configs/examples/classifier.yaml is valid."""
        config_path = _PROJECT_ROOT / "configs/examples/classifier.yaml"

        if not config_path.exists():
            pytest.fail("Required file missing: configs/examples/classifier.yaml")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should validate
        validate_config(config)
