"""Tests for Classifier Configuration

This module tests the classifier configuration management, including:
- Default configuration generation
- Configuration validation
- Model-specific configuration overrides
"""

from pathlib import Path

import pytest
import yaml

from src.experiments.classifier.config import (
    get_default_config,
    get_model_specific_config,
    is_v2_config,
    validate_config,
    validate_config_v2,
)

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
        required_keys = ["experiment", "model", "data", "training", "output"]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_experiment_type(self):
        """Test that experiment type is 'classifier'."""
        config = get_default_config()
        assert config["experiment"] == "classifier"

    def test_model_defaults(self):
        """Test model configuration defaults."""
        config = get_default_config()
        model = config["model"]

        assert isinstance(model, dict)
        assert "name" in model
        assert "pretrained" in model
        assert "num_classes" in model
        assert "freeze_backbone" in model

        # Check types
        assert isinstance(model["name"], str)
        assert isinstance(model["pretrained"], bool)
        assert isinstance(model["num_classes"], int)
        assert isinstance(model["freeze_backbone"], bool)

    def test_data_defaults(self):
        """Test data configuration defaults."""
        config = get_default_config()
        data = config["data"]

        assert isinstance(data, dict)
        assert "train_path" in data
        assert "batch_size" in data
        assert "num_workers" in data
        assert "image_size" in data
        assert "crop_size" in data

        # Check types and valid ranges
        assert isinstance(data["batch_size"], int)
        assert data["batch_size"] > 0
        assert isinstance(data["num_workers"], int)
        assert data["num_workers"] >= 0

    def test_training_defaults(self):
        """Test training configuration defaults."""
        config = get_default_config()
        training = config["training"]

        assert isinstance(training, dict)
        assert "epochs" in training
        assert "learning_rate" in training
        assert "optimizer" in training
        assert "scheduler" in training
        assert "device" in training

        # Check types and valid ranges
        assert isinstance(training["epochs"], int)
        assert training["epochs"] > 0
        assert isinstance(training["learning_rate"], (int, float))
        assert training["learning_rate"] > 0

    def test_output_defaults(self):
        """Test output configuration defaults."""
        config = get_default_config()
        output = config["output"]

        assert isinstance(output, dict)
        assert "checkpoint_dir" in output
        assert "log_dir" in output


@pytest.mark.unit
class TestValidateConfig:
    """Test configuration validation."""

    def test_valid_default_config(self):
        """Test that default config passes validation."""
        config = get_default_config()
        # Should not raise any exception
        validate_config(config)

    def test_missing_top_level_key(self):
        """Test validation fails with missing top-level key."""
        config = get_default_config()
        del config["model"]

        with pytest.raises(KeyError, match="Missing required config key: model"):
            validate_config(config)

    def test_invalid_experiment_type(self):
        """Test validation fails with wrong experiment type."""
        config = get_default_config()
        config["experiment"] = "invalid"

        with pytest.raises(ValueError, match="Invalid experiment type"):
            validate_config(config)

    def test_invalid_model_name(self):
        """Test validation fails with invalid model name."""
        config = get_default_config()
        config["model"]["name"] = "invalid_model"

        with pytest.raises(ValueError, match="Invalid model name"):
            validate_config(config)

    def test_invalid_num_classes(self):
        """Test validation fails with invalid num_classes."""
        config = get_default_config()

        # Test negative value
        config["model"]["num_classes"] = -1
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            validate_config(config)

        # Test zero
        config["model"]["num_classes"] = 0
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            validate_config(config)

        # Test non-integer
        config["model"]["num_classes"] = 2.5
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            validate_config(config)

    def test_invalid_pretrained(self):
        """Test validation fails with non-boolean pretrained."""
        config = get_default_config()
        config["model"]["pretrained"] = "true"  # String instead of bool

        with pytest.raises(ValueError, match="pretrained must be a boolean"):
            validate_config(config)

    def test_invalid_freeze_backbone(self):
        """Test validation fails with non-boolean freeze_backbone."""
        config = get_default_config()
        config["model"]["freeze_backbone"] = "false"  # String instead of bool

        with pytest.raises(ValueError, match="freeze_backbone must be a boolean"):
            validate_config(config)

    def test_invalid_trainable_layers_type(self):
        """Test validation fails with invalid trainable_layers type."""
        config = get_default_config()
        config["model"]["trainable_layers"] = "Mixed_7*"  # String instead of list

        with pytest.raises(ValueError, match="trainable_layers must be a list or None"):
            validate_config(config)

    def test_invalid_trainable_layers_content(self):
        """Test validation fails with non-string elements in trainable_layers."""
        config = get_default_config()
        config["model"]["trainable_layers"] = ["Mixed_7*", 123]  # Contains non-string

        with pytest.raises(ValueError, match="All trainable_layers must be strings"):
            validate_config(config)

    def test_valid_trainable_layers(self):
        """Test validation succeeds with valid trainable_layers."""
        config = get_default_config()
        config["model"]["trainable_layers"] = ["Mixed_7*", "Mixed_6*"]

        # Should not raise
        validate_config(config)

    def test_invalid_batch_size(self):
        """Test validation fails with invalid batch_size."""
        config = get_default_config()

        # Test zero
        config["data"]["batch_size"] = 0
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_config(config)

        # Test negative
        config["data"]["batch_size"] = -1
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_config(config)

    def test_invalid_num_workers(self):
        """Test validation fails with negative num_workers."""
        config = get_default_config()
        config["data"]["num_workers"] = -1

        with pytest.raises(
            ValueError, match="num_workers must be a non-negative integer"
        ):
            validate_config(config)

    def test_invalid_normalize(self):
        """Test validation fails with invalid normalize option."""
        config = get_default_config()
        config["data"]["normalize"] = "invalid"

        with pytest.raises(ValueError, match="Invalid normalize option"):
            validate_config(config)

    def test_valid_normalize_options(self):
        """Test validation succeeds with all valid normalize options."""
        config = get_default_config()
        valid_options = ["imagenet", "cifar10", "none", None]

        for option in valid_options:
            config["data"]["normalize"] = option
            validate_config(config)  # Should not raise

    def test_invalid_epochs(self):
        """Test validation fails with invalid epochs."""
        config = get_default_config()
        config["training"]["epochs"] = 0

        with pytest.raises(ValueError, match="epochs must be a positive integer"):
            validate_config(config)

    def test_invalid_learning_rate(self):
        """Test validation fails with invalid learning_rate."""
        config = get_default_config()

        # Test zero
        config["training"]["learning_rate"] = 0
        with pytest.raises(ValueError, match="learning_rate must be a positive number"):
            validate_config(config)

        # Test negative
        config["training"]["learning_rate"] = -0.001
        with pytest.raises(ValueError, match="learning_rate must be a positive number"):
            validate_config(config)

    def test_invalid_optimizer(self):
        """Test validation fails with invalid optimizer."""
        config = get_default_config()
        config["training"]["optimizer"] = "invalid"

        with pytest.raises(ValueError, match="Invalid optimizer"):
            validate_config(config)

    def test_invalid_scheduler(self):
        """Test validation fails with invalid scheduler."""
        config = get_default_config()
        config["training"]["scheduler"] = "invalid"

        with pytest.raises(ValueError, match="Invalid scheduler"):
            validate_config(config)

    def test_invalid_device(self):
        """Test validation fails with invalid device."""
        config = get_default_config()
        config["training"]["device"] = "tpu"

        with pytest.raises(ValueError, match="Invalid device"):
            validate_config(config)

    def test_missing_output_dirs(self):
        """Test validation fails with missing output directories."""
        config = get_default_config()

        # Test missing checkpoint_dir
        del config["output"]["checkpoint_dir"]
        with pytest.raises(KeyError, match="checkpoint_dir"):
            validate_config(config)

        # Reset and test missing log_dir
        config = get_default_config()
        del config["output"]["log_dir"]
        with pytest.raises(KeyError, match="log_dir"):
            validate_config(config)


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


@pytest.mark.component
class TestConfigFileValidation:
    """Test validation of actual config files."""

    def test_baseline_config_file(self):
        """Test that baseline.yaml is valid."""
        config_path = Path("configs/classifier/baseline.yaml")

        if not config_path.exists():
            pytest.skip("baseline.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should not raise
        validate_config(config)

    def test_inceptionv3_config_file(self):
        """Test that inceptionv3.yaml is valid."""
        config_path = Path("configs/classifier/inceptionv3.yaml")

        if not config_path.exists():
            pytest.skip("inceptionv3.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should not raise
        validate_config(config)

    def test_baseline_config_structure(self):
        """Test baseline config has expected structure."""
        config_path = Path("configs/classifier/baseline.yaml")

        if not config_path.exists():
            pytest.skip("baseline.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check experiment type
        assert config["experiment"] == "classifier"

        # Check it uses ResNet50 (V2 structure)
        if is_v2_config(config):
            assert config["model"]["architecture"]["name"] == "resnet50"
            assert config["data"]["preprocessing"]["image_size"] == 256
            assert config["data"]["preprocessing"]["crop_size"] == 224
            assert config["data"]["preprocessing"]["normalize"] == "imagenet"
        else:
            # V1 structure
            assert config["model"]["name"] == "resnet50"
            assert config["data"]["image_size"] == 256
            assert config["data"]["crop_size"] == 224
            assert config["data"]["normalize"] == "imagenet"

    def test_inceptionv3_config_structure(self):
        """Test InceptionV3 config has expected structure."""
        config_path = Path("configs/classifier/inceptionv3.yaml")

        if not config_path.exists():
            pytest.skip("inceptionv3.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check experiment type
        assert config["experiment"] == "classifier"

        # Check it uses InceptionV3 (V2 structure)
        if is_v2_config(config):
            assert config["model"]["architecture"]["name"] == "inceptionv3"
            assert config["data"]["preprocessing"]["image_size"] == 320
            assert config["data"]["preprocessing"]["crop_size"] == 299
            assert config["data"]["preprocessing"]["normalize"] == "imagenet"
        else:
            # V1 structure
            assert config["model"]["name"] == "inceptionv3"
            assert config["data"]["image_size"] == 320
            assert config["data"]["crop_size"] == 299
            assert config["data"]["normalize"] == "imagenet"

        # Check dropout parameter (location depends on config version)
        if is_v2_config(config):
            assert "dropout" in config["model"]["regularization"]
        else:
            assert "dropout" in config["model"]


# ============================================================================
# V2 Configuration Tests
# ============================================================================


def get_v2_default_config():
    """Helper function to get a valid V2 config for testing."""
    return {
        "experiment": "classifier",
        "mode": "train",
        "compute": {"device": "cuda", "seed": None},
        "model": {
            "architecture": {"name": "resnet50", "num_classes": 2},
            "initialization": {
                "pretrained": True,
                "freeze_backbone": False,
                "trainable_layers": None,
            },
            "regularization": {"dropout": 0.5},
        },
        "data": {
            "paths": {"train": "data/train", "val": "data/val"},
            "loading": {
                "batch_size": 32,
                "num_workers": 4,
                "pin_memory": True,
                "shuffle_train": True,
                "drop_last": False,
            },
            "preprocessing": {
                "image_size": 256,
                "crop_size": 224,
                "normalize": "imagenet",
            },
            "augmentation": {
                "horizontal_flip": True,
                "rotation_degrees": 0,
                "color_jitter": {
                    "enabled": False,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1,
                },
            },
        },
        "output": {
            "base_dir": "outputs",
            "subdirs": {"logs": "logs", "checkpoints": "checkpoints"},
        },
        "training": {
            "epochs": 100,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_norm": None,
            },
            "scheduler": {"type": "cosine", "T_max": "auto", "eta_min": 1.0e-6},
            "checkpointing": {
                "save_frequency": 10,
                "save_best_only": True,
                "save_optimizer": True,
            },
            "validation": {
                "enabled": True,
                "frequency": 1,
                "metric": "accuracy",
                "early_stopping_patience": None,
            },
            "performance": {
                "use_amp": False,
                "use_tf32": True,
                "cudnn_benchmark": True,
                "compile_model": False,
            },
            "resume": {
                "enabled": False,
                "checkpoint": None,
                "reset_optimizer": False,
                "reset_scheduler": False,
            },
        },
    }


@pytest.mark.unit
class TestIsV2Config:
    """Test V2 config detection."""

    def test_detects_v1_config(self):
        """Test that V1 config is correctly identified."""
        v1_config = get_default_config()
        assert not is_v2_config(v1_config)

    def test_detects_v2_config(self):
        """Test that V2 config is correctly identified."""
        v2_config = get_v2_default_config()
        assert is_v2_config(v2_config)

    def test_empty_config(self):
        """Test empty config is not identified as V2."""
        assert not is_v2_config({})


@pytest.mark.unit
class TestValidateConfigV2:
    """Test V2 configuration validation."""

    def test_valid_v2_config(self):
        """Test that valid V2 config passes validation."""
        config = get_v2_default_config()
        # Should not raise
        validate_config_v2(config)

    def test_missing_mode_key(self):
        """Test validation fails with missing mode key."""
        config = get_v2_default_config()
        del config["mode"]

        with pytest.raises(KeyError, match="Missing required config key: mode"):
            validate_config_v2(config)

    def test_missing_compute_key(self):
        """Test validation fails with missing compute key."""
        config = get_v2_default_config()
        del config["compute"]

        with pytest.raises(KeyError, match="Missing required config key: compute"):
            validate_config_v2(config)

    def test_invalid_mode(self):
        """Test validation fails with invalid mode."""
        config = get_v2_default_config()
        config["mode"] = "invalid"

        with pytest.raises(ValueError, match="Invalid mode"):
            validate_config_v2(config)

    def test_valid_modes(self):
        """Test validation succeeds with all valid modes."""
        config = get_v2_default_config()

        for mode in ["train", "evaluate"]:
            config["mode"] = mode
            # For evaluate mode, need to add evaluation section
            if mode == "evaluate":
                config["evaluation"] = {
                    "checkpoint": "path/to/checkpoint.pth",
                    "data": {"test_path": "data/test", "batch_size": 32},
                    "output": {
                        "save_predictions": True,
                        "save_confusion_matrix": True,
                        "save_metrics": True,
                    },
                }
            validate_config_v2(config)

    def test_invalid_device_v2(self):
        """Test validation fails with invalid device in V2."""
        config = get_v2_default_config()
        config["compute"]["device"] = "tpu"

        with pytest.raises(ValueError, match="Invalid device"):
            validate_config_v2(config)

    def test_valid_devices_v2(self):
        """Test validation succeeds with all valid devices in V2."""
        config = get_v2_default_config()

        for device in ["cuda", "cpu", "auto"]:
            config["compute"]["device"] = device
            validate_config_v2(config)

    def test_missing_architecture_section(self):
        """Test validation fails with missing architecture section."""
        config = get_v2_default_config()
        del config["model"]["architecture"]

        with pytest.raises(
            KeyError, match="Missing required field: model.architecture"
        ):
            validate_config_v2(config)

    def test_invalid_model_name_v2(self):
        """Test validation fails with invalid model name in V2."""
        config = get_v2_default_config()
        config["model"]["architecture"]["name"] = "invalid_model"

        with pytest.raises(ValueError, match="Invalid model name"):
            validate_config_v2(config)

    def test_missing_data_sections(self):
        """Test validation fails with missing data sections."""
        config = get_v2_default_config()

        for section in ["paths", "loading", "preprocessing", "augmentation"]:
            config_copy = get_v2_default_config()
            del config_copy["data"][section]

            with pytest.raises(
                KeyError, match=f"Missing required field: data.{section}"
            ):
                validate_config_v2(config_copy)

    def test_missing_output_subdirs(self):
        """Test validation fails with missing output subdirs."""
        config = get_v2_default_config()
        del config["output"]["subdirs"]

        with pytest.raises(KeyError, match="Missing required field: output.subdirs"):
            validate_config_v2(config)

    def test_missing_required_subdirs(self):
        """Test validation fails with missing required subdirs."""
        config = get_v2_default_config()

        for subdir in ["logs", "checkpoints"]:
            config_copy = get_v2_default_config()
            del config_copy["output"]["subdirs"][subdir]

            with pytest.raises(
                KeyError, match=f"Missing required field: output.subdirs.{subdir}"
            ):
                validate_config_v2(config_copy)

    def test_invalid_optimizer_type(self):
        """Test validation fails with invalid optimizer type."""
        config = get_v2_default_config()
        config["training"]["optimizer"]["type"] = "invalid"

        with pytest.raises(ValueError, match="Invalid optimizer"):
            validate_config_v2(config)

    def test_invalid_scheduler_type(self):
        """Test validation fails with invalid scheduler type."""
        config = get_v2_default_config()
        config["training"]["scheduler"]["type"] = "invalid"

        with pytest.raises(ValueError, match="Invalid scheduler"):
            validate_config_v2(config)

    def test_evaluate_mode_requires_checkpoint(self):
        """Test validation fails in evaluate mode without checkpoint."""
        config = get_v2_default_config()
        config["mode"] = "evaluate"
        config["evaluation"] = {
            "checkpoint": None,  # Invalid: should be a path
            "data": {"test_path": "data/test", "batch_size": 32},
            "output": {
                "save_predictions": True,
                "save_confusion_matrix": True,
                "save_metrics": True,
            },
        }

        with pytest.raises(
            ValueError, match="evaluation.checkpoint is required for evaluate mode"
        ):
            validate_config_v2(config)


@pytest.mark.unit
class TestValidateConfigAutoDetect:
    """Test that validate_config auto-detects V1 vs V2."""

    def test_validates_v1_config(self):
        """Test that validate_config handles V1 configs."""
        v1_config = get_default_config()
        # Should not raise
        validate_config(v1_config)

    def test_validates_v2_config(self):
        """Test that validate_config handles V2 configs."""
        v2_config = get_v2_default_config()
        # Should not raise
        validate_config(v2_config)

    def test_v1_config_triggers_deprecation_warning(self):
        """Test that V1 config triggers deprecation warning."""
        v1_config = get_default_config()

        with pytest.warns(DeprecationWarning):
            validate_config(v1_config)


@pytest.mark.component
class TestV2ConfigFiles:
    """Test actual V2 config files."""

    def test_default_v2_config_file(self):
        """Test that default.yaml (V1) is valid."""
        config_path = Path("src/experiments/classifier/default.yaml")

        if not config_path.exists():
            pytest.skip("default.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should be V1 format (for backward compatibility)
        assert not is_v2_config(config), "default.yaml should be V1 format"

        # Should validate (with deprecation warning)
        with pytest.warns(DeprecationWarning):
            validate_config(config)

    def test_baseline_v2_config_file(self):
        """Test that baseline.yaml (V2) is valid."""
        config_path = Path("configs/classifier/baseline.yaml")

        if not config_path.exists():
            pytest.skip("baseline.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should be V2 format
        assert is_v2_config(config)

        # Should validate
        validate_config(config)

    def test_inceptionv3_v2_config_file(self):
        """Test that inceptionv3.yaml (V2) is valid."""
        config_path = Path("configs/classifier/inceptionv3.yaml")

        if not config_path.exists():
            pytest.skip("inceptionv3.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should be V2 format
        assert is_v2_config(config)

        # Should validate
        validate_config(config)

    def test_legacy_config_file(self):
        """Test that legacy.yaml (V1) is valid."""
        config_path = Path("configs/classifier/legacy.yaml")

        if not config_path.exists():
            pytest.skip("legacy.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should be V1 format
        assert not is_v2_config(config)

        # Should still validate with deprecation warning
        with pytest.warns(DeprecationWarning):
            validate_config(config)
