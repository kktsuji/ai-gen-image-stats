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
            "split_file": "outputs/splits/train_val_split.json",
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
            validate_config(config)

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

    def test_save_latest_missing_defaults_to_true(self):
        """save_latest absent from config is accepted (defaults to True)."""
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
        # Don't add evaluation section
        with pytest.raises(
            KeyError,
            match="Missing required section: evaluation",
        ):
            validate_config(config)


@pytest.mark.component
class TestConfigFiles:
    """Test actual config files (classifier-example.yaml)."""

    def test_default_config_file(self):
        """Test that classifier-example.yaml is valid."""
        config_path = _PROJECT_ROOT / "configs/classifier-example.yaml"

        if not config_path.exists():
            pytest.skip("configs/classifier-example.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should validate
        validate_config(config)
