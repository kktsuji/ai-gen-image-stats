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
    validate_config,
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
        required_keys = [
            "experiment",
            "mode",
            "compute",
            "model",
            "data",
            "training",
            "output",
        ]
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
        assert "architecture" in model
        assert "initialization" in model
        assert "regularization" in model

        # Check architecture section
        arch = model["architecture"]
        assert "name" in arch
        assert "num_classes" in arch
        assert isinstance(arch["name"], str)
        assert isinstance(arch["num_classes"], int)

        # Check initialization section
        init = model["initialization"]
        assert "pretrained" in init
        assert "freeze_backbone" in init
        assert isinstance(init["pretrained"], bool)
        assert isinstance(init["freeze_backbone"], bool)

    def test_data_defaults(self):
        """Test data configuration defaults."""
        config = get_default_config()
        data = config["data"]

        assert isinstance(data, dict)
        assert "split_file" in data
        assert "loading" in data
        assert "preprocessing" in data
        assert "augmentation" in data

        # Check loading section
        loading = data["loading"]
        assert "batch_size" in loading
        assert "num_workers" in loading
        assert isinstance(loading["batch_size"], int)
        assert loading["batch_size"] > 0
        assert isinstance(loading["num_workers"], int)
        assert loading["num_workers"] >= 0

        # Check preprocessing section
        preprocessing = data["preprocessing"]
        assert "image_size" in preprocessing
        assert "crop_size" in preprocessing

    def test_training_defaults(self):
        """Test training configuration defaults."""
        config = get_default_config()
        training = config["training"]

        assert isinstance(training, dict)
        assert "epochs" in training
        assert "optimizer" in training
        assert "scheduler" in training
        assert "checkpointing" in training
        assert "validation" in training

        # Check optimizer section
        optimizer = training["optimizer"]
        assert "type" in optimizer
        assert "learning_rate" in optimizer

        # Check types and valid ranges
        assert isinstance(training["epochs"], int)
        assert training["epochs"] > 0
        assert isinstance(optimizer["learning_rate"], (int, float))
        assert optimizer["learning_rate"] > 0

    def test_output_defaults(self):
        """Test output configuration defaults."""
        config = get_default_config()
        output = config["output"]

        assert isinstance(output, dict)
        assert "base_dir" in output
        assert "subdirs" in output
        assert "logs" in output["subdirs"]
        assert "checkpoints" in output["subdirs"]


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

        # Check it uses ResNet50
        assert config["model"]["architecture"]["name"].lower() == "resnet50"
        assert config["data"]["preprocessing"]["image_size"] == 256
        assert config["data"]["preprocessing"]["crop_size"] == 224
        assert config["data"]["preprocessing"]["normalize"] == "imagenet"

    def test_inceptionv3_config_structure(self):
        """Test InceptionV3 config has expected structure."""
        config_path = Path("configs/classifier/inceptionv3.yaml")

        if not config_path.exists():
            pytest.skip("inceptionv3.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check experiment type
        assert config["experiment"] == "classifier"

        # Check it uses InceptionV3
        assert config["model"]["architecture"]["name"].lower() == "inceptionv3"
        assert config["data"]["preprocessing"]["image_size"] == 320
        assert config["data"]["preprocessing"]["crop_size"] == 299
        assert config["data"]["preprocessing"]["normalize"] == "imagenet"

        # Check dropout parameter
        assert "dropout" in config["model"]["regularization"]


# ============================================================================
# ============================================================================
# Configuration Tests
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
        config = get_v2_default_config()

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
        config = get_v2_default_config()

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
@pytest.mark.component
class TestConfigFiles:
    """Test actual config files."""

    def test_default_config_file(self):
        """Test that default.yaml is valid."""
        config_path = Path("src/experiments/classifier/default.yaml")

        if not config_path.exists():
            pytest.skip("default.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should validate
        validate_config(config)

    def test_baseline_config_file(self):
        """Test that baseline.yaml is valid."""
        config_path = Path("configs/classifier/baseline.yaml")

        if not config_path.exists():
            pytest.skip("baseline.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should validate
        validate_config(config)

    def test_inceptionv3_config_file(self):
        """Test that inceptionv3.yaml is valid."""
        config_path = Path("configs/classifier/inceptionv3.yaml")

        if not config_path.exists():
            pytest.skip("inceptionv3.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should validate
        validate_config(config)
