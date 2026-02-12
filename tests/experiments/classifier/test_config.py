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

        # Check it uses ResNet50
        assert config["model"]["name"] == "resnet50"

        # Check standard ImageNet preprocessing
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

        # Check it uses InceptionV3
        assert config["model"]["name"] == "inceptionv3"

        # Check InceptionV3-specific preprocessing
        assert config["data"]["image_size"] == 320
        assert config["data"]["crop_size"] == 299
        assert config["data"]["normalize"] == "imagenet"

        # Check dropout parameter
        assert "dropout" in config["model"]
