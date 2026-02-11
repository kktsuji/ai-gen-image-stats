"""Tests for Diffusion Configuration

This module tests the diffusion model configuration management, including:
- Default configuration generation
- Configuration validation
- Resolution-specific configuration overrides
"""

import json
from pathlib import Path

import pytest

from src.experiments.diffusion.config import (
    get_default_config,
    get_resolution_config,
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
        """Test that experiment type is 'diffusion'."""
        config = get_default_config()
        assert config["experiment"] == "diffusion"

    def test_model_defaults(self):
        """Test model configuration defaults."""
        config = get_default_config()
        model = config["model"]

        assert isinstance(model, dict)
        assert "image_size" in model
        assert "in_channels" in model
        assert "model_channels" in model
        assert "channel_multipliers" in model
        assert "num_timesteps" in model
        assert "beta_schedule" in model

        # Check types
        assert isinstance(model["image_size"], int)
        assert isinstance(model["in_channels"], int)
        assert isinstance(model["model_channels"], int)
        assert isinstance(model["channel_multipliers"], list)
        assert isinstance(model["num_timesteps"], int)
        assert isinstance(model["beta_schedule"], str)

    def test_data_defaults(self):
        """Test data configuration defaults."""
        config = get_default_config()
        data = config["data"]

        assert isinstance(data, dict)
        assert "train_path" in data
        assert "batch_size" in data
        assert "num_workers" in data
        assert "image_size" in data

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
        assert "device" in training
        assert "use_ema" in training
        assert "ema_decay" in training

        # Check types and valid ranges
        assert isinstance(training["epochs"], int)
        assert training["epochs"] > 0
        assert isinstance(training["learning_rate"], (int, float))
        assert training["learning_rate"] > 0
        assert isinstance(training["use_ema"], bool)

    def test_generation_defaults(self):
        """Test generation configuration defaults."""
        config = get_default_config()
        generation = config["generation"]

        assert isinstance(generation, dict)
        assert "sample_images" in generation
        assert "sample_interval" in generation
        assert "samples_per_class" in generation
        assert "guidance_scale" in generation

        # Check types
        assert isinstance(generation["sample_images"], bool)
        assert isinstance(generation["sample_interval"], int)
        assert isinstance(generation["samples_per_class"], int)
        assert isinstance(generation["guidance_scale"], (int, float))

    def test_output_defaults(self):
        """Test output configuration defaults."""
        config = get_default_config()
        output = config["output"]

        assert isinstance(output, dict)
        assert "checkpoint_dir" in output
        assert "log_dir" in output

    def test_image_size_consistency(self):
        """Test that model.image_size matches data.image_size."""
        config = get_default_config()
        assert config["model"]["image_size"] == config["data"]["image_size"]


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

    def test_invalid_image_size(self):
        """Test validation fails with invalid image_size."""
        config = get_default_config()

        # Test negative value
        config["model"]["image_size"] = -1
        with pytest.raises(ValueError, match="image_size must be a positive integer"):
            validate_config(config)

        # Test zero
        config["model"]["image_size"] = 0
        with pytest.raises(ValueError, match="image_size must be a positive integer"):
            validate_config(config)

    def test_invalid_in_channels(self):
        """Test validation fails with invalid in_channels."""
        config = get_default_config()
        config["model"]["in_channels"] = 0

        with pytest.raises(ValueError, match="in_channels must be a positive integer"):
            validate_config(config)

    def test_invalid_model_channels(self):
        """Test validation fails with invalid model_channels."""
        config = get_default_config()
        config["model"]["model_channels"] = -1

        with pytest.raises(
            ValueError, match="model_channels must be a positive integer"
        ):
            validate_config(config)

    def test_invalid_channel_multipliers_type(self):
        """Test validation fails with non-list channel_multipliers."""
        config = get_default_config()
        config["model"]["channel_multipliers"] = (1, 2, 4)  # Tuple instead of list

        with pytest.raises(ValueError, match="channel_multipliers must be a list"):
            validate_config(config)

    def test_invalid_channel_multipliers_values(self):
        """Test validation fails with invalid channel_multiplier values."""
        config = get_default_config()
        config["model"]["channel_multipliers"] = [1, 2, -1]

        with pytest.raises(
            ValueError, match="All channel_multipliers must be positive integers"
        ):
            validate_config(config)

    def test_invalid_num_classes(self):
        """Test validation fails with invalid num_classes."""
        config = get_default_config()

        # Test negative value (when not None)
        config["model"]["num_classes"] = -1
        with pytest.raises(
            ValueError, match="num_classes must be a positive integer or None"
        ):
            validate_config(config)

        # Test zero
        config["model"]["num_classes"] = 0
        with pytest.raises(
            ValueError, match="num_classes must be a positive integer or None"
        ):
            validate_config(config)

    def test_valid_num_classes_none(self):
        """Test validation succeeds with num_classes=None."""
        config = get_default_config()
        config["model"]["num_classes"] = None

        # Should not raise
        validate_config(config)

    def test_invalid_num_timesteps(self):
        """Test validation fails with invalid num_timesteps."""
        config = get_default_config()
        config["model"]["num_timesteps"] = 0

        with pytest.raises(
            ValueError, match="num_timesteps must be a positive integer"
        ):
            validate_config(config)

    def test_invalid_beta_schedule(self):
        """Test validation fails with invalid beta_schedule."""
        config = get_default_config()
        config["model"]["beta_schedule"] = "invalid"

        with pytest.raises(ValueError, match="Invalid beta_schedule"):
            validate_config(config)

    def test_valid_beta_schedules(self):
        """Test validation succeeds with all valid beta schedules."""
        config = get_default_config()
        valid_schedules = ["linear", "cosine", "quadratic", "sigmoid"]

        for schedule in valid_schedules:
            config["model"]["beta_schedule"] = schedule
            validate_config(config)  # Should not raise

    def test_invalid_beta_start(self):
        """Test validation fails with invalid beta_start."""
        config = get_default_config()

        # Test negative
        config["model"]["beta_start"] = -0.1
        with pytest.raises(
            ValueError, match="beta_start must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test >= 1
        config["model"]["beta_start"] = 1.0
        with pytest.raises(
            ValueError, match="beta_start must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_invalid_beta_end(self):
        """Test validation fails with invalid beta_end."""
        config = get_default_config()

        # Test negative
        config["model"]["beta_end"] = -0.1
        with pytest.raises(
            ValueError, match="beta_end must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test >= 1
        config["model"]["beta_end"] = 1.5
        with pytest.raises(
            ValueError, match="beta_end must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_beta_start_greater_than_beta_end(self):
        """Test validation fails when beta_start >= beta_end."""
        config = get_default_config()
        config["model"]["beta_start"] = 0.02
        config["model"]["beta_end"] = 0.01

        with pytest.raises(ValueError, match="beta_start must be less than beta_end"):
            validate_config(config)

    def test_invalid_class_dropout_prob(self):
        """Test validation fails with invalid class_dropout_prob."""
        config = get_default_config()

        # Test negative
        config["model"]["class_dropout_prob"] = -0.1
        with pytest.raises(
            ValueError, match="class_dropout_prob must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test > 1
        config["model"]["class_dropout_prob"] = 1.5
        with pytest.raises(
            ValueError, match="class_dropout_prob must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_invalid_use_attention_type(self):
        """Test validation fails with non-list use_attention."""
        config = get_default_config()
        config["model"]["use_attention"] = (False, False, True)  # Tuple instead of list

        with pytest.raises(ValueError, match="use_attention must be a list"):
            validate_config(config)

    def test_invalid_use_attention_values(self):
        """Test validation fails with non-boolean use_attention values."""
        config = get_default_config()
        config["model"]["use_attention"] = [False, 1, True]

        with pytest.raises(
            ValueError, match="All use_attention values must be booleans"
        ):
            validate_config(config)

    def test_use_attention_length_mismatch(self):
        """Test validation fails when use_attention length doesn't match channel_multipliers."""
        config = get_default_config()
        config["model"]["channel_multipliers"] = [1, 2, 4]
        config["model"]["use_attention"] = [False, True]  # Wrong length

        with pytest.raises(
            ValueError,
            match="use_attention length must match channel_multipliers length",
        ):
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

    def test_invalid_rotation_degrees(self):
        """Test validation fails with negative rotation_degrees."""
        config = get_default_config()
        config["data"]["rotation_degrees"] = -10

        with pytest.raises(
            ValueError, match="rotation_degrees must be a non-negative number"
        ):
            validate_config(config)

    def test_image_size_mismatch(self):
        """Test validation fails when data.image_size != model.image_size."""
        config = get_default_config()
        config["model"]["image_size"] = 64
        config["data"]["image_size"] = 40

        with pytest.raises(
            ValueError, match="data.image_size.*must match.*model.image_size"
        ):
            validate_config(config)

    def test_conditional_requires_labels(self):
        """Test validation fails when conditional generation doesn't have labels."""
        config = get_default_config()
        config["model"]["num_classes"] = 2  # Conditional
        config["data"]["return_labels"] = False  # But no labels

        with pytest.raises(
            ValueError,
            match="data.return_labels must be True when model.num_classes is set",
        ):
            validate_config(config)

    def test_valid_conditional_config(self):
        """Test validation succeeds with proper conditional configuration."""
        config = get_default_config()
        config["model"]["num_classes"] = 2
        config["data"]["return_labels"] = True

        # Should not raise
        validate_config(config)

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
        config["training"]["optimizer"] = "sgd"

        with pytest.raises(ValueError, match="Invalid optimizer"):
            validate_config(config)

    def test_valid_optimizers(self):
        """Test validation succeeds with all valid optimizers."""
        config = get_default_config()
        valid_optimizers = ["adam", "adamw"]

        for optimizer in valid_optimizers:
            config["training"]["optimizer"] = optimizer
            validate_config(config)  # Should not raise

    def test_invalid_scheduler(self):
        """Test validation fails with invalid scheduler."""
        config = get_default_config()
        config["training"]["scheduler"] = "invalid"

        with pytest.raises(ValueError, match="Invalid scheduler"):
            validate_config(config)

    def test_valid_schedulers(self):
        """Test validation succeeds with all valid schedulers."""
        config = get_default_config()
        valid_schedulers = ["cosine", "step", "plateau", "none", None]

        for scheduler in valid_schedulers:
            config["training"]["scheduler"] = scheduler
            validate_config(config)  # Should not raise

    def test_invalid_device(self):
        """Test validation fails with invalid device."""
        config = get_default_config()
        config["training"]["device"] = "tpu"

        with pytest.raises(ValueError, match="Invalid device"):
            validate_config(config)

    def test_invalid_use_ema(self):
        """Test validation fails with non-boolean use_ema."""
        config = get_default_config()
        config["training"]["use_ema"] = "true"

        with pytest.raises(ValueError, match="use_ema must be a boolean"):
            validate_config(config)

    def test_invalid_ema_decay(self):
        """Test validation fails with invalid ema_decay."""
        config = get_default_config()

        # Test negative
        config["training"]["ema_decay"] = -0.1
        with pytest.raises(
            ValueError, match="ema_decay must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test >= 1
        config["training"]["ema_decay"] = 1.0
        with pytest.raises(
            ValueError, match="ema_decay must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_invalid_use_amp(self):
        """Test validation fails with non-boolean use_amp."""
        config = get_default_config()
        config["training"]["use_amp"] = "false"

        with pytest.raises(ValueError, match="use_amp must be a boolean"):
            validate_config(config)

    def test_invalid_gradient_clip_norm(self):
        """Test validation fails with invalid gradient_clip_norm."""
        config = get_default_config()
        config["training"]["gradient_clip_norm"] = -1.0

        with pytest.raises(
            ValueError, match="gradient_clip_norm must be a positive number or None"
        ):
            validate_config(config)

    def test_valid_gradient_clip_norm_none(self):
        """Test validation succeeds with gradient_clip_norm=None."""
        config = get_default_config()
        config["training"]["gradient_clip_norm"] = None

        # Should not raise
        validate_config(config)

    def test_invalid_sample_images(self):
        """Test validation fails with non-boolean sample_images."""
        config = get_default_config()
        config["generation"]["sample_images"] = "true"

        with pytest.raises(ValueError, match="sample_images must be a boolean"):
            validate_config(config)

    def test_invalid_sample_interval(self):
        """Test validation fails with invalid sample_interval."""
        config = get_default_config()
        config["generation"]["sample_interval"] = 0

        with pytest.raises(
            ValueError, match="sample_interval must be a positive integer"
        ):
            validate_config(config)

    def test_invalid_samples_per_class(self):
        """Test validation fails with invalid samples_per_class."""
        config = get_default_config()
        config["generation"]["samples_per_class"] = -1

        with pytest.raises(
            ValueError, match="samples_per_class must be a positive integer"
        ):
            validate_config(config)

    def test_invalid_guidance_scale(self):
        """Test validation fails with invalid guidance_scale."""
        config = get_default_config()
        config["generation"]["guidance_scale"] = 0.5

        with pytest.raises(ValueError, match="guidance_scale must be a number >= 1.0"):
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
class TestGetResolutionConfig:
    """Test resolution-specific configuration overrides."""

    def test_resolution_40_config(self):
        """Test 40x40 resolution configuration."""
        config = get_resolution_config(40)

        assert isinstance(config, dict)
        assert "model" in config
        assert "data" in config
        assert config["model"]["image_size"] == 40
        assert config["data"]["image_size"] == 40
        assert config["model"]["model_channels"] == 64

    def test_resolution_64_config(self):
        """Test 64x64 resolution configuration."""
        config = get_resolution_config(64)

        assert isinstance(config, dict)
        assert config["model"]["image_size"] == 64
        assert config["data"]["image_size"] == 64
        assert config["model"]["model_channels"] == 128

    def test_resolution_128_config(self):
        """Test 128x128 resolution configuration."""
        config = get_resolution_config(128)

        assert isinstance(config, dict)
        assert config["model"]["image_size"] == 128
        assert config["data"]["image_size"] == 128

    def test_resolution_256_config(self):
        """Test 256x256 resolution configuration."""
        config = get_resolution_config(256)

        assert isinstance(config, dict)
        assert config["model"]["image_size"] == 256
        assert config["data"]["image_size"] == 256

    def test_invalid_resolution(self):
        """Test error with invalid resolution."""
        with pytest.raises(ValueError, match="Unsupported image_size"):
            get_resolution_config(100)

    def test_all_supported_resolutions(self):
        """Test that all supported resolutions have configurations."""
        supported_resolutions = [40, 64, 128, 256]

        for resolution in supported_resolutions:
            config = get_resolution_config(resolution)
            assert isinstance(config, dict)
            assert config["model"]["image_size"] == resolution
            assert config["data"]["image_size"] == resolution

    def test_channel_multipliers_length(self):
        """Test that use_attention matches channel_multipliers length."""
        supported_resolutions = [40, 64, 128, 256]

        for resolution in supported_resolutions:
            config = get_resolution_config(resolution)
            multipliers = config["model"]["channel_multipliers"]
            attention = config["model"]["use_attention"]
            assert len(attention) == len(multipliers)


# ============================================================================
# Component Tests - With File I/O
# ============================================================================


@pytest.mark.component
class TestConfigFileValidation:
    """Test validation of actual config files."""

    def test_default_config_file(self):
        """Test that default.json is valid."""
        config_path = Path("configs/diffusion/default.json")

        if not config_path.exists():
            pytest.skip("default.json not found")

        with open(config_path) as f:
            config = json.load(f)

        # Should not raise
        validate_config(config)

    def test_default_config_structure(self):
        """Test default config has expected structure."""
        config_path = Path("configs/diffusion/default.json")

        if not config_path.exists():
            pytest.skip("default.json not found")

        with open(config_path) as f:
            config = json.load(f)

        # Check experiment type
        assert config["experiment"] == "diffusion"

        # Check model type
        assert config["model"]["image_size"] == 40
        assert config["model"]["in_channels"] == 3
        assert config["model"]["beta_schedule"] == "cosine"

        # Check data configuration
        assert config["data"]["image_size"] == 40

        # Check training configuration
        assert config["training"]["use_ema"] is True
        assert isinstance(config["training"]["epochs"], int)

        # Check generation configuration
        assert "generation" in config
        assert "sample_images" in config["generation"]
