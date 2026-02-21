"""Tests for Diffusion Configuration

This module tests the diffusion model configuration management, including:
- Default configuration generation
- Configuration validation
- Resolution-specific configuration overrides
"""

from pathlib import Path

import pytest
import yaml

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
        """Test that default config has all required top-level keys (V2)."""
        config = get_default_config()
        required_keys = [
            "experiment",
            "compute",
            "model",
            "data",
            "training",
            "output",
            "generation",
        ]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_experiment_type(self):
        """Test that experiment type is 'diffusion'."""
        config = get_default_config()
        assert config["experiment"] == "diffusion"

    def test_device_in_compute_section(self):
        """Test that device is in compute section (V2)."""
        config = get_default_config()
        assert "compute" in config
        assert "device" in config["compute"]
        assert isinstance(config["compute"]["device"], str)

    def test_seed_in_compute_section(self):
        """Test that seed is in compute section (V2)."""
        config = get_default_config()
        assert "compute" in config
        assert "seed" in config["compute"]
        # seed can be None or int

    def test_model_defaults(self):
        """Test model configuration defaults (V2)."""
        config = get_default_config()
        model = config["model"]

        assert isinstance(model, dict)
        assert "architecture" in model
        assert "diffusion" in model
        assert "conditioning" in model

        # Check architecture subsection
        arch = model["architecture"]
        assert "image_size" in arch
        assert "in_channels" in arch
        assert "model_channels" in arch
        assert "channel_multipliers" in arch
        assert isinstance(arch["image_size"], int)
        assert isinstance(arch["in_channels"], int)
        assert isinstance(arch["model_channels"], int)
        assert isinstance(arch["channel_multipliers"], list)

        # Check diffusion subsection
        diff = model["diffusion"]
        assert "num_timesteps" in diff
        assert "beta_schedule" in diff
        assert isinstance(diff["num_timesteps"], int)
        assert isinstance(diff["beta_schedule"], str)

        # Check conditioning subsection
        cond = model["conditioning"]
        assert "type" in cond
        assert "num_classes" in cond

    def test_data_defaults(self):
        """Test data configuration defaults (V2)."""
        config = get_default_config()
        data = config["data"]

        assert isinstance(data, dict)
        assert "paths" in data
        assert "loading" in data
        assert "augmentation" in data

        # Check paths subsection
        paths = data["paths"]
        assert "train" in paths

        # Check loading subsection
        loading = data["loading"]
        assert "batch_size" in loading
        assert "num_workers" in loading

        # Check types and valid ranges
        assert isinstance(loading["batch_size"], int)
        assert loading["batch_size"] > 0
        assert isinstance(loading["num_workers"], int)
        assert loading["num_workers"] >= 0

    def test_training_defaults(self):
        """Test training configuration defaults (V2)."""
        config = get_default_config()
        training = config["training"]

        assert isinstance(training, dict)
        assert "epochs" in training
        assert "optimizer" in training
        assert "ema" in training
        assert "checkpointing" in training
        assert "validation" in training
        assert "visualization" in training

        # Check optimizer subsection
        optimizer = training["optimizer"]
        assert "type" in optimizer
        assert "learning_rate" in optimizer
        assert isinstance(optimizer["learning_rate"], (int, float))
        assert optimizer["learning_rate"] > 0

        # Check ema subsection
        ema = training["ema"]
        assert "enabled" in ema
        assert "decay" in ema
        assert isinstance(ema["enabled"], bool)

        # Check types and valid ranges
        assert isinstance(training["epochs"], int)
        assert training["epochs"] > 0

    def test_training_nested_validation(self):
        """Test that validation is nested under training."""
        config = get_default_config()
        assert "validation" in config["training"]
        validation = config["training"]["validation"]
        assert "frequency" in validation
        assert "metric" in validation
        assert isinstance(validation["frequency"], int)
        assert isinstance(validation["metric"], str)

    def test_training_nested_visualization(self):
        """Test that visualization is nested under training (V2)."""
        config = get_default_config()
        assert "visualization" in config["training"]
        visualization = config["training"]["visualization"]
        assert "enabled" in visualization
        assert "log_images_interval" in visualization
        assert "log_sample_comparison_interval" in visualization
        assert "log_denoising_interval" in visualization
        assert "num_samples" in visualization
        assert "guidance_scale" in visualization
        assert isinstance(visualization["enabled"], bool)
        assert isinstance(visualization["log_images_interval"], int)
        assert isinstance(visualization["num_samples"], int)
        assert isinstance(visualization["guidance_scale"], (int, float))

    def test_generation_defaults(self):
        """Test generation configuration defaults (V2)."""
        config = get_default_config()
        generation = config["generation"]

        assert isinstance(generation, dict)
        assert "checkpoint" in generation
        assert "sampling" in generation
        assert "output" in generation

        # Check sampling subsection
        sampling = generation["sampling"]
        assert "num_samples" in sampling
        assert "batch_size" in sampling
        assert "guidance_scale" in sampling
        assert "use_ema" in sampling
        assert "ema_decay" in sampling
        assert isinstance(sampling["num_samples"], int)
        assert isinstance(sampling["batch_size"], int)
        assert isinstance(sampling["guidance_scale"], (int, float))
        assert isinstance(sampling["use_ema"], bool)
        assert isinstance(sampling["ema_decay"], (int, float))

        # Check output subsection
        output = generation["output"]
        assert "save_individual" in output
        assert "save_grid" in output

    def test_output_defaults(self):
        """Test output configuration defaults (V2)."""
        config = get_default_config()
        output = config["output"]

        assert isinstance(output, dict)
        assert "base_dir" in output
        assert "subdirs" in output

        subdirs = output["subdirs"]
        assert "logs" in subdirs
        assert "checkpoints" in subdirs
        assert "samples" in subdirs
        assert "generated" in subdirs

    def test_image_size_consistency(self):
        """Test that image_size is only in model.architecture (V2)."""
        config = get_default_config()
        # V2: image_size only in model.architecture, derived for data
        assert "image_size" in config["model"]["architecture"]
        assert "image_size" not in config["data"]
        assert "image_size" not in config["data"].get("loading", {})


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
        """Test validation fails with invalid image_size (V2)."""
        config = get_default_config()

        # Test negative value
        config["model"]["architecture"]["image_size"] = -1
        with pytest.raises(ValueError, match="image_size must be a positive integer"):
            validate_config(config)

        # Test zero
        config = get_default_config()
        config["model"]["architecture"]["image_size"] = 0
        with pytest.raises(ValueError, match="image_size must be a positive integer"):
            validate_config(config)

    def test_invalid_in_channels(self):
        """Test validation fails with invalid in_channels (V2)."""
        config = get_default_config()
        config["model"]["architecture"]["in_channels"] = 0

        with pytest.raises(ValueError, match="in_channels must be a positive integer"):
            validate_config(config)

    def test_invalid_model_channels(self):
        """Test validation fails with invalid model_channels (V2)."""
        config = get_default_config()
        config["model"]["architecture"]["model_channels"] = -1

        with pytest.raises(
            ValueError, match="model_channels must be a positive integer"
        ):
            validate_config(config)

    def test_invalid_channel_multipliers_type(self):
        """Test validation fails with non-list channel_multipliers (V2)."""
        config = get_default_config()
        config["model"]["architecture"]["channel_multipliers"] = (
            1,
            2,
            4,
        )  # Tuple instead of list

        with pytest.raises(ValueError, match="channel_multipliers must be a list"):
            validate_config(config)

    def test_invalid_channel_multipliers_values(self):
        """Test validation fails with invalid channel_multiplier values (V2)."""
        config = get_default_config()
        config["model"]["architecture"]["channel_multipliers"] = [1, 2, -1]

        with pytest.raises(
            ValueError, match="All channel_multipliers must be positive integers"
        ):
            validate_config(config)

    def test_invalid_num_classes(self):
        """Test validation fails with invalid num_classes (V2)."""
        config = get_default_config()

        # Test negative value (when not None)
        config["model"]["conditioning"]["num_classes"] = -1
        with pytest.raises(
            ValueError, match="num_classes must be a positive integer or None"
        ):
            validate_config(config)

        # Test zero
        config = get_default_config()
        config["model"]["conditioning"]["num_classes"] = 0
        with pytest.raises(
            ValueError, match="num_classes must be a positive integer or None"
        ):
            validate_config(config)

    def test_valid_num_classes_none(self):
        """Test validation succeeds with num_classes=None (V2)."""
        config = get_default_config()
        config["model"]["conditioning"]["num_classes"] = None

        # Should not raise
        validate_config(config)

    def test_invalid_num_timesteps(self):
        """Test validation fails with invalid num_timesteps (V2)."""
        config = get_default_config()
        config["model"]["diffusion"]["num_timesteps"] = 0

        with pytest.raises(
            ValueError, match="num_timesteps must be a positive integer"
        ):
            validate_config(config)

    def test_invalid_beta_schedule(self):
        """Test validation fails with invalid beta_schedule (V2)."""
        config = get_default_config()
        config["model"]["diffusion"]["beta_schedule"] = "invalid"

        with pytest.raises(ValueError, match="Invalid beta_schedule"):
            validate_config(config)

    def test_valid_beta_schedules(self):
        """Test validation succeeds with all valid beta schedules (V2)."""
        config = get_default_config()
        valid_schedules = ["linear", "cosine", "quadratic", "sigmoid"]

        for schedule in valid_schedules:
            config["model"]["diffusion"]["beta_schedule"] = schedule
            validate_config(config)  # Should not raise

    def test_invalid_beta_start(self):
        """Test validation fails with invalid beta_start (V2)."""
        config = get_default_config()

        # Test negative
        config["model"]["diffusion"]["beta_start"] = -0.1
        with pytest.raises(
            ValueError, match="beta_start must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test >= 1
        config = get_default_config()
        config["model"]["diffusion"]["beta_start"] = 1.0
        with pytest.raises(
            ValueError, match="beta_start must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_invalid_beta_end(self):
        """Test validation fails with invalid beta_end (V2)."""
        config = get_default_config()

        # Test negative
        config["model"]["diffusion"]["beta_end"] = -0.1
        with pytest.raises(
            ValueError, match="beta_end must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test >= 1
        config = get_default_config()
        config["model"]["diffusion"]["beta_end"] = 1.5
        with pytest.raises(
            ValueError, match="beta_end must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_beta_start_greater_than_beta_end(self):
        """Test validation fails when beta_start >= beta_end (V2)."""
        config = get_default_config()
        config["model"]["diffusion"]["beta_start"] = 0.02
        config["model"]["diffusion"]["beta_end"] = 0.01

        with pytest.raises(ValueError, match="beta_start must be less than beta_end"):
            validate_config(config)

    def test_invalid_class_dropout_prob(self):
        """Test validation fails with invalid class_dropout_prob (V2)."""
        config = get_default_config()

        # Test negative
        config["model"]["conditioning"]["class_dropout_prob"] = -0.1
        with pytest.raises(
            ValueError, match="class_dropout_prob must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test > 1
        config = get_default_config()
        config["model"]["conditioning"]["class_dropout_prob"] = 1.5
        with pytest.raises(
            ValueError, match="class_dropout_prob must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_invalid_use_attention_type(self):
        """Test validation fails with non-list use_attention (V2)."""
        config = get_default_config()
        config["model"]["architecture"]["use_attention"] = (
            False,
            False,
            True,
        )  # Tuple instead of list

        with pytest.raises(ValueError, match="use_attention must be a list"):
            validate_config(config)

    def test_invalid_use_attention_values(self):
        """Test validation fails with non-boolean use_attention values (V2)."""
        config = get_default_config()
        config["model"]["architecture"]["use_attention"] = [False, 1, True]

        with pytest.raises(
            ValueError, match="All use_attention values must be booleans"
        ):
            validate_config(config)

    def test_use_attention_length_mismatch(self):
        """Test validation fails when use_attention length doesn't match channel_multipliers (V2)."""
        config = get_default_config()
        config["model"]["architecture"]["channel_multipliers"] = [1, 2, 4]
        config["model"]["architecture"]["use_attention"] = [False, True]  # Wrong length

        with pytest.raises(
            ValueError,
            match="use_attention length must match channel_multipliers length",
        ):
            validate_config(config)

    def test_invalid_batch_size(self):
        """Test validation fails with invalid batch_size (V2)."""
        config = get_default_config()

        # Test zero
        config["data"]["loading"]["batch_size"] = 0
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_config(config)

        # Test negative
        config = get_default_config()
        config["data"]["loading"]["batch_size"] = -1
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_config(config)

    def test_invalid_num_workers(self):
        """Test validation fails with negative num_workers (V2)."""
        config = get_default_config()
        config["data"]["loading"]["num_workers"] = -1

        with pytest.raises(
            ValueError, match="num_workers must be a non-negative integer"
        ):
            validate_config(config)

    def test_invalid_rotation_degrees(self):
        """Test validation fails with negative rotation_degrees (V2)."""
        config = get_default_config()
        config["data"]["augmentation"]["rotation_degrees"] = -10

        with pytest.raises(
            ValueError, match="rotation_degrees must be a non-negative number"
        ):
            validate_config(config)

    def test_image_size_mismatch(self):
        """Test NOT APPLICABLE for V2 - image_size derived from model."""
        # In V2, image_size is only in model.architecture and derived for data
        # This test is no longer relevant
        pass

    def test_conditional_requires_labels(self):
        """Test validation for conditional generation (V2)."""
        config = get_default_config()
        # In V2, return_labels is derived from conditioning.type
        # Set conditioning type to "class" requires num_classes
        config["model"]["conditioning"]["type"] = "class"
        config["model"]["conditioning"]["num_classes"] = None  # Invalid

        with pytest.raises(
            ValueError,
            match="num_classes must be set",
        ):
            validate_config(config)

    def test_valid_conditional_config(self):
        """Test validation succeeds with proper conditional configuration (V2)."""
        config = get_default_config()
        config["model"]["conditioning"]["type"] = "class"
        config["model"]["conditioning"]["num_classes"] = 2

        # Should not raise
        validate_config(config)

    def test_invalid_epochs(self):
        """Test validation fails with invalid epochs."""
        config = get_default_config()
        config["training"]["epochs"] = 0

        with pytest.raises(ValueError, match="epochs must be a positive integer"):
            validate_config(config)

    def test_invalid_learning_rate(self):
        """Test validation fails with invalid learning_rate (V2)."""
        config = get_default_config()

        # Test zero
        config["training"]["optimizer"]["learning_rate"] = 0
        with pytest.raises(ValueError, match="learning_rate must be a positive number"):
            validate_config(config)

        # Test negative
        config = get_default_config()
        config["training"]["optimizer"]["learning_rate"] = -0.001
        with pytest.raises(ValueError, match="learning_rate must be a positive number"):
            validate_config(config)

    def test_invalid_optimizer(self):
        """Test validation fails with invalid optimizer (V2)."""
        config = get_default_config()
        config["training"]["optimizer"]["type"] = "sgd"

        with pytest.raises(ValueError, match="Invalid optimizer"):
            validate_config(config)

    def test_valid_optimizers(self):
        """Test validation succeeds with all valid optimizers (V2)."""
        config = get_default_config()
        valid_optimizers = ["adam", "adamw"]

        for optimizer in valid_optimizers:
            config["training"]["optimizer"]["type"] = optimizer
            validate_config(config)  # Should not raise

    def test_invalid_scheduler(self):
        """Test validation fails with invalid scheduler (V2)."""
        config = get_default_config()
        config["training"]["scheduler"]["type"] = "invalid"

        with pytest.raises(ValueError, match="Invalid scheduler"):
            validate_config(config)

    def test_valid_schedulers(self):
        """Test validation succeeds with all valid schedulers (V2)."""
        config = get_default_config()
        valid_schedulers = ["cosine", "step", "plateau", None]

        for scheduler in valid_schedulers:
            config["training"]["scheduler"]["type"] = scheduler
            validate_config(config)  # Should not raise

    def test_invalid_device(self):
        """Test validation fails with invalid device (V2)."""
        config = get_default_config()
        config["compute"]["device"] = "tpu"  # Device is now in compute section

        with pytest.raises(ValueError, match="Invalid device"):
            validate_config(config)

    def test_invalid_use_ema(self):
        """Test validation fails with non-boolean use_ema (V2)."""
        config = get_default_config()
        config["training"]["ema"]["enabled"] = "true"

        with pytest.raises(ValueError, match="enabled must be a boolean"):
            validate_config(config)

    def test_invalid_ema_decay(self):
        """Test validation fails with invalid ema_decay (V2)."""
        config = get_default_config()

        # Test negative
        config["training"]["ema"]["decay"] = -0.1
        with pytest.raises(ValueError, match="decay must be a number between 0 and 1"):
            validate_config(config)

        # Test >= 1
        config = get_default_config()
        config["training"]["ema"]["decay"] = 1.0
        with pytest.raises(ValueError, match="decay must be a number between 0 and 1"):
            validate_config(config)

    def test_invalid_use_amp(self):
        """Test validation fails with non-boolean use_amp (V2)."""
        config = get_default_config()
        config["training"]["performance"]["use_amp"] = "false"

        with pytest.raises(ValueError, match="use_amp must be a boolean"):
            validate_config(config)

    def test_invalid_gradient_clip_norm(self):
        """Test validation fails with invalid gradient_clip_norm (V2)."""
        config = get_default_config()
        config["training"]["optimizer"]["gradient_clip_norm"] = -1.0

        with pytest.raises(
            ValueError, match="gradient_clip_norm must be a positive number or None"
        ):
            validate_config(config)

    def test_valid_gradient_clip_norm_none(self):
        """Test validation succeeds with gradient_clip_norm=None (V2)."""
        config = get_default_config()
        config["training"]["optimizer"]["gradient_clip_norm"] = None

        # Should not raise
        validate_config(config)

    def test_invalid_sample_images(self):
        """Test validation fails with non-boolean enabled in visualization (V2)."""
        config = get_default_config()
        config["training"]["visualization"]["enabled"] = "true"

        with pytest.raises(ValueError, match="enabled must be a boolean"):
            validate_config(config)

    def test_invalid_sample_interval(self):
        """Test validation fails with invalid log_images_interval in visualization (V2)."""
        config = get_default_config()
        config["training"]["visualization"]["log_images_interval"] = 0

        with pytest.raises(
            ValueError, match="interval must be.*positive integer or null"
        ):
            validate_config(config)

    def test_invalid_samples_per_class(self):
        """Test validation fails with invalid num_samples (V2)."""
        config = get_default_config()
        config["training"]["visualization"]["num_samples"] = -1

        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            validate_config(config)

    def test_invalid_guidance_scale(self):
        """Test validation fails with invalid guidance_scale."""
        config = get_default_config()
        config["training"]["visualization"]["guidance_scale"] = 0.5

        with pytest.raises(
            ValueError, match="training.visualization.guidance_scale must be >= 1.0"
        ):
            validate_config(config)

    def test_missing_output_dirs(self):
        """Test validation for required output config (V2)."""
        config = get_default_config()

        # Test missing output section
        del config["output"]
        with pytest.raises(KeyError, match="Missing required config key: output"):
            validate_config(config)

        # Reset and test missing subdirs
        config = get_default_config()
        del config["output"]["subdirs"]
        with pytest.raises(KeyError, match="output.subdirs"):
            validate_config(config)


@pytest.mark.unit
class TestModeAwareValidation:
    """Test mode-aware configuration validation."""

    def test_train_mode_requires_checkpoint_dir(self):
        """Test that train mode requires training.checkpointing section (V2)."""
        config = get_default_config()
        config["mode"] = "train"
        del config["training"]["checkpointing"]

        with pytest.raises(KeyError, match="training.checkpointing"):
            validate_config(config)

    def test_generate_mode_requires_checkpoint(self):
        """Test that generate mode requires generation.checkpoint."""
        config = get_default_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = None

        with pytest.raises(ValueError, match="generation.checkpoint is required"):
            validate_config(config)

    def test_generate_mode_with_valid_checkpoint(self):
        """Test that generate mode validates with checkpoint set."""
        config = get_default_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"

        # Should not raise
        validate_config(config)

    def test_invalid_mode(self):
        """Test validation fails with invalid mode."""
        config = get_default_config()
        config["mode"] = "invalid"

        with pytest.raises(ValueError, match="Invalid mode"):
            validate_config(config)

    def test_training_validation_nested(self):
        """Test that training.validation section is properly validated."""
        config = get_default_config()
        config["training"]["validation"]["frequency"] = -1

        with pytest.raises(ValueError, match="training.validation.frequency"):
            validate_config(config)

    def test_training_visualization_nested(self):
        """Test that training.visualization section is properly validated (V2)."""
        config = get_default_config()
        config["training"]["visualization"]["log_images_interval"] = 0

        with pytest.raises(
            ValueError, match="interval must be.*positive integer or null"
        ):
            validate_config(config)

    def test_generation_use_ema(self):
        """Test that generation.sampling.use_ema is validated (V2)."""
        config = get_default_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["use_ema"] = "true"  # Should be boolean

        with pytest.raises(ValueError, match="use_ema must be a boolean"):
            validate_config(config)

    def test_generation_num_samples(self):
        """Test that generation.sampling.num_samples is validated (V2)."""
        config = get_default_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["num_samples"] = 0

        with pytest.raises(ValueError, match="num_samples must be a positive"):
            validate_config(config)

    def test_generation_ema_decay_valid(self):
        """Test that generation.sampling.ema_decay passes validation with valid value."""
        config = get_default_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["ema_decay"] = 0.9999

        # Should not raise
        validate_config(config)

    def test_generation_ema_decay_invalid(self):
        """Test that generation.sampling.ema_decay fails with out-of-range value."""
        config = get_default_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["ema_decay"] = 1.5

        with pytest.raises(
            ValueError, match="ema_decay must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_generation_ema_decay_default(self):
        """Test that default config has ema_decay in generation.sampling."""
        config = get_default_config()
        assert "ema_decay" in config["generation"]["sampling"]
        assert config["generation"]["sampling"]["ema_decay"] == 0.9999

    def test_generation_batch_size_valid(self):
        """Test that generation.sampling.batch_size passes validation with valid value."""
        config = get_default_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["batch_size"] = 50

        # Should not raise
        validate_config(config)

    def test_generation_batch_size_invalid(self):
        """Test that generation.sampling.batch_size fails with negative value."""
        config = get_default_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["batch_size"] = -1

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_config(config)

    def test_generation_batch_size_default(self):
        """Test that default config has batch_size in generation.sampling."""
        config = get_default_config()
        assert "batch_size" in config["generation"]["sampling"]
        assert config["generation"]["sampling"]["batch_size"] == 50


@pytest.mark.unit
class TestGetResolutionConfig:
    """Test resolution-specific configuration overrides."""

    def test_resolution_40_config(self):
        """Test 40x40 resolution configuration (V2)."""
        config = get_resolution_config(40)

        assert isinstance(config, dict)
        assert "model" in config
        assert "data" in config
        assert config["model"]["architecture"]["image_size"] == 40
        assert config["data"]["loading"]["batch_size"] == 32
        assert config["model"]["architecture"]["model_channels"] == 64

    def test_resolution_64_config(self):
        """Test 64x64 resolution configuration (V2)."""
        config = get_resolution_config(64)

        assert isinstance(config, dict)
        assert config["model"]["architecture"]["image_size"] == 64
        assert config["data"]["loading"]["batch_size"] == 64
        assert config["model"]["architecture"]["model_channels"] == 128

    def test_resolution_128_config(self):
        """Test 128x128 resolution configuration (V2)."""
        config = get_resolution_config(128)

        assert isinstance(config, dict)
        assert config["model"]["architecture"]["image_size"] == 128
        assert config["data"]["loading"]["batch_size"] == 32

    def test_resolution_256_config(self):
        """Test 256x256 resolution configuration (V2)."""
        config = get_resolution_config(256)

        assert isinstance(config, dict)
        assert config["model"]["architecture"]["image_size"] == 256
        assert config["data"]["loading"]["batch_size"] == 16

    def test_invalid_resolution(self):
        """Test error with invalid resolution."""
        with pytest.raises(ValueError, match="Unsupported image_size"):
            get_resolution_config(100)

    def test_all_supported_resolutions(self):
        """Test that all supported resolutions have configurations (V2)."""
        supported_resolutions = [40, 64, 128, 256]

        for resolution in supported_resolutions:
            config = get_resolution_config(resolution)
            assert isinstance(config, dict)
            assert config["model"]["architecture"]["image_size"] == resolution

    def test_channel_multipliers_length(self):
        """Test that use_attention matches channel_multipliers length (V2)."""
        supported_resolutions = [40, 64, 128, 256]

        for resolution in supported_resolutions:
            config = get_resolution_config(resolution)
            multipliers = config["model"]["architecture"]["channel_multipliers"]
            attention = config["model"]["architecture"]["use_attention"]
            assert len(attention) == len(multipliers)


# ============================================================================
# Component Tests - With File I/O
# ============================================================================


@pytest.mark.unit
class TestSaveLatestValidation:
    """Unit tests for training.checkpointing.save_latest validation."""

    def test_save_latest_valid_true(self):
        """save_latest: true is accepted."""
        config = get_default_config()
        config["training"]["checkpointing"]["save_latest"] = True
        validate_config(config)  # should not raise

    def test_save_latest_valid_false(self):
        """save_latest: false is accepted."""
        config = get_default_config()
        config["training"]["checkpointing"]["save_latest"] = False
        validate_config(config)  # should not raise

    def test_save_latest_missing_defaults_to_true(self):
        """save_latest absent from config is accepted (defaults to True)."""
        config = get_default_config()
        config["training"]["checkpointing"].pop("save_latest", None)
        validate_config(config)  # should not raise

    def test_save_latest_invalid_type(self):
        """save_latest with non-bool raises ValueError."""
        config = get_default_config()
        config["training"]["checkpointing"]["save_latest"] = "yes"
        with pytest.raises(ValueError, match="save_latest must be a boolean"):
            validate_config(config)


@pytest.mark.component
class TestConfigFileValidation:
    """Test validation of actual config files."""

    def test_default_config_file(self):
        """Test that default.yaml is valid."""
        config_path = Path("src/experiments/diffusion/default.yaml")

        if not config_path.exists():
            pytest.skip("default.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should not raise
        validate_config(config)

    def test_default_config_structure(self):
        """Test default config has expected structure (V2)."""
        config_path = Path("src/experiments/diffusion/default.yaml")

        if not config_path.exists():
            pytest.skip("default.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check experiment type
        assert config["experiment"] == "diffusion"

        # Check model structure (V2)
        assert config["model"]["architecture"]["image_size"] == 40
        assert config["model"]["architecture"]["in_channels"] == 3
        assert config["model"]["diffusion"]["beta_schedule"] == "cosine"

        # Check data configuration (V2 - no image_size here)
        assert "paths" in config["data"]
        assert "loading" in config["data"]

        # Check training configuration (V2)
        assert config["training"]["ema"]["enabled"] is True
        assert isinstance(config["training"]["epochs"], int)
        assert "validation" in config["training"]
        assert "visualization" in config["training"]

        # Check generation configuration (V2)
        assert "generation" in config
        assert "checkpoint" in config["generation"]
        assert "sampling" in config["generation"]
        assert "use_ema" in config["generation"]["sampling"]
