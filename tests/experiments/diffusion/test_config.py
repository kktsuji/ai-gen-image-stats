"""Tests for Diffusion Configuration

This module tests the diffusion model configuration management, including:
- Configuration validation
- Resolution-specific configuration overrides
"""

from pathlib import Path

import pytest
import yaml

from src.experiments.diffusion.config import (
    get_resolution_config,
    validate_config,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ============================================================================
# Unit Tests - Fast, Pure Logic
# ============================================================================


def _make_valid_config():
    """Return a valid diffusion configuration dict for testing.

    Loads from configs/diffusion-example.yaml to stay in sync with the
    canonical example config (single source of truth).
    """
    import copy

    config_path = _PROJECT_ROOT / "configs/diffusion-example.yaml"
    with open(config_path) as f:
        return copy.deepcopy(yaml.safe_load(f))


@pytest.mark.unit
class TestValidateConfig:
    """Test configuration validation."""

    def test_valid_example_config(self):
        """Test that example config passes validation."""
        config = _make_valid_config()
        # Should not raise any exception
        validate_config(config)

    def test_missing_top_level_key(self):
        """Test validation fails with missing top-level key."""
        config = _make_valid_config()
        del config["model"]

        with pytest.raises(KeyError, match="Missing required config key: model"):
            validate_config(config)

    def test_invalid_experiment_type(self):
        """Test validation fails with wrong experiment type."""
        config = _make_valid_config()
        config["experiment"] = "invalid"

        with pytest.raises(ValueError, match="Invalid experiment type"):
            validate_config(config)

    def test_invalid_image_size(self):
        """Test validation fails with invalid image_size (V2)."""
        config = _make_valid_config()

        # Test negative value
        config["model"]["architecture"]["image_size"] = -1
        with pytest.raises(ValueError, match="image_size must be a positive integer"):
            validate_config(config)

        # Test zero
        config = _make_valid_config()
        config["model"]["architecture"]["image_size"] = 0
        with pytest.raises(ValueError, match="image_size must be a positive integer"):
            validate_config(config)

    def test_invalid_in_channels(self):
        """Test validation fails with invalid in_channels (V2)."""
        config = _make_valid_config()
        config["model"]["architecture"]["in_channels"] = 0

        with pytest.raises(ValueError, match="in_channels must be a positive integer"):
            validate_config(config)

    def test_invalid_model_channels(self):
        """Test validation fails with invalid model_channels (V2)."""
        config = _make_valid_config()
        config["model"]["architecture"]["model_channels"] = -1

        with pytest.raises(
            ValueError, match="model_channels must be a positive integer"
        ):
            validate_config(config)

    def test_invalid_channel_multipliers_type(self):
        """Test validation fails with non-list channel_multipliers (V2)."""
        config = _make_valid_config()
        config["model"]["architecture"]["channel_multipliers"] = (
            1,
            2,
            4,
        )  # Tuple instead of list

        with pytest.raises(ValueError, match="channel_multipliers must be a list"):
            validate_config(config)

    def test_invalid_channel_multipliers_values(self):
        """Test validation fails with invalid channel_multiplier values (V2)."""
        config = _make_valid_config()
        config["model"]["architecture"]["channel_multipliers"] = [1, 2, -1]

        with pytest.raises(
            ValueError, match="All channel_multipliers must be positive integers"
        ):
            validate_config(config)

    def test_invalid_channel_multipliers_first_not_one(self):
        """Test validation fails when channel_multipliers[0] != 1."""
        config = _make_valid_config()
        config["model"]["architecture"]["channel_multipliers"] = [2, 4, 8]
        config["model"]["architecture"]["use_attention"] = [False, False, False]

        with pytest.raises(ValueError, match=r"channel_multipliers\[0\] must be 1"):
            validate_config(config)

    def test_valid_channel_multipliers_first_is_one(self):
        """Test validation passes when channel_multipliers[0] == 1."""
        config = _make_valid_config()
        config["model"]["architecture"]["channel_multipliers"] = [1, 2, 4]
        config["model"]["architecture"]["use_attention"] = [False, False, False]

        # Should not raise
        validate_config(config)

    def test_invalid_num_classes(self):
        """Test validation fails with invalid num_classes (V2)."""
        config = _make_valid_config()

        # Test negative value (when not None)
        config["model"]["conditioning"]["num_classes"] = -1
        with pytest.raises(
            ValueError, match="num_classes must be a positive integer or None"
        ):
            validate_config(config)

        # Test zero
        config = _make_valid_config()
        config["model"]["conditioning"]["num_classes"] = 0
        with pytest.raises(
            ValueError, match="num_classes must be a positive integer or None"
        ):
            validate_config(config)

    def test_valid_num_classes_none(self):
        """Test validation succeeds with num_classes=None (V2)."""
        config = _make_valid_config()
        config["model"]["conditioning"]["type"] = None
        config["model"]["conditioning"]["num_classes"] = None

        # Should not raise
        validate_config(config)

    def test_invalid_num_timesteps(self):
        """Test validation fails with invalid num_timesteps (V2)."""
        config = _make_valid_config()
        config["model"]["diffusion"]["num_timesteps"] = 0

        with pytest.raises(
            ValueError, match="num_timesteps must be a positive integer"
        ):
            validate_config(config)

    def test_invalid_beta_schedule(self):
        """Test validation fails with invalid beta_schedule (V2)."""
        config = _make_valid_config()
        config["model"]["diffusion"]["beta_schedule"] = "invalid"

        with pytest.raises(ValueError, match="Invalid beta_schedule"):
            validate_config(config)

    def test_valid_beta_schedules(self):
        """Test validation succeeds with all valid beta schedules (V2)."""
        config = _make_valid_config()
        valid_schedules = ["linear", "cosine", "quadratic", "sigmoid"]

        for schedule in valid_schedules:
            config["model"]["diffusion"]["beta_schedule"] = schedule
            validate_config(config)  # Should not raise

    def test_invalid_beta_start(self):
        """Test validation fails with invalid beta_start (V2)."""
        config = _make_valid_config()

        # Test negative
        config["model"]["diffusion"]["beta_start"] = -0.1
        with pytest.raises(
            ValueError, match="beta_start must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test >= 1
        config = _make_valid_config()
        config["model"]["diffusion"]["beta_start"] = 1.0
        with pytest.raises(
            ValueError, match="beta_start must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_invalid_beta_end(self):
        """Test validation fails with invalid beta_end (V2)."""
        config = _make_valid_config()

        # Test negative
        config["model"]["diffusion"]["beta_end"] = -0.1
        with pytest.raises(
            ValueError, match="beta_end must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test >= 1
        config = _make_valid_config()
        config["model"]["diffusion"]["beta_end"] = 1.5
        with pytest.raises(
            ValueError, match="beta_end must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_beta_start_greater_than_beta_end(self):
        """Test validation fails when beta_start >= beta_end (V2)."""
        config = _make_valid_config()
        config["model"]["diffusion"]["beta_start"] = 0.02
        config["model"]["diffusion"]["beta_end"] = 0.01

        with pytest.raises(ValueError, match="beta_start must be less than beta_end"):
            validate_config(config)

    def test_invalid_class_dropout_prob(self):
        """Test validation fails with invalid class_dropout_prob (V2)."""
        config = _make_valid_config()

        # Test negative
        config["model"]["conditioning"]["class_dropout_prob"] = -0.1
        with pytest.raises(
            ValueError, match="class_dropout_prob must be a number between 0 and 1"
        ):
            validate_config(config)

        # Test > 1
        config = _make_valid_config()
        config["model"]["conditioning"]["class_dropout_prob"] = 1.5
        with pytest.raises(
            ValueError, match="class_dropout_prob must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_invalid_use_attention_type(self):
        """Test validation fails with non-list use_attention (V2)."""
        config = _make_valid_config()
        config["model"]["architecture"]["use_attention"] = (
            False,
            False,
            True,
        )  # Tuple instead of list

        with pytest.raises(ValueError, match="use_attention must be a list"):
            validate_config(config)

    def test_invalid_use_attention_values(self):
        """Test validation fails with non-boolean use_attention values (V2)."""
        config = _make_valid_config()
        config["model"]["architecture"]["use_attention"] = [False, 1, True]

        with pytest.raises(
            ValueError, match="All use_attention values must be booleans"
        ):
            validate_config(config)

    def test_use_attention_length_mismatch(self):
        """Test validation fails when use_attention length doesn't match channel_multipliers (V2)."""
        config = _make_valid_config()
        config["model"]["architecture"]["channel_multipliers"] = [1, 2, 4]
        config["model"]["architecture"]["use_attention"] = [False, True]  # Wrong length

        with pytest.raises(
            ValueError,
            match="use_attention length must match channel_multipliers length",
        ):
            validate_config(config)

    def test_invalid_batch_size(self):
        """Test validation fails with invalid batch_size (V2)."""
        config = _make_valid_config()

        # Test zero
        config["data"]["loading"]["batch_size"] = 0
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_config(config)

        # Test negative
        config = _make_valid_config()
        config["data"]["loading"]["batch_size"] = -1
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_config(config)

    def test_invalid_num_workers(self):
        """Test validation fails with negative num_workers (V2)."""
        config = _make_valid_config()
        config["data"]["loading"]["num_workers"] = -1

        with pytest.raises(
            ValueError, match="num_workers must be a non-negative integer"
        ):
            validate_config(config)

    def test_invalid_rotation_degrees(self):
        """Test validation fails with negative rotation_degrees (V2)."""
        config = _make_valid_config()
        config["data"]["augmentation"]["rotation_degrees"] = -10

        with pytest.raises(
            ValueError, match="rotation_degrees must be a non-negative number"
        ):
            validate_config(config)

    def test_conditional_requires_labels(self):
        """Test validation for conditional generation (V2)."""
        config = _make_valid_config()
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
        config = _make_valid_config()
        config["model"]["conditioning"]["type"] = "class"
        config["model"]["conditioning"]["num_classes"] = 2

        # Should not raise
        validate_config(config)

    def test_invalid_epochs(self):
        """Test validation fails with invalid epochs."""
        config = _make_valid_config()
        config["training"]["epochs"] = 0

        with pytest.raises(ValueError, match="epochs must be a positive integer"):
            validate_config(config)

    def test_invalid_learning_rate(self):
        """Test validation fails with invalid learning_rate (V2)."""
        config = _make_valid_config()

        # Test zero
        config["training"]["optimizer"]["learning_rate"] = 0
        with pytest.raises(ValueError, match="learning_rate must be a positive number"):
            validate_config(config)

        # Test negative
        config = _make_valid_config()
        config["training"]["optimizer"]["learning_rate"] = -0.001
        with pytest.raises(ValueError, match="learning_rate must be a positive number"):
            validate_config(config)

    def test_invalid_optimizer(self):
        """Test validation fails with invalid optimizer (V2)."""
        config = _make_valid_config()
        config["training"]["optimizer"]["type"] = "sgd"

        with pytest.raises(ValueError, match="Invalid optimizer"):
            validate_config(config)

    def test_valid_optimizers(self):
        """Test validation succeeds with all valid optimizers (V2)."""
        config = _make_valid_config()
        valid_optimizers = ["adam", "adamw"]

        for optimizer in valid_optimizers:
            config["training"]["optimizer"]["type"] = optimizer
            validate_config(config)  # Should not raise

    def test_invalid_scheduler(self):
        """Test validation fails with invalid scheduler (V2)."""
        config = _make_valid_config()
        config["training"]["scheduler"]["type"] = "invalid"

        with pytest.raises(ValueError, match="Invalid scheduler"):
            validate_config(config)

    def test_valid_schedulers(self):
        """Test validation succeeds with all valid schedulers (V2)."""
        config = _make_valid_config()
        valid_schedulers = ["cosine", "step", "plateau", None]

        for scheduler in valid_schedulers:
            config["training"]["scheduler"]["type"] = scheduler
            validate_config(config)  # Should not raise

    def test_invalid_device(self):
        """Test validation fails with invalid device (V2)."""
        config = _make_valid_config()
        config["compute"]["device"] = "tpu"  # Device is now in compute section

        with pytest.raises(ValueError, match="Invalid device"):
            validate_config(config)

    def test_invalid_use_ema(self):
        """Test validation fails with non-boolean use_ema (V2)."""
        config = _make_valid_config()
        config["training"]["ema"]["enabled"] = "true"

        with pytest.raises(ValueError, match="enabled must be a boolean"):
            validate_config(config)

    def test_invalid_ema_decay(self):
        """Test validation fails with invalid ema_decay (V2)."""
        config = _make_valid_config()

        # Test negative
        config["training"]["ema"]["decay"] = -0.1
        with pytest.raises(ValueError, match="decay must be a number between 0 and 1"):
            validate_config(config)

        # Test >= 1
        config = _make_valid_config()
        config["training"]["ema"]["decay"] = 1.0
        with pytest.raises(ValueError, match="decay must be a number between 0 and 1"):
            validate_config(config)

    def test_invalid_use_amp(self):
        """Test validation fails with non-boolean use_amp (V2)."""
        config = _make_valid_config()
        config["training"]["performance"]["use_amp"] = "false"

        with pytest.raises(ValueError, match="use_amp must be a boolean"):
            validate_config(config)

    def test_invalid_gradient_clip_norm(self):
        """Test validation fails with invalid gradient_clip_norm (V2)."""
        config = _make_valid_config()
        config["training"]["optimizer"]["gradient_clip_norm"] = -1.0

        with pytest.raises(
            ValueError, match="gradient_clip_norm must be a positive number or None"
        ):
            validate_config(config)

    def test_valid_gradient_clip_norm_none(self):
        """Test validation succeeds with gradient_clip_norm=None (V2)."""
        config = _make_valid_config()
        config["training"]["optimizer"]["gradient_clip_norm"] = None

        # Should not raise
        validate_config(config)

    def test_invalid_sample_images(self):
        """Test validation fails with non-boolean enabled in visualization (V2)."""
        config = _make_valid_config()
        config["training"]["visualization"]["enabled"] = "true"

        with pytest.raises(ValueError, match="enabled must be a boolean"):
            validate_config(config)

    def test_invalid_sample_interval(self):
        """Test validation fails with invalid log_images_interval in visualization (V2)."""
        config = _make_valid_config()
        config["training"]["visualization"]["log_images_interval"] = 0

        with pytest.raises(
            ValueError, match="interval must be.*positive integer or null"
        ):
            validate_config(config)

    def test_invalid_samples_per_class(self):
        """Test validation fails with invalid num_samples (V2)."""
        config = _make_valid_config()
        config["training"]["visualization"]["num_samples"] = -1

        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            validate_config(config)

    def test_invalid_guidance_scale(self):
        """Test validation fails with invalid guidance_scale."""
        config = _make_valid_config()
        config["training"]["visualization"]["guidance_scale"] = -1.0

        with pytest.raises(
            ValueError, match="training.visualization.guidance_scale must be >= 0.0"
        ):
            validate_config(config)

    def test_valid_guidance_scale_zero(self):
        """Test that guidance_scale: 0.0 (disable guidance) is accepted."""
        config = _make_valid_config()
        config["training"]["visualization"]["guidance_scale"] = 0.0
        # Should not raise
        validate_config(config)

    def test_valid_guidance_scale_less_than_one(self):
        """Test that guidance_scale: 0.5 (weak guidance) is accepted."""
        config = _make_valid_config()
        config["training"]["visualization"]["guidance_scale"] = 0.5
        # Should not raise
        validate_config(config)

    def test_invalid_generation_guidance_scale(self):
        """Test validation fails with invalid generation.sampling.guidance_scale."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["guidance_scale"] = -1.0

        with pytest.raises(
            ValueError, match="generation.sampling.guidance_scale must be >= 0.0"
        ):
            validate_config(config)

    def test_missing_output_dirs(self):
        """Test validation for required output config (V2)."""
        config = _make_valid_config()

        # Test missing output section
        del config["output"]
        with pytest.raises(KeyError, match="Missing required config key: output"):
            validate_config(config)

        # Reset and test missing subdirs
        config = _make_valid_config()
        del config["output"]["subdirs"]
        with pytest.raises(KeyError, match="output.subdirs"):
            validate_config(config)


@pytest.mark.unit
class TestModeAwareValidation:
    """Test mode-aware configuration validation."""

    def test_train_mode_requires_checkpoint_dir(self):
        """Test that train mode requires training.checkpointing section (V2)."""
        config = _make_valid_config()
        config["mode"] = "train"
        del config["training"]["checkpointing"]

        with pytest.raises(KeyError, match="training.checkpointing"):
            validate_config(config)

    def test_generate_mode_requires_checkpoint(self):
        """Test that generate mode requires generation.checkpoint."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = None

        with pytest.raises(ValueError, match="generation.checkpoint is required"):
            validate_config(config)

    def test_generate_mode_with_valid_checkpoint(self):
        """Test that generate mode validates with checkpoint set."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"

        # Should not raise
        validate_config(config)

    def test_invalid_mode(self):
        """Test validation fails with invalid mode."""
        config = _make_valid_config()
        config["mode"] = "invalid"

        with pytest.raises(ValueError, match="Invalid mode"):
            validate_config(config)

    def test_training_validation_nested(self):
        """Test that training.validation section is properly validated."""
        config = _make_valid_config()
        config["training"]["validation"]["frequency"] = -1

        with pytest.raises(ValueError, match="training.validation.frequency"):
            validate_config(config)

    def test_training_visualization_nested(self):
        """Test that training.visualization section is properly validated (V2)."""
        config = _make_valid_config()
        config["training"]["visualization"]["log_images_interval"] = 0

        with pytest.raises(
            ValueError, match="interval must be.*positive integer or null"
        ):
            validate_config(config)

    def test_generation_use_ema(self):
        """Test that generation.sampling.use_ema is validated (V2)."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["use_ema"] = "true"  # Should be boolean

        with pytest.raises(ValueError, match="use_ema must be a boolean"):
            validate_config(config)

    def test_generation_num_samples(self):
        """Test that generation.sampling.num_samples is validated (V2)."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["num_samples"] = 0

        with pytest.raises(ValueError, match="num_samples must be a positive"):
            validate_config(config)

    def test_generation_ema_decay_valid(self):
        """Test that generation.sampling.ema_decay passes validation with valid value."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["ema_decay"] = 0.9999

        # Should not raise
        validate_config(config)

    def test_generation_ema_decay_invalid(self):
        """Test that generation.sampling.ema_decay fails with out-of-range value."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["ema_decay"] = 1.5

        with pytest.raises(
            ValueError, match="ema_decay must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_generation_ema_decay_present(self):
        """Test that example config has ema_decay in generation.sampling."""
        config = _make_valid_config()
        assert "ema_decay" in config["generation"]["sampling"]
        assert config["generation"]["sampling"]["ema_decay"] == 0.9999

    def test_generation_batch_size_valid(self):
        """Test that generation.sampling.batch_size passes validation with valid value."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["batch_size"] = 50

        # Should not raise
        validate_config(config)

    def test_generation_batch_size_invalid(self):
        """Test that generation.sampling.batch_size fails with negative value."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["batch_size"] = -1

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_config(config)

    def test_generation_batch_size_present(self):
        """Test that example config has batch_size in generation.sampling."""
        config = _make_valid_config()
        assert "batch_size" in config["generation"]["sampling"]
        assert config["generation"]["sampling"]["batch_size"] == 50

    def test_generation_class_selection_present(self):
        """Test that example config has class_selection as null."""
        config = _make_valid_config()
        assert config["generation"]["sampling"]["class_selection"] is None

    def test_generation_class_selection_null_passes(self):
        """Test that class_selection=null is accepted."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["class_selection"] = None
        validate_config(config)

    def test_generation_class_selection_valid_single(self):
        """Test that class_selection=[0] is accepted."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["class_selection"] = [0]
        validate_config(config)

    def test_generation_class_selection_valid_subset(self):
        """Test that class_selection=[0, 1] is accepted."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["class_selection"] = [0, 1]
        validate_config(config)

    def test_generation_class_selection_empty_list_raises(self):
        """Test that class_selection=[] raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["class_selection"] = []
        with pytest.raises(ValueError, match="must be a non-empty list or null"):
            validate_config(config)

    def test_generation_class_selection_non_integer_raises(self):
        """Test that class_selection=[0, 'a'] raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["class_selection"] = [0, "a"]
        with pytest.raises(ValueError, match="must contain non-negative integers"):
            validate_config(config)

    def test_generation_class_selection_negative_raises(self):
        """Test that class_selection=[-1] raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["class_selection"] = [-1]
        with pytest.raises(ValueError, match="must contain non-negative integers"):
            validate_config(config)

    def test_generation_class_selection_duplicates_raises(self):
        """Test that class_selection=[0, 0] raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["sampling"]["class_selection"] = [0, 0]
        with pytest.raises(
            ValueError, match="must not contain duplicate class indices"
        ):
            validate_config(config)

    def test_generation_class_selection_out_of_range_raises(self):
        """Test that class_selection=[99] with num_classes=2 raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["model"]["conditioning"]["num_classes"] = 2
        config["generation"]["sampling"]["class_selection"] = [99]
        with pytest.raises(ValueError, match="contains indices"):
            validate_config(config)


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
        config = _make_valid_config()
        config["training"]["checkpointing"]["save_latest"] = True
        validate_config(config)  # should not raise

    def test_save_latest_valid_false(self):
        """save_latest: false is accepted."""
        config = _make_valid_config()
        config["training"]["checkpointing"]["save_latest"] = False
        validate_config(config)  # should not raise

    def test_save_latest_missing_is_optional(self):
        """save_latest absent from config is accepted (optional field)."""
        config = _make_valid_config()
        config["training"]["checkpointing"].pop("save_latest", None)
        validate_config(config)  # should not raise

    def test_save_latest_invalid_type(self):
        """save_latest with non-bool raises ValueError."""
        config = _make_valid_config()
        config["training"]["checkpointing"]["save_latest"] = "yes"
        with pytest.raises(ValueError, match="save_latest must be a boolean"):
            validate_config(config)


@pytest.mark.component
class TestConfigFileValidation:
    """Test validation of actual config files."""

    def test_example_config_file(self):
        """Test that diffusion-example.yaml is valid."""
        config_path = _PROJECT_ROOT / "configs/diffusion-example.yaml"

        if not config_path.exists():
            pytest.skip("configs/diffusion-example.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Should not raise
        validate_config(config)

    def test_example_config_structure(self):
        """Test diffusion-example config has expected structure (V2)."""
        config_path = _PROJECT_ROOT / "configs/diffusion-example.yaml"

        if not config_path.exists():
            pytest.skip("configs/diffusion-example.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check experiment type
        assert config["experiment"] == "diffusion"

        # Check model structure (V2)
        assert config["model"]["architecture"]["image_size"] == 40
        assert config["model"]["architecture"]["in_channels"] == 3
        assert config["model"]["diffusion"]["beta_schedule"] == "cosine"

        # Check data configuration (V2 - no image_size here)
        assert "split_file" in config["data"]
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


@pytest.mark.unit
class TestValidateConfigErrorPaths:
    """Test untested validation error paths in diffusion validate_config."""

    def test_architecture_field_none(self):
        """architecture field set to None raises ValueError."""
        config = _make_valid_config()
        config["model"]["architecture"]["image_size"] = None
        with pytest.raises(
            ValueError, match="model.architecture.image_size cannot be None"
        ):
            validate_config(config)

    def test_diffusion_section_missing(self):
        """No model.diffusion raises KeyError."""
        config = _make_valid_config()
        del config["model"]["diffusion"]
        with pytest.raises(
            KeyError, match="Missing required config key: model.diffusion"
        ):
            validate_config(config)

    def test_diffusion_field_none(self):
        """diffusion field set to None raises ValueError."""
        config = _make_valid_config()
        config["model"]["diffusion"]["num_timesteps"] = None
        with pytest.raises(
            ValueError, match="model.diffusion.num_timesteps cannot be None"
        ):
            validate_config(config)

    def test_conditioning_section_missing(self):
        """No model.conditioning raises KeyError."""
        config = _make_valid_config()
        del config["model"]["conditioning"]
        with pytest.raises(
            KeyError, match="Missing required config key: model.conditioning"
        ):
            validate_config(config)

    def test_conditioning_field_missing(self):
        """Missing required conditioning field raises KeyError."""
        config = _make_valid_config()
        del config["model"]["conditioning"]["type"]
        with pytest.raises(
            KeyError, match="Missing required field: model.conditioning.type"
        ):
            validate_config(config)

    def test_invalid_conditioning_type(self):
        """conditioning.type = 'vae' raises ValueError."""
        config = _make_valid_config()
        config["model"]["conditioning"]["type"] = "vae"
        with pytest.raises(ValueError, match="Invalid conditioning.type"):
            validate_config(config)

    def test_data_section_missing(self):
        """No data key raises KeyError."""
        config = _make_valid_config()
        del config["data"]
        with pytest.raises(KeyError, match="Missing required config key: data"):
            validate_config(config)

    def test_split_file_empty(self):
        """split_file = '' raises ValueError."""
        config = _make_valid_config()
        config["data"]["split_file"] = ""
        with pytest.raises(
            ValueError, match="data.split_file must be a non-empty string"
        ):
            validate_config(config)

    def test_augmentation_missing(self):
        """No data.augmentation raises KeyError."""
        config = _make_valid_config()
        del config["data"]["augmentation"]
        with pytest.raises(
            KeyError, match="Missing required config key: data.augmentation"
        ):
            validate_config(config)

    def test_ema_enabled_missing_decay(self):
        """ema.enabled: true without decay raises ValueError."""
        config = _make_valid_config()
        config["training"]["ema"]["enabled"] = True
        del config["training"]["ema"]["decay"]
        with pytest.raises(
            ValueError, match="training.ema.decay is required when EMA is enabled"
        ):
            validate_config(config)

    def test_ema_decay_out_of_range_zero(self):
        """ema.decay: 0 raises ValueError."""
        config = _make_valid_config()
        config["training"]["ema"]["enabled"] = True
        config["training"]["ema"]["decay"] = 0
        with pytest.raises(
            ValueError, match="training.ema.decay must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_ema_decay_out_of_range_one(self):
        """ema.decay: 1.0 raises ValueError."""
        config = _make_valid_config()
        config["training"]["ema"]["enabled"] = True
        config["training"]["ema"]["decay"] = 1.0
        with pytest.raises(
            ValueError, match="training.ema.decay must be a number between 0 and 1"
        ):
            validate_config(config)

    def test_resume_enabled_without_checkpoint(self):
        """resume.enabled: true, checkpoint: null raises ValueError."""
        config = _make_valid_config()
        config["training"]["resume"] = {"enabled": True, "checkpoint": None}
        with pytest.raises(
            ValueError,
            match="training.resume.checkpoint is required when resume is enabled",
        ):
            validate_config(config)

    def test_performance_field_not_bool(self):
        """use_amp: 'yes' raises ValueError."""
        config = _make_valid_config()
        config["training"]["performance"]["use_amp"] = "yes"
        with pytest.raises(ValueError, match="use_amp must be a boolean"):
            validate_config(config)

    def test_visualization_invalid_interval(self):
        """log_images_interval: -1 raises ValueError."""
        config = _make_valid_config()
        config["training"]["visualization"]["log_images_interval"] = -1
        with pytest.raises(
            ValueError, match="interval must be.*positive integer or null"
        ):
            validate_config(config)

    def test_diffusion_field_missing(self):
        """Missing required diffusion field raises KeyError."""
        config = _make_valid_config()
        del config["model"]["diffusion"]["num_timesteps"]
        with pytest.raises(
            KeyError, match="Missing required field: model.diffusion.num_timesteps"
        ):
            validate_config(config)

    def test_architecture_field_missing(self):
        """Missing required architecture field raises KeyError."""
        config = _make_valid_config()
        del config["model"]["architecture"]["image_size"]
        with pytest.raises(
            KeyError, match="Missing required field: model.architecture.image_size"
        ):
            validate_config(config)

    def test_generation_output_grid_nrow_invalid(self):
        """generation.output.grid_nrow = 0 raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["output"]["grid_nrow"] = 0
        with pytest.raises(
            ValueError, match="generation.output.grid_nrow must be a positive integer"
        ):
            validate_config(config)

    def test_generation_output_save_individual_not_bool(self):
        """generation.output.save_individual = 'yes' raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["output"]["save_individual"] = "yes"
        with pytest.raises(
            ValueError, match="generation.output.save_individual must be a boolean"
        ):
            validate_config(config)

    def test_generation_output_save_grid_not_bool(self):
        """generation.output.save_grid = 'yes' raises ValueError."""
        config = _make_valid_config()
        config["mode"] = "generate"
        config["generation"]["checkpoint"] = "path/to/checkpoint.pth"
        config["generation"]["output"]["save_grid"] = "yes"
        with pytest.raises(
            ValueError, match="generation.output.save_grid must be a boolean"
        ):
            validate_config(config)


# ============================================================================
# Unit Tests - Balancing Config Validation
# ============================================================================


@pytest.mark.unit
class TestBalancingConfigValidation:
    """Test data.balancing configuration validation."""

    def test_valid_example_balancing_config(self):
        """Test that example config with balancing passes validation."""
        config = _make_valid_config()
        # Example config should have balancing section and pass validation
        assert "balancing" in config["data"]
        validate_config(config)

    def test_missing_balancing_key_is_ok(self):
        """Test backwards compatibility: missing balancing key is OK."""
        config = _make_valid_config()
        del config["data"]["balancing"]
        # Should not raise
        validate_config(config)

    def test_invalid_weighted_sampler_method(self):
        """Test that invalid method raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["method"] = "invalid_method"

        with pytest.raises(ValueError, match="method must be one of"):
            validate_config(config)

    def test_invalid_class_weights_method(self):
        """Test that invalid class_weights method raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["class_weights"]["method"] = "bad_method"

        with pytest.raises(ValueError, match="method must be one of"):
            validate_config(config)

    def test_manual_method_without_weights_raises(self):
        """Test that manual method without manual_weights raises error."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["method"] = "manual"
        config["data"]["balancing"]["weighted_sampler"]["manual_weights"] = None

        with pytest.raises(ValueError, match="manual_weights must be a non-empty"):
            validate_config(config)

    def test_manual_method_with_empty_weights_raises(self):
        """Test that manual method with empty weights raises error."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["method"] = "manual"
        config["data"]["balancing"]["weighted_sampler"]["manual_weights"] = []

        with pytest.raises(ValueError, match="manual_weights must be a non-empty"):
            validate_config(config)

    def test_manual_method_with_negative_weights_raises(self):
        """Test that manual method with negative weights raises error."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["method"] = "manual"
        config["data"]["balancing"]["weighted_sampler"]["manual_weights"] = [1.0, -0.5]

        with pytest.raises(ValueError, match="positive numbers"):
            validate_config(config)

    def test_manual_method_with_valid_weights(self):
        """Test that manual method with valid weights passes."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["method"] = "manual"
        config["data"]["balancing"]["weighted_sampler"]["manual_weights"] = [1.0, 5.0]
        validate_config(config)

    def test_invalid_beta_zero(self):
        """Test that beta=0 raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["beta"] = 0

        with pytest.raises(ValueError, match="beta must be between 0 and 1"):
            validate_config(config)

    def test_invalid_beta_one(self):
        """Test that beta=1 raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["beta"] = 1

        with pytest.raises(ValueError, match="beta must be between 0 and 1"):
            validate_config(config)

    def test_invalid_target_ratio_zero(self):
        """Test that target_ratio=0 raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["downsampling"]["target_ratio"] = 0

        with pytest.raises(ValueError, match="target_ratio must be a positive float"):
            validate_config(config)

    def test_invalid_target_ratio_above_one(self):
        """Test that target_ratio > 1.0 raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["downsampling"]["target_ratio"] = 1.5

        with pytest.raises(ValueError, match="target_ratio must be a positive float"):
            validate_config(config)

    def test_valid_target_ratio(self):
        """Test that valid target_ratio passes."""
        config = _make_valid_config()
        config["data"]["balancing"]["downsampling"]["target_ratio"] = 0.5
        validate_config(config)

    def test_invalid_replacement_type(self):
        """Test that non-boolean replacement raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["replacement"] = "yes"

        with pytest.raises(ValueError, match="replacement must be a boolean"):
            validate_config(config)

    def test_invalid_num_samples(self):
        """Test that non-positive num_samples raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["num_samples"] = -1

        with pytest.raises(ValueError, match="num_samples must be a positive"):
            validate_config(config)

    def test_invalid_normalize_type(self):
        """Test that non-boolean normalize raises ValueError."""
        config = _make_valid_config()
        config["data"]["balancing"]["class_weights"]["normalize"] = "yes"

        with pytest.raises(ValueError, match="normalize must be a boolean"):
            validate_config(config)

    def test_multiple_strategies_enabled_warns(self, caplog):
        """Test that multiple strategies enabled logs a warning."""
        import logging

        config = _make_valid_config()
        config["data"]["balancing"]["weighted_sampler"]["enabled"] = True
        config["data"]["balancing"]["downsampling"]["enabled"] = True

        with caplog.at_level(logging.WARNING):
            validate_config(config)

        assert "Multiple data balancing strategies enabled" in caplog.text
