"""Tests for diffusion_pretrained config validation."""

import copy

import pytest

from src.experiments.diffusion_pretrained.config import validate_config


@pytest.mark.unit
def test_example_config_is_valid_train_mode(example_config):
    validate_config(example_config)  # should not raise


@pytest.mark.unit
def test_example_config_is_valid_generate_mode(example_config):
    cfg = copy.deepcopy(example_config)
    cfg["mode"] = "generate"
    cfg["generation"]["checkpoint"] = "outputs/x/checkpoints/best_model.pth"
    validate_config(cfg)  # should not raise


@pytest.mark.unit
def test_rejects_non_rgb_in_channels(example_config):
    example_config["model"]["architecture"]["in_channels"] = 1
    with pytest.raises(ValueError, match="in_channels must be 3"):
        validate_config(example_config)


@pytest.mark.unit
def test_rejects_unknown_pretrained_source(example_config):
    example_config["model"]["pretrained"]["source"] = "stable_diffusion"
    with pytest.raises(ValueError, match="pretrained.source"):
        validate_config(example_config)


@pytest.mark.unit
def test_rejects_bad_noise_schedule(example_config):
    example_config["model"]["diffusion"]["noise_schedule"] = "bogus"
    with pytest.raises(ValueError, match="noise_schedule"):
        validate_config(example_config)


@pytest.mark.unit
def test_rejects_non_string_respacing(example_config):
    example_config["model"]["diffusion"]["sample_timestep_respacing"] = 250
    with pytest.raises(ValueError, match="sample_timestep_respacing"):
        validate_config(example_config)


@pytest.mark.unit
def test_rejects_non_list_trainable_layers(example_config):
    example_config["model"]["initialization"]["trainable_layers"] = "out.*"
    with pytest.raises(ValueError, match="trainable_layers"):
        validate_config(example_config)


@pytest.mark.unit
def test_rejects_non_bool_freeze_backbone(example_config):
    example_config["model"]["initialization"]["freeze_backbone"] = "yes"
    with pytest.raises(ValueError, match="freeze_backbone"):
        validate_config(example_config)


@pytest.mark.unit
def test_missing_pretrained_section_raises(example_config):
    del example_config["model"]["pretrained"]
    with pytest.raises(KeyError):
        validate_config(example_config)


@pytest.mark.unit
def test_pretrained_requires_path_or_cache(example_config):
    example_config["model"]["pretrained"]["checkpoint_path"] = None
    example_config["model"]["pretrained"]["cache_path"] = None
    with pytest.raises(ValueError, match="checkpoint_path|cache_path"):
        validate_config(example_config)
