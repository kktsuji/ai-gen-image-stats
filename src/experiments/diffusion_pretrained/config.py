"""Configuration validation for the pretrained-transfer diffusion experiment.

This slice fine-tunes a public ADM (guided-diffusion) checkpoint, so its model
section differs from the from-scratch diffusion slice: instead of a U-Net
architecture spec (channel multipliers, beta schedule, ...), it specifies the
pretrained source and a freeze/unfreeze policy. The model-agnostic sections
(data, training, generation, output, compute, cross-parameter consistency) are
reused verbatim from the diffusion slice to avoid duplication.
"""

import logging
from typing import Any, Dict

# Reuse the model-agnostic validators from the diffusion slice (shared logic;
# only the model section is specific to pretrained transfer).
from src.experiments.diffusion.config import (
    _validate_compute_config,
    _validate_config_consistency,
    _validate_data_config,
    _validate_generation_config,
    _validate_output_config,
    _validate_training_config,
)
from src.utils.config import validate_experiment_section

_logger = logging.getLogger(__name__)

_VALID_NOISE_SCHEDULES = ["linear", "cosine", "quadratic", "sigmoid"]
_VALID_PRETRAINED_SOURCES = ["adm_imagenet64"]


def validate_config(config: Dict[str, Any]) -> None:
    """Validate pretrained-transfer diffusion configuration (strict).

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If configuration is invalid.
        KeyError: If required fields are missing.
    """
    validate_experiment_section(config, "diffusion_pretrained", ["train", "generate"])
    mode = config.get("mode", "train")

    _validate_compute_config(config)
    _validate_model_config(config)
    _validate_data_config(config)
    _validate_output_config(config)

    if mode == "train":
        _validate_training_config(config)
    elif mode == "generate":
        _validate_generation_config(config)

    _validate_config_consistency(config)


def _validate_model_config(config: Dict[str, Any]) -> None:
    """Validate the ADM-transfer model section."""
    if "model" not in config:
        raise KeyError("Missing required config key: model")
    model = config["model"]

    _validate_architecture(model)
    _validate_pretrained(model)
    _validate_diffusion(model)
    _validate_conditioning(model)
    _validate_initialization(model)


def _validate_architecture(model: Dict[str, Any]) -> None:
    if "architecture" not in model:
        raise KeyError("Missing required config key: model.architecture")
    arch = model["architecture"]
    for field in ["image_size", "in_channels"]:
        if field not in arch:
            raise KeyError(f"Missing required field: model.architecture.{field}")
        if not isinstance(arch[field], int) or arch[field] < 1:
            raise ValueError(f"model.architecture.{field} must be a positive integer")
    # The ADM ImageNet-64 backbone is RGB; grayscale data must be stored as 3ch.
    if arch["in_channels"] != 3:
        raise ValueError(
            "model.architecture.in_channels must be 3 for the ADM backbone "
            "(grayscale inputs should be replicated to 3 channels)"
        )


def _validate_pretrained(model: Dict[str, Any]) -> None:
    if "pretrained" not in model:
        raise KeyError("Missing required config key: model.pretrained")
    pre = model["pretrained"]

    if pre.get("source") not in _VALID_PRETRAINED_SOURCES:
        raise ValueError(
            f"model.pretrained.source must be one of {_VALID_PRETRAINED_SOURCES}, "
            f"got {pre.get('source')!r}"
        )

    has_path = bool(pre.get("checkpoint_path"))
    has_url = bool(pre.get("checkpoint_url"))
    has_cache = bool(pre.get("cache_path"))
    # Either an explicit local checkpoint, or a cache target to download into.
    if not has_path and not has_cache:
        raise ValueError(
            "model.pretrained requires either checkpoint_path (local file) or "
            "cache_path (download target)"
        )
    if not has_path and not has_url and not has_cache:
        raise ValueError(
            "model.pretrained needs checkpoint_url to download when no local "
            "checkpoint_path/cache_path exists"
        )


def _validate_diffusion(model: Dict[str, Any]) -> None:
    if "diffusion" not in model:
        raise KeyError("Missing required config key: model.diffusion")
    diff = model["diffusion"]

    if not isinstance(diff.get("num_timesteps"), int) or diff["num_timesteps"] < 1:
        raise ValueError("model.diffusion.num_timesteps must be a positive integer")

    if diff.get("noise_schedule") not in _VALID_NOISE_SCHEDULES:
        raise ValueError(
            f"model.diffusion.noise_schedule must be one of {_VALID_NOISE_SCHEDULES}, "
            f"got {diff.get('noise_schedule')!r}"
        )

    respacing = diff.get("sample_timestep_respacing")
    if not isinstance(respacing, str):
        raise ValueError(
            "model.diffusion.sample_timestep_respacing must be a string "
            "(e.g. '250', 'ddim250', or '' for the full chain)"
        )


def _validate_conditioning(model: Dict[str, Any]) -> None:
    if "conditioning" not in model:
        raise KeyError("Missing required config key: model.conditioning")
    cond = model["conditioning"]

    for field in ["type", "num_classes", "class_dropout_prob"]:
        if field not in cond:
            raise KeyError(f"Missing required field: model.conditioning.{field}")

    if cond["type"] is not None and cond["type"] != "class":
        raise ValueError(
            f"model.conditioning.type must be None or 'class', got {cond['type']!r}"
        )

    if cond["num_classes"] is not None:
        if not isinstance(cond["num_classes"], int) or cond["num_classes"] < 1:
            raise ValueError(
                "model.conditioning.num_classes must be a positive integer or None"
            )

    cdp = cond["class_dropout_prob"]
    if not isinstance(cdp, (int, float)) or cdp < 0 or cdp > 1:
        raise ValueError(
            "model.conditioning.class_dropout_prob must be a number between 0 and 1"
        )


def _validate_initialization(model: Dict[str, Any]) -> None:
    if "initialization" not in model:
        raise KeyError("Missing required config key: model.initialization")
    init = model["initialization"]

    if "freeze_backbone" not in init or not isinstance(init["freeze_backbone"], bool):
        raise ValueError("model.initialization.freeze_backbone must be a boolean")

    layers = init.get("trainable_layers")
    if layers is not None:
        if not isinstance(layers, list) or not all(isinstance(p, str) for p in layers):
            raise ValueError(
                "model.initialization.trainable_layers must be a list of strings "
                "(fnmatch patterns) or null"
            )
