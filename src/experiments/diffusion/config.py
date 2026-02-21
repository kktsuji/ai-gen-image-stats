"""Diffusion Configuration

This module provides default configuration values for diffusion model experiments.
Provides logical organization and eliminates parameter duplication.

Key features:
- Logical grouping of related parameters
- Single source of truth (no duplicate parameters)
- Mode-specific sections properly scoped
- Derived parameters (image_size, return_labels)
- Default values loaded from YAML for maintainability
"""

from typing import Any, Dict

from src.utils.config import (
    get_default_config_from_module,
    validate_compute_section,
    validate_data_loading_section,
    validate_optimizer_section,
    validate_output_section,
    validate_scheduler_section,
)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration by loading default.yaml.

    The default.yaml file is colocated with this module in the same directory,
    following the principle of keeping related files together. YAML serves as
    the single source of truth for default configuration values.

    Configuration Structure:
    - compute: Device and seed settings
    - model: Architecture, diffusion, and conditioning subsections
    - data: Paths, loading, and augmentation subsections
    - output: Base directory and subdirectories
    - training: All training-related parameters
    - generation: All generation-related parameters

    Returns:
        Dictionary containing default configuration values from YAML file

    Raises:
        FileNotFoundError: If default.yaml is not found
        yaml.YAMLError: If YAML is invalid

    Example:
        >>> config = get_default_config()
        >>> print(config["training"]["epochs"])
        200
        >>> print(config["model"]["architecture"]["image_size"])
        40
    """
    return get_default_config_from_module(__file__)


def validate_config(config: Dict[str, Any]) -> None:
    """Validate diffusion model configuration.

    Performs mode-aware validation with clear error messages.
    Validates configuration structure with nested sections.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing

    Example:
        >>> config = get_default_config()
        >>> validate_config(config)  # Should not raise
        >>> config["model"]["architecture"]["image_size"] = -1
        >>> validate_config(config)  # Raises ValueError
    """
    # Validate experiment type
    if config.get("experiment") != "diffusion":
        raise ValueError(
            f"Invalid experiment type: {config.get('experiment')}. Must be 'diffusion'"
        )

    # Validate mode
    mode = config.get("mode", "train")
    if mode not in ["train", "generate"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'generate'")

    # Validate compute section
    _validate_compute_config(config)

    # Validate model configuration
    _validate_model_config(config)

    # Validate data configuration
    _validate_data_config(config)

    # Validate output configuration
    _validate_output_config(config)

    # Mode-specific validation
    if mode == "train":
        _validate_training_config(config)
    elif mode == "generate":
        _validate_generation_config(config)

    # Cross-parameter validation
    _validate_config_consistency(config)


def _validate_compute_config(config: Dict[str, Any]) -> None:
    """Validate compute configuration section."""
    validate_compute_section(config)


def _validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration section."""
    if "model" not in config:
        raise KeyError("Missing required config key: model")

    model = config["model"]

    # Validate architecture subsection
    if "architecture" not in model:
        raise KeyError("Missing required config key: model.architecture")

    arch = model["architecture"]
    required_arch_fields = [
        "image_size",
        "in_channels",
        "model_channels",
        "channel_multipliers",
        "use_attention",
    ]
    for field in required_arch_fields:
        if field not in arch:
            raise KeyError(f"Missing required field: model.architecture.{field}")
        if arch[field] is None:
            raise ValueError(f"model.architecture.{field} cannot be None")

    if not isinstance(arch["image_size"], int) or arch["image_size"] < 1:
        raise ValueError("model.architecture.image_size must be a positive integer")

    if not isinstance(arch["in_channels"], int) or arch["in_channels"] < 1:
        raise ValueError("model.architecture.in_channels must be a positive integer")

    if not isinstance(arch["model_channels"], int) or arch["model_channels"] < 1:
        raise ValueError("model.architecture.model_channels must be a positive integer")

    if not isinstance(arch["channel_multipliers"], list):
        raise ValueError("model.architecture.channel_multipliers must be a list")

    if not all(isinstance(m, int) and m > 0 for m in arch["channel_multipliers"]):
        raise ValueError("All channel_multipliers must be positive integers")

    if not isinstance(arch["use_attention"], list):
        raise ValueError("model.architecture.use_attention must be a list")

    if not all(isinstance(a, bool) for a in arch["use_attention"]):
        raise ValueError("All use_attention values must be booleans")

    if len(arch["use_attention"]) != len(arch["channel_multipliers"]):
        raise ValueError("use_attention length must match channel_multipliers length")

    # Validate diffusion subsection
    if "diffusion" not in model:
        raise KeyError("Missing required config key: model.diffusion")

    diff = model["diffusion"]
    required_diff_fields = [
        "num_timesteps",
        "beta_schedule",
        "beta_start",
        "beta_end",
    ]
    for field in required_diff_fields:
        if field not in diff:
            raise KeyError(f"Missing required field: model.diffusion.{field}")
        if diff[field] is None:
            raise ValueError(f"model.diffusion.{field} cannot be None")

    if not isinstance(diff["num_timesteps"], int) or diff["num_timesteps"] < 1:
        raise ValueError("model.diffusion.num_timesteps must be a positive integer")

    valid_schedules = ["linear", "cosine", "quadratic", "sigmoid"]
    if diff["beta_schedule"] not in valid_schedules:
        raise ValueError(
            f"Invalid beta_schedule: {diff['beta_schedule']}. "
            f"Must be one of {valid_schedules}"
        )

    if (
        not isinstance(diff["beta_start"], (int, float))
        or diff["beta_start"] <= 0
        or diff["beta_start"] >= 1
    ):
        raise ValueError("model.diffusion.beta_start must be a number between 0 and 1")

    if (
        not isinstance(diff["beta_end"], (int, float))
        or diff["beta_end"] <= 0
        or diff["beta_end"] >= 1
    ):
        raise ValueError("model.diffusion.beta_end must be a number between 0 and 1")

    if diff["beta_start"] >= diff["beta_end"]:
        raise ValueError("beta_start must be less than beta_end")

    # Validate conditioning subsection
    if "conditioning" not in model:
        raise KeyError("Missing required config key: model.conditioning")

    cond = model["conditioning"]
    required_cond_fields = ["type", "class_dropout_prob"]
    for field in required_cond_fields:
        if field not in cond:
            raise KeyError(f"Missing required field: model.conditioning.{field}")

    cond_type = cond["type"]
    if cond_type is not None and cond_type not in ["class"]:
        raise ValueError(
            f"Invalid conditioning.type: {cond_type}. Must be None or 'class'"
        )

    if "num_classes" not in cond:
        raise KeyError("Missing required field: model.conditioning.num_classes")

    if cond["num_classes"] is not None:
        if not isinstance(cond["num_classes"], int) or cond["num_classes"] < 1:
            raise ValueError(
                "model.conditioning.num_classes must be a positive integer or None"
            )

    if (
        not isinstance(cond["class_dropout_prob"], (int, float))
        or cond["class_dropout_prob"] < 0
        or cond["class_dropout_prob"] > 1
    ):
        raise ValueError(
            "model.conditioning.class_dropout_prob must be a number between 0 and 1"
        )


def _validate_data_config(config: Dict[str, Any]) -> None:
    """Validate data configuration section."""
    if "data" not in config:
        raise KeyError("Missing required config key: data")

    data = config["data"]

    # Validate paths subsection
    if "paths" not in data:
        raise KeyError("Missing required config key: data.paths")

    paths = data["paths"]
    if "train" not in paths or paths["train"] is None:
        raise ValueError("data.paths.train is required and cannot be None")

    # Validate loading subsection
    validate_data_loading_section(data)

    # Validate augmentation subsection
    if "augmentation" not in data:
        raise KeyError("Missing required config key: data.augmentation")

    aug = data["augmentation"]
    if "rotation_degrees" in aug:
        if (
            not isinstance(aug["rotation_degrees"], (int, float))
            or aug["rotation_degrees"] < 0
        ):
            raise ValueError(
                "data.augmentation.rotation_degrees must be a non-negative number"
            )


def _validate_output_config(config: Dict[str, Any]) -> None:
    """Validate output configuration section."""
    required_subdirs = ["logs", "checkpoints", "samples", "generated"]
    validate_output_section(config, required_subdirs)


def _validate_config_consistency(config: Dict[str, Any]) -> None:
    """Validate cross-parameter consistency in configuration.

    Checks:
    - If conditioning.type == "class", num_classes must be set
    - scheduler.T_max == "auto" handling

    Args:
        config: Full configuration dictionary

    Raises:
        ValueError: If configuration has consistency issues
    """
    # Check conditioning consistency
    model = config.get("model", {})
    cond = model.get("conditioning", {})
    cond_type = cond.get("type")

    if cond_type == "class":
        num_classes = cond.get("num_classes")
        if num_classes is None or num_classes <= 0:
            raise ValueError(
                "model.conditioning.num_classes must be set and positive when "
                "conditioning.type='class'"
            )

    # Check scheduler T_max (will be handled in code, not validation)
    # Just validate the value is valid
    if "training" in config:
        scheduler_config = config["training"].get("scheduler", {})
        t_max = scheduler_config.get("T_max")
        if t_max is not None and t_max != "auto":
            if not isinstance(t_max, int) or t_max < 1:
                raise ValueError(
                    "training.scheduler.T_max must be 'auto' or a positive integer"
                )


def _validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training-specific configuration."""
    if "training" not in config:
        raise KeyError("Missing required config key: training")

    training = config["training"]

    # Validate core training parameters
    if "epochs" not in training or training["epochs"] is None:
        raise ValueError("training.epochs is required and cannot be None")

    if not isinstance(training["epochs"], int) or training["epochs"] < 1:
        raise ValueError("training.epochs must be a positive integer")

    # Validate optimizer subsection
    if "optimizer" not in training:
        raise KeyError("Missing required config key: training.optimizer")

    opt = training["optimizer"]
    if opt.get("type") is None:
        raise ValueError("training.optimizer.type is required and cannot be None")
    if opt.get("learning_rate") is None:
        raise ValueError(
            "training.optimizer.learning_rate is required and cannot be None"
        )

    validate_optimizer_section(opt, valid_types=["adam", "adamw"])

    # Validate scheduler subsection
    if "scheduler" in training:
        validate_scheduler_section(
            training["scheduler"], valid_types=["cosine", "step", "plateau", None]
        )

    # Validate EMA subsection
    if "ema" not in training:
        raise KeyError("Missing required config key: training.ema")

    ema = training["ema"]
    if "enabled" not in ema or not isinstance(ema["enabled"], bool):
        raise ValueError("training.ema.enabled must be a boolean")

    if ema["enabled"]:
        if "decay" not in ema or ema["decay"] is None:
            raise ValueError("training.ema.decay is required when EMA is enabled")

        if (
            not isinstance(ema["decay"], (int, float))
            or ema["decay"] <= 0
            or ema["decay"] >= 1
        ):
            raise ValueError("training.ema.decay must be a number between 0 and 1")

    # Validate checkpointing subsection
    if "checkpointing" not in training:
        raise KeyError("Missing required config key: training.checkpointing")

    ckpt = training["checkpointing"]
    if "save_frequency" in ckpt:
        if not isinstance(ckpt["save_frequency"], int) or ckpt["save_frequency"] < 1:
            raise ValueError(
                "training.checkpointing.save_frequency must be a positive integer"
            )
    if "save_latest" in ckpt and not isinstance(ckpt["save_latest"], bool):
        raise ValueError("training.checkpointing.save_latest must be a boolean")

    # Validate validation subsection
    if "validation" in training:
        val = training["validation"]
        if "frequency" in val:
            if not isinstance(val["frequency"], int) or val["frequency"] < 1:
                raise ValueError(
                    "training.validation.frequency must be a positive integer"
                )
        if "metric" in val and not isinstance(val["metric"], str):
            raise ValueError("training.validation.metric must be a string")

    # Validate visualization subsection
    if "visualization" in training:
        vis = training["visualization"]
        if "enabled" in vis and not isinstance(vis["enabled"], bool):
            raise ValueError("training.visualization.enabled must be a boolean")
        # Validate each interval key: must be positive int or null (None)
        for interval_key in [
            "log_images_interval",
            "log_sample_comparison_interval",
            "log_denoising_interval",
        ]:
            if interval_key in vis and vis[interval_key] is not None:
                if not isinstance(vis[interval_key], int) or vis[interval_key] < 1:
                    raise ValueError(
                        f"training.visualization.{interval_key} "
                        "interval must be a positive integer or null"
                    )
        if "num_samples" in vis:
            if not isinstance(vis["num_samples"], int) or vis["num_samples"] < 1:
                raise ValueError(
                    "training.visualization.num_samples must be a positive integer"
                )
        if "guidance_scale" in vis:
            if (
                not isinstance(vis["guidance_scale"], (int, float))
                or vis["guidance_scale"] < 1.0
            ):
                raise ValueError("training.visualization.guidance_scale must be >= 1.0")

    # Validate performance subsection
    if "performance" in training:
        perf = training["performance"]
        bool_fields = ["use_amp", "use_tf32", "cudnn_benchmark", "compile_model"]
        for field in bool_fields:
            if field in perf and not isinstance(perf[field], bool):
                raise ValueError(f"training.performance.{field} must be a boolean")

    # Validate resume subsection
    if "resume" in training:
        resume = training["resume"]
        if "enabled" in resume and not isinstance(resume["enabled"], bool):
            raise ValueError("training.resume.enabled must be a boolean")
        if resume.get("enabled") and not resume.get("checkpoint"):
            raise ValueError(
                "training.resume.checkpoint is required when resume is enabled"
            )


def _validate_generation_config(config: Dict[str, Any]) -> None:
    """Validate generation-specific configuration."""
    if "generation" not in config:
        raise KeyError("Missing required config key: generation")

    generation = config["generation"]

    # Validate checkpoint (required for generation)
    if "checkpoint" not in generation or generation["checkpoint"] is None:
        raise ValueError("generation.checkpoint is required for generate mode")

    # Validate sampling subsection
    if "sampling" in generation:
        sampling = generation["sampling"]
        if "num_samples" in sampling:
            if (
                not isinstance(sampling["num_samples"], int)
                or sampling["num_samples"] < 1
            ):
                raise ValueError(
                    "generation.sampling.num_samples must be a positive integer"
                )

        if "guidance_scale" in sampling:
            if (
                not isinstance(sampling["guidance_scale"], (int, float))
                or sampling["guidance_scale"] < 1.0
            ):
                raise ValueError("generation.sampling.guidance_scale must be >= 1.0")

        if "use_ema" in sampling and not isinstance(sampling["use_ema"], bool):
            raise ValueError("generation.sampling.use_ema must be a boolean")

        if "ema_decay" in sampling:
            if sampling["use_ema"] if "use_ema" in sampling else True:
                if (
                    not isinstance(sampling["ema_decay"], (int, float))
                    or sampling["ema_decay"] <= 0
                    or sampling["ema_decay"] >= 1
                ):
                    raise ValueError(
                        "generation.sampling.ema_decay must be a number between 0 and 1"
                    )

        if "batch_size" in sampling:
            if (
                not isinstance(sampling["batch_size"], int)
                or sampling["batch_size"] < 1
            ):
                raise ValueError(
                    "generation.sampling.batch_size must be a positive integer"
                )

    # Validate output subsection
    if "output" in generation:
        out = generation["output"]
        if "save_individual" in out and not isinstance(out["save_individual"], bool):
            raise ValueError("generation.output.save_individual must be a boolean")
        if "save_grid" in out and not isinstance(out["save_grid"], bool):
            raise ValueError("generation.output.save_grid must be a boolean")
        if "grid_nrow" in out:
            if not isinstance(out["grid_nrow"], int) or out["grid_nrow"] < 1:
                raise ValueError(
                    "generation.output.grid_nrow must be a positive integer"
                )


def get_resolution_config(image_size: int) -> Dict[str, Any]:
    """Get resolution-specific configuration overrides.

    Different image sizes may require different model architectures and
    training settings for optimal results.

    Args:
        image_size: Target image resolution (e.g., 40, 64, 128, 256)

    Returns:
        Dictionary containing resolution-specific configuration overrides

    Raises:
        ValueError: If image_size is not supported

    Example:
        >>> config = get_resolution_config(64)
        >>> print(config["model"]["architecture"]["model_channels"])
        128
    """
    resolution_configs = {
        40: {
            "model": {
                "architecture": {
                    "image_size": 40,
                    "model_channels": 64,
                    "channel_multipliers": [1, 2, 4],
                    "use_attention": [False, False, True],
                },
            },
            "data": {
                "loading": {
                    "batch_size": 32,
                },
            },
        },
        64: {
            "model": {
                "architecture": {
                    "image_size": 64,
                    "model_channels": 128,
                    "channel_multipliers": [1, 2, 2, 2],
                    "use_attention": [False, False, False, True],
                },
            },
            "data": {
                "loading": {
                    "batch_size": 64,
                },
            },
        },
        128: {
            "model": {
                "architecture": {
                    "image_size": 128,
                    "model_channels": 128,
                    "channel_multipliers": [1, 1, 2, 2, 4],
                    "use_attention": [False, False, False, False, True],
                },
            },
            "data": {
                "loading": {
                    "batch_size": 32,
                },
            },
        },
        256: {
            "model": {
                "architecture": {
                    "image_size": 256,
                    "model_channels": 128,
                    "channel_multipliers": [1, 1, 2, 2, 4, 4],
                    "use_attention": [False, False, False, False, False, True],
                },
            },
            "data": {
                "loading": {
                    "batch_size": 16,
                },
            },
        },
    }

    if image_size not in resolution_configs:
        raise ValueError(
            f"Unsupported image_size: {image_size}. "
            f"Must be one of {list(resolution_configs.keys())}"
        )

    return resolution_configs[image_size]
