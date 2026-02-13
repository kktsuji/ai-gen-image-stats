"""Diffusion Configuration

This module provides default configuration values for diffusion model experiments.
It defines sensible defaults for training, data loading, model architecture, and generation.
"""

from typing import Any, Dict, Optional


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for diffusion model experiments.

    Configuration Structure:
    - Common parameters at top level (device, seed)
    - Mode-specific parameters in training/generation sections
    - Logical nesting for related features

    Returns:
        Dictionary containing default configuration values

    Example:
        >>> config = get_default_config()
        >>> print(config["training"]["epochs"])
        200
    """
    return {
        "experiment": "diffusion",
        "mode": "train",  # Options: train, generate
        # Common parameters (used in both modes)
        "device": "cuda",  # Options: cuda, cpu, auto
        "seed": None,  # Random seed for reproducibility
        "model": {
            "image_size": 40,  # Size of generated images
            "in_channels": 3,  # Number of input channels (RGB)
            "model_channels": 64,  # Base number of U-Net channels
            "channel_multipliers": [1, 2, 4],  # Channel multipliers per stage
            "num_classes": None,  # Number of classes for conditional generation (None for unconditional)
            "num_timesteps": 1000,  # Number of diffusion timesteps
            "beta_schedule": "cosine",  # Options: linear, cosine, quadratic, sigmoid
            "beta_start": 0.0001,  # Starting beta value
            "beta_end": 0.02,  # Ending beta value
            "class_dropout_prob": 0.1,  # Classifier-free guidance dropout
            "use_attention": [False, False, True],  # Use attention at each stage
        },
        "data": {
            "train_path": "data/train",
            "val_path": None,  # Optional validation data path
            "batch_size": 32,
            "num_workers": 4,
            "image_size": 40,  # Should match model.image_size
            "horizontal_flip": True,
            "rotation_degrees": 0,
            "color_jitter": False,
            "color_jitter_strength": 0.1,
            "pin_memory": True,
            "drop_last": False,
            "shuffle_train": True,
            "return_labels": False,  # Set to True for conditional generation
        },
        "output": {
            "log_dir": "outputs/logs",  # Only common output parameter
        },
        "training": {
            # Core training parameters
            "epochs": 200,
            "learning_rate": 0.0001,
            "optimizer": "adam",  # Options: adam, adamw
            "optimizer_kwargs": {
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": None,  # Options: cosine, step, plateau, none, None
            "scheduler_kwargs": {
                "T_max": 200,  # For cosine
                "eta_min": 1e-6,  # For cosine
            },
            # Advanced training features
            "use_ema": True,  # Use exponential moving average
            "ema_decay": 0.9999,  # EMA decay rate
            "use_amp": False,  # Use automatic mixed precision
            "gradient_clip_norm": None,  # Max gradient norm (None to disable)
            # Training checkpointing
            "checkpoint_dir": "outputs/checkpoints",
            "save_best_only": False,  # Save all checkpoints (diffusion typically needs many)
            "save_frequency": 10,  # Save checkpoint every N epochs
            # Training validation (nested)
            "validation": {
                "frequency": 1,  # Run validation every N epochs
                "metric": "loss",  # Metric to monitor for best model
            },
            # Training visualization (nested)
            "visualization": {
                "sample_images": True,  # Generate samples during training
                "sample_interval": 10,  # Generate every N epochs
                "samples_per_class": 2,  # Samples per class for conditional models
                "guidance_scale": 3.0,  # Classifier-free guidance scale
            },
        },
        "generation": {
            # Generation input
            "checkpoint": None,  # Required for generate mode
            # Generation parameters
            "num_samples": 100,  # Number of samples to generate in generation mode
            "guidance_scale": 3.0,  # Classifier-free guidance scale
            "use_ema": True,  # Use EMA weights if available
            # Generation output
            "output_dir": None,  # Output directory for generated images (defaults to log_dir/generated)
            "save_grid": True,
            "grid_nrow": 10,
        },
    }


def validate_config(config: Dict[str, Any]) -> None:
    """Validate diffusion model configuration.

    Performs mode-aware validation with clear error messages.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing

    Example:
        >>> config = get_default_config()
        >>> validate_config(config)  # Should not raise
        >>> config["model"]["image_size"] = -1
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

    # Validate common parameters
    device = config.get("device", "cuda")
    valid_devices = ["cuda", "cpu", "auto"]
    if device not in valid_devices:
        raise ValueError(f"Invalid device: {device}. Must be one of {valid_devices}")

    seed = config.get("seed")
    if seed is not None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be None or a non-negative integer")

    # Validate model configuration
    _validate_model_config(config)

    # Validate data configuration
    _validate_data_config(config)

    # Validate output section (simplified)
    if "output" not in config:
        raise KeyError("Missing required config key: output")

    output = config["output"]
    if "log_dir" not in output or output["log_dir"] is None:
        raise ValueError("output.log_dir is required and cannot be None")

    # Mode-specific validation
    if mode == "train":
        _validate_training_config(config)
    elif mode == "generate":
        _validate_generation_config(config)


def _validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration section."""
    if "model" not in config:
        raise KeyError("Missing required config key: model")

    model = config["model"]
    required_model_fields = [
        "image_size",
        "in_channels",
        "model_channels",
        "channel_multipliers",
        "num_timesteps",
        "beta_schedule",
        "beta_start",
        "beta_end",
        "class_dropout_prob",
        "use_attention",
    ]
    for field in required_model_fields:
        if field not in model:
            raise KeyError(f"Missing required field: model.{field}")
        if model[field] is None:
            raise ValueError(f"model.{field} cannot be None")

    if not isinstance(model["image_size"], int) or model["image_size"] < 1:
        raise ValueError("image_size must be a positive integer")

    if not isinstance(model["in_channels"], int) or model["in_channels"] < 1:
        raise ValueError("in_channels must be a positive integer")

    if not isinstance(model["model_channels"], int) or model["model_channels"] < 1:
        raise ValueError("model_channels must be a positive integer")

    if not isinstance(model["channel_multipliers"], list):
        raise ValueError("channel_multipliers must be a list")

    if not all(isinstance(m, int) and m > 0 for m in model["channel_multipliers"]):
        raise ValueError("All channel_multipliers must be positive integers")

    if model["num_classes"] is not None:
        if not isinstance(model["num_classes"], int) or model["num_classes"] < 1:
            raise ValueError("num_classes must be a positive integer or None")

    if not isinstance(model["num_timesteps"], int) or model["num_timesteps"] < 1:
        raise ValueError("num_timesteps must be a positive integer")

    valid_schedules = ["linear", "cosine", "quadratic", "sigmoid"]
    if model["beta_schedule"] not in valid_schedules:
        raise ValueError(
            f"Invalid beta_schedule: {model['beta_schedule']}. "
            f"Must be one of {valid_schedules}"
        )

    if (
        not isinstance(model["beta_start"], (int, float))
        or model["beta_start"] <= 0
        or model["beta_start"] >= 1
    ):
        raise ValueError("beta_start must be a number between 0 and 1")

    if (
        not isinstance(model["beta_end"], (int, float))
        or model["beta_end"] <= 0
        or model["beta_end"] >= 1
    ):
        raise ValueError("beta_end must be a number between 0 and 1")

    if model["beta_start"] >= model["beta_end"]:
        raise ValueError("beta_start must be less than beta_end")

    if (
        not isinstance(model["class_dropout_prob"], (int, float))
        or model["class_dropout_prob"] < 0
        or model["class_dropout_prob"] > 1
    ):
        raise ValueError("class_dropout_prob must be a number between 0 and 1")

    if not isinstance(model["use_attention"], list):
        raise ValueError("use_attention must be a list")

    if not all(isinstance(a, bool) for a in model["use_attention"]):
        raise ValueError("All use_attention values must be booleans")

    if len(model["use_attention"]) != len(model["channel_multipliers"]):
        raise ValueError("use_attention length must match channel_multipliers length")


def _validate_data_config(config: Dict[str, Any]) -> None:
    """Validate data configuration section."""
    if "data" not in config:
        raise KeyError("Missing required config key: data")

    data = config["data"]
    model = config["model"]

    required_data_fields = [
        "train_path",
        "batch_size",
        "num_workers",
        "image_size",
        "return_labels",
    ]
    for field in required_data_fields:
        if field not in data:
            raise KeyError(f"Missing required field: data.{field}")
        if data[field] is None:
            raise ValueError(f"data.{field} cannot be None")

    if not isinstance(data["batch_size"], int) or data["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")

    if not isinstance(data["num_workers"], int) or data["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")

    if not isinstance(data["image_size"], int) or data["image_size"] < 1:
        raise ValueError("image_size must be a positive integer")

    # Check that data.image_size matches model.image_size
    if data["image_size"] != model["image_size"]:
        raise ValueError(
            f"data.image_size ({data['image_size']}) must match "
            f"model.image_size ({model['image_size']})"
        )

    if (
        not isinstance(data.get("rotation_degrees", 0), (int, float))
        or data.get("rotation_degrees", 0) < 0
    ):
        raise ValueError("rotation_degrees must be a non-negative number")

    if not isinstance(data["return_labels"], bool):
        raise ValueError("return_labels must be a boolean")

    # Check consistency between conditional generation and labels
    if model.get("num_classes") is not None and not data["return_labels"]:
        raise ValueError(
            "data.return_labels must be True when model.num_classes is set "
            "(conditional generation requires labels)"
        )


def _validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training-specific configuration."""
    if "training" not in config:
        raise KeyError("Missing required config key: training")

    training = config["training"]

    # Validate core training parameters
    required_training_fields = [
        "epochs",
        "learning_rate",
        "optimizer",
        "use_ema",
        "ema_decay",
        "use_amp",
    ]
    for field in required_training_fields:
        if field not in training:
            raise KeyError(f"Missing required field: training.{field}")
        if training[field] is None:
            raise ValueError(f"training.{field} cannot be None")

    if not isinstance(training["epochs"], int) or training["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")

    if (
        not isinstance(training["learning_rate"], (int, float))
        or training["learning_rate"] <= 0
    ):
        raise ValueError("learning_rate must be a positive number")

    valid_optimizers = ["adam", "adamw"]
    if training["optimizer"] not in valid_optimizers:
        raise ValueError(
            f"Invalid optimizer: {training['optimizer']}. "
            f"Must be one of {valid_optimizers}"
        )

    valid_schedulers = ["cosine", "step", "plateau", "none", None]
    if training.get("scheduler") not in valid_schedulers:
        raise ValueError(
            f"Invalid scheduler: {training.get('scheduler')}. "
            f"Must be one of {valid_schedulers}"
        )

    if not isinstance(training["use_ema"], bool):
        raise ValueError("use_ema must be a boolean")

    if (
        not isinstance(training["ema_decay"], (int, float))
        or training["ema_decay"] <= 0
        or training["ema_decay"] >= 1
    ):
        raise ValueError("ema_decay must be a number between 0 and 1")

    if not isinstance(training["use_amp"], bool):
        raise ValueError("use_amp must be a boolean")

    if training.get("gradient_clip_norm") is not None:
        if (
            not isinstance(training["gradient_clip_norm"], (int, float))
            or training["gradient_clip_norm"] <= 0
        ):
            raise ValueError("gradient_clip_norm must be a positive number or None")

    # Validate checkpointing
    if "checkpoint_dir" not in training or training["checkpoint_dir"] is None:
        raise ValueError("training.checkpoint_dir is required for training mode")

    # Validate nested validation section
    if "validation" in training:
        val = training["validation"]
        if "frequency" in val:
            if not isinstance(val["frequency"], int) or val["frequency"] < 1:
                raise ValueError(
                    "training.validation.frequency must be a positive integer"
                )
        if "metric" in val and not isinstance(val["metric"], str):
            raise ValueError("training.validation.metric must be a string")

    # Validate nested visualization section
    if "visualization" in training:
        vis = training["visualization"]
        if "sample_images" in vis and not isinstance(vis["sample_images"], bool):
            raise ValueError("training.visualization.sample_images must be a boolean")
        if "sample_interval" in vis:
            if (
                not isinstance(vis["sample_interval"], int)
                or vis["sample_interval"] < 1
            ):
                raise ValueError(
                    "training.visualization.sample_interval must be a positive integer"
                )
        if "samples_per_class" in vis:
            if (
                not isinstance(vis["samples_per_class"], int)
                or vis["samples_per_class"] < 1
            ):
                raise ValueError(
                    "training.visualization.samples_per_class must be a positive integer"
                )
        if "guidance_scale" in vis:
            if (
                not isinstance(vis["guidance_scale"], (int, float))
                or vis["guidance_scale"] < 1.0
            ):
                raise ValueError("training.visualization.guidance_scale must be >= 1.0")


def _validate_generation_config(config: Dict[str, Any]) -> None:
    """Validate generation-specific configuration."""
    if "generation" not in config:
        raise KeyError("Missing required config key: generation")

    generation = config["generation"]

    # Validate checkpoint (required for generation)
    if "checkpoint" not in generation or generation["checkpoint"] is None:
        raise ValueError("generation.checkpoint is required for generate mode")

    # Validate generation parameters
    if "num_samples" in generation:
        if (
            not isinstance(generation["num_samples"], int)
            or generation["num_samples"] < 1
        ):
            raise ValueError("generation.num_samples must be a positive integer")

    if "guidance_scale" in generation:
        if (
            not isinstance(generation["guidance_scale"], (int, float))
            or generation["guidance_scale"] < 1.0
        ):
            raise ValueError("generation.guidance_scale must be >= 1.0")

    if "use_ema" in generation and not isinstance(generation["use_ema"], bool):
        raise ValueError("generation.use_ema must be a boolean")

    if "save_grid" in generation and not isinstance(generation["save_grid"], bool):
        raise ValueError("generation.save_grid must be a boolean")

    if "grid_nrow" in generation:
        if not isinstance(generation["grid_nrow"], int) or generation["grid_nrow"] < 1:
            raise ValueError("generation.grid_nrow must be a positive integer")


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
        >>> print(config["model"]["model_channels"])
        128
    """
    resolution_configs = {
        40: {
            "model": {
                "image_size": 40,
                "model_channels": 64,
                "channel_multipliers": [1, 2, 4],
                "use_attention": [False, False, True],
            },
            "data": {
                "image_size": 40,
                "batch_size": 32,
            },
        },
        64: {
            "model": {
                "image_size": 64,
                "model_channels": 128,
                "channel_multipliers": [1, 2, 2, 2],
                "use_attention": [False, False, False, True],
            },
            "data": {
                "image_size": 64,
                "batch_size": 64,
            },
        },
        128: {
            "model": {
                "image_size": 128,
                "model_channels": 128,
                "channel_multipliers": [1, 1, 2, 2, 4],
                "use_attention": [False, False, False, False, True],
            },
            "data": {
                "image_size": 128,
                "batch_size": 32,
            },
        },
        256: {
            "model": {
                "image_size": 256,
                "model_channels": 128,
                "channel_multipliers": [1, 1, 2, 2, 4, 4],
                "use_attention": [False, False, False, False, False, True],
            },
            "data": {
                "image_size": 256,
                "batch_size": 16,
            },
        },
    }

    if image_size not in resolution_configs:
        raise ValueError(
            f"Unsupported image_size: {image_size}. "
            f"Must be one of {list(resolution_configs.keys())}"
        )

    return resolution_configs[image_size]
