"""Diffusion Configuration

This module provides default configuration values for diffusion model experiments.
It defines sensible defaults for training, data loading, model architecture, and generation.
"""

from typing import Any, Dict, Optional


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for diffusion model experiments.

    Returns:
        Dictionary containing default configuration values

    Example:
        >>> config = get_default_config()
        >>> print(config["training"]["epochs"])
        200
    """
    return {
        "experiment": "diffusion",
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
        "training": {
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
            "device": "cuda",  # Options: cuda, cpu
            "use_ema": True,  # Use exponential moving average
            "ema_decay": 0.9999,  # EMA decay rate
            "use_amp": False,  # Use automatic mixed precision
            "gradient_clip_norm": None,  # Max gradient norm (None to disable)
        },
        "generation": {
            "sample_images": True,  # Generate samples during training
            "sample_interval": 10,  # Generate every N epochs
            "samples_per_class": 2,  # Samples per class for conditional models
            "guidance_scale": 3.0,  # Classifier-free guidance scale
        },
        "output": {
            "checkpoint_dir": "outputs/checkpoints",
            "log_dir": "outputs/logs",
            "save_best_only": False,  # Save all checkpoints (diffusion typically needs many)
            "save_frequency": 10,  # Save checkpoint every N epochs
        },
        "validation": {
            "frequency": 1,  # Run validation every N epochs
            "metric": "loss",  # Metric to monitor for best model
        },
    }


def validate_config(config: Dict[str, Any]) -> None:
    """Validate diffusion model configuration.

    Checks that all required fields are present and have valid values.

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
    # Check required top-level keys
    required_keys = ["experiment", "model", "data", "training", "output"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    # Validate experiment type
    if config["experiment"] != "diffusion":
        raise ValueError(
            f"Invalid experiment type: {config['experiment']}. Must be 'diffusion'"
        )

    # Validate model configuration
    model = config["model"]

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

    # Validate data configuration
    data = config["data"]

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
        not isinstance(data["rotation_degrees"], (int, float))
        or data["rotation_degrees"] < 0
    ):
        raise ValueError("rotation_degrees must be a non-negative number")

    if not isinstance(data["return_labels"], bool):
        raise ValueError("return_labels must be a boolean")

    # Check consistency between conditional generation and labels
    if model["num_classes"] is not None and not data["return_labels"]:
        raise ValueError(
            "data.return_labels must be True when model.num_classes is set "
            "(conditional generation requires labels)"
        )

    # Validate training configuration
    training = config["training"]

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
    if training["scheduler"] not in valid_schedulers:
        raise ValueError(
            f"Invalid scheduler: {training['scheduler']}. "
            f"Must be one of {valid_schedulers}"
        )

    valid_devices = ["cuda", "cpu"]
    if training["device"] not in valid_devices:
        raise ValueError(
            f"Invalid device: {training['device']}. Must be one of {valid_devices}"
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

    if training["gradient_clip_norm"] is not None:
        if (
            not isinstance(training["gradient_clip_norm"], (int, float))
            or training["gradient_clip_norm"] <= 0
        ):
            raise ValueError("gradient_clip_norm must be a positive number or None")

    # Validate generation configuration
    if "generation" in config:
        generation = config["generation"]

        if not isinstance(generation["sample_images"], bool):
            raise ValueError("sample_images must be a boolean")

        if (
            not isinstance(generation["sample_interval"], int)
            or generation["sample_interval"] < 1
        ):
            raise ValueError("sample_interval must be a positive integer")

        if (
            not isinstance(generation["samples_per_class"], int)
            or generation["samples_per_class"] < 1
        ):
            raise ValueError("samples_per_class must be a positive integer")

        if (
            not isinstance(generation["guidance_scale"], (int, float))
            or generation["guidance_scale"] < 1.0
        ):
            raise ValueError("guidance_scale must be a number >= 1.0")

    # Validate output configuration
    output = config["output"]

    if "checkpoint_dir" not in output:
        raise KeyError("Missing required output.checkpoint_dir")

    if "log_dir" not in output:
        raise KeyError("Missing required output.log_dir")


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
