"""Classifier Configuration

This module provides default configuration values for classifier experiments.
It defines sensible defaults for training, data loading, model selection, and logging.
"""

from typing import Any, Dict, Optional


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for classifier experiments.

    Returns:
        Dictionary containing default configuration values

    Example:
        >>> config = get_default_config()
        >>> print(config["training"]["epochs"])
        100
    """
    return {
        "experiment": "classifier",
        "model": {
            "name": "resnet50",  # Options: resnet50, resnet101, resnet152, inceptionv3
            "pretrained": True,
            "num_classes": 2,
            "freeze_backbone": False,
            "trainable_layers": None,  # List of layer patterns to unfreeze (InceptionV3 only)
            "dropout": 0.5,  # Only for InceptionV3
        },
        "data": {
            "train_path": "data/train",
            "val_path": "data/val",
            "batch_size": 32,
            "num_workers": 4,
            "image_size": 256,
            "crop_size": 224,
            "horizontal_flip": True,
            "color_jitter": False,
            "rotation_degrees": 0,
            "normalize": "imagenet",  # Options: imagenet, cifar10, none, None
            "pin_memory": True,
            "drop_last": False,
            "shuffle_train": True,
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",  # Options: adam, sgd, adamw
            "optimizer_kwargs": {
                "weight_decay": 1e-4,
            },
            "scheduler": "cosine",  # Options: cosine, step, plateau, none
            "scheduler_kwargs": {
                "T_max": 100,  # For cosine
                "eta_min": 1e-6,  # For cosine
            },
            "device": "cuda",  # Options: cuda, cpu
            "mixed_precision": False,  # Use automatic mixed precision (AMP)
            "gradient_clip": None,  # Max gradient norm for clipping (None to disable)
            "early_stopping_patience": None,  # Epochs to wait before early stopping (None to disable)
        },
        "output": {
            "checkpoint_dir": "outputs/checkpoints",
            "log_dir": "outputs/logs",
            "save_best_only": True,
            "save_frequency": 10,  # Save checkpoint every N epochs (0 to save only best)
        },
        "validation": {
            "frequency": 1,  # Run validation every N epochs
            "metric": "accuracy",  # Metric to monitor for best model (accuracy or loss)
        },
    }


def validate_config(config: Dict[str, Any]) -> None:
    """Validate classifier configuration.

    Checks that all required fields are present and have valid values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing

    Example:
        >>> config = get_default_config()
        >>> validate_config(config)  # Should not raise
        >>> config["model"]["name"] = "invalid_model"
        >>> validate_config(config)  # Raises ValueError
    """
    # Check required top-level keys
    required_keys = ["experiment", "model", "data", "training", "output"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    # Validate experiment type
    if config["experiment"] != "classifier":
        raise ValueError(
            f"Invalid experiment type: {config['experiment']}. Must be 'classifier'"
        )

    # Validate model configuration
    model = config["model"]
    valid_models = ["resnet50", "resnet101", "resnet152", "inceptionv3"]
    if model["name"] not in valid_models:
        raise ValueError(
            f"Invalid model name: {model['name']}. Must be one of {valid_models}"
        )

    if not isinstance(model["num_classes"], int) or model["num_classes"] < 1:
        raise ValueError("num_classes must be a positive integer")

    if not isinstance(model["pretrained"], bool):
        raise ValueError("pretrained must be a boolean")

    if not isinstance(model["freeze_backbone"], bool):
        raise ValueError("freeze_backbone must be a boolean")

    # Validate trainable_layers if specified
    if model.get("trainable_layers") is not None:
        if not isinstance(model["trainable_layers"], list):
            raise ValueError("trainable_layers must be a list or None")
        if not all(isinstance(layer, str) for layer in model["trainable_layers"]):
            raise ValueError("All trainable_layers must be strings")

    # Validate data configuration
    data = config["data"]
    if not isinstance(data["batch_size"], int) or data["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")

    if not isinstance(data["num_workers"], int) or data["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")

    if not isinstance(data["image_size"], int) or data["image_size"] < 1:
        raise ValueError("image_size must be a positive integer")

    if not isinstance(data["crop_size"], int) or data["crop_size"] < 1:
        raise ValueError("crop_size must be a positive integer")

    valid_normalize = ["imagenet", "cifar10", "none", None]
    if data["normalize"] not in valid_normalize:
        raise ValueError(
            f"Invalid normalize option: {data['normalize']}. "
            f"Must be one of {valid_normalize}"
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

    valid_optimizers = ["adam", "sgd", "adamw"]
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

    # Validate output configuration
    output = config["output"]
    if "checkpoint_dir" not in output:
        raise KeyError("Missing required output.checkpoint_dir")

    if "log_dir" not in output:
        raise KeyError("Missing required output.log_dir")


def get_model_specific_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration overrides.

    Args:
        model_name: Name of the model (resnet50, resnet101, resnet152, inceptionv3)

    Returns:
        Dictionary containing model-specific configuration overrides

    Raises:
        ValueError: If model_name is not supported

    Example:
        >>> config = get_model_specific_config("inceptionv3")
        >>> print(config["data"]["image_size"])
        299
    """
    model_configs = {
        "resnet50": {
            "data": {
                "image_size": 256,
                "crop_size": 224,
                "normalize": "imagenet",
            },
        },
        "resnet101": {
            "data": {
                "image_size": 256,
                "crop_size": 224,
                "normalize": "imagenet",
            },
        },
        "resnet152": {
            "data": {
                "image_size": 256,
                "crop_size": 224,
                "normalize": "imagenet",
            },
        },
        "inceptionv3": {
            "data": {
                "image_size": 320,  # Resize to slightly larger
                "crop_size": 299,  # InceptionV3 expects 299x299
                "normalize": "imagenet",
            },
            "model": {
                "dropout": 0.5,
            },
        },
    }

    if model_name not in model_configs:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Must be one of {list(model_configs.keys())}"
        )

    return model_configs[model_name]
