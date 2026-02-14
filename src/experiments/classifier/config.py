"""Classifier Configuration

This module provides default configuration values for classifier experiments.
It defines sensible defaults for training, data loading, model selection, and logging.

Supports both V1 (legacy) and V2 configuration formats.
"""

import warnings
from typing import Any, Dict


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


# Note: validate_config() is defined later in this file, after validate_config_v2()
# and _validate_config_v1(). It auto-detects V1 vs V2 format.


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


def validate_config_v2(config: Dict[str, Any]) -> None:
    """Validate V2 classifier configuration.

    Checks that all required fields are present and have valid values in V2 format.

    Args:
        config: Configuration dictionary to validate (V2 format)

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing
    """
    # Check required top-level keys
    required_keys = ["experiment", "mode", "compute", "model", "data", "output"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    # Validate experiment type
    if config["experiment"] != "classifier":
        raise ValueError(
            f"Invalid experiment type: {config['experiment']}. Must be 'classifier'"
        )

    # Validate mode
    valid_modes = ["train", "evaluate"]
    if config["mode"] not in valid_modes:
        raise ValueError(
            f"Invalid mode: {config['mode']}. Must be one of {valid_modes}"
        )

    # Validate compute configuration
    compute = config["compute"]
    required_compute_fields = ["device"]
    for field in required_compute_fields:
        if field not in compute:
            raise KeyError(f"Missing required field: compute.{field}")

    valid_devices = ["cuda", "cpu", "auto"]
    if compute["device"] not in valid_devices:
        raise ValueError(
            f"Invalid device: {compute['device']}. Must be one of {valid_devices}"
        )

    # Validate model configuration
    model = config["model"]
    if "architecture" not in model:
        raise KeyError("Missing required field: model.architecture")

    architecture = model["architecture"]
    required_arch_fields = ["name", "num_classes"]
    for field in required_arch_fields:
        if field not in architecture:
            raise KeyError(f"Missing required field: model.architecture.{field}")
        if architecture[field] is None:
            raise ValueError(f"model.architecture.{field} cannot be None")

    valid_models = ["resnet50", "resnet101", "resnet152", "inceptionv3"]
    if architecture["name"] not in valid_models:
        raise ValueError(
            f"Invalid model name: {architecture['name']}. Must be one of {valid_models}"
        )

    if (
        not isinstance(architecture["num_classes"], int)
        or architecture["num_classes"] < 1
    ):
        raise ValueError("num_classes must be a positive integer")

    # Validate initialization
    if "initialization" in model:
        init = model["initialization"]
        if "pretrained" in init and not isinstance(init["pretrained"], bool):
            raise ValueError("initialization.pretrained must be a boolean")
        if "freeze_backbone" in init and not isinstance(init["freeze_backbone"], bool):
            raise ValueError("initialization.freeze_backbone must be a boolean")

    # Validate data configuration
    data = config["data"]
    required_data_sections = ["paths", "loading", "preprocessing", "augmentation"]
    for section in required_data_sections:
        if section not in data:
            raise KeyError(f"Missing required field: data.{section}")

    # Validate paths
    paths = data["paths"]
    if "train" not in paths:
        raise KeyError("Missing required field: data.paths.train")

    # Validate loading
    loading = data["loading"]
    required_loading_fields = ["batch_size", "num_workers"]
    for field in required_loading_fields:
        if field not in loading:
            raise KeyError(f"Missing required field: data.loading.{field}")

    if not isinstance(loading["batch_size"], int) or loading["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")

    if not isinstance(loading["num_workers"], int) or loading["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")

    # Validate preprocessing
    preprocessing = data["preprocessing"]
    required_preprocessing_fields = ["image_size", "crop_size"]
    for field in required_preprocessing_fields:
        if field not in preprocessing:
            raise KeyError(f"Missing required field: data.preprocessing.{field}")

    if (
        not isinstance(preprocessing["image_size"], int)
        or preprocessing["image_size"] < 1
    ):
        raise ValueError("image_size must be a positive integer")

    if (
        not isinstance(preprocessing["crop_size"], int)
        or preprocessing["crop_size"] < 1
    ):
        raise ValueError("crop_size must be a positive integer")

    valid_normalize = ["imagenet", "cifar10", "custom", None]
    if preprocessing.get("normalize") not in valid_normalize:
        raise ValueError(
            f"Invalid normalize option: {preprocessing.get('normalize')}. "
            f"Must be one of {valid_normalize}"
        )

    # Validate output configuration
    output = config["output"]
    required_output_fields = ["base_dir", "subdirs"]
    for field in required_output_fields:
        if field not in output:
            raise KeyError(f"Missing required field: output.{field}")

    required_subdirs = ["logs", "checkpoints"]
    for subdir in required_subdirs:
        if subdir not in output["subdirs"]:
            raise KeyError(f"Missing required field: output.subdirs.{subdir}")

    # Mode-specific validation
    if config["mode"] == "train":
        if "training" not in config:
            raise KeyError(
                "Missing required section: training (required for train mode)"
            )

        training = config["training"]
        required_training_fields = ["epochs", "optimizer", "scheduler"]
        for field in required_training_fields:
            if field not in training:
                raise KeyError(f"Missing required field: training.{field}")

        if not isinstance(training["epochs"], int) or training["epochs"] < 1:
            raise ValueError("epochs must be a positive integer")

        # Validate optimizer
        optimizer = training["optimizer"]
        if "type" not in optimizer:
            raise KeyError("Missing required field: training.optimizer.type")
        if "learning_rate" not in optimizer:
            raise KeyError("Missing required field: training.optimizer.learning_rate")

        valid_optimizers = ["adam", "adamw", "sgd"]
        if optimizer["type"] not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer: {optimizer['type']}. Must be one of {valid_optimizers}"
            )

        if (
            not isinstance(optimizer["learning_rate"], (int, float))
            or optimizer["learning_rate"] <= 0
        ):
            raise ValueError("learning_rate must be a positive number")

        # Validate scheduler
        scheduler = training["scheduler"]
        if "type" not in scheduler:
            raise KeyError("Missing required field: training.scheduler.type")

        valid_schedulers = ["cosine", "step", "plateau", None]
        if scheduler["type"] not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler: {scheduler['type']}. Must be one of {valid_schedulers}"
            )

    elif config["mode"] == "evaluate":
        if "evaluation" not in config:
            raise KeyError(
                "Missing required section: evaluation (required for evaluate mode)"
            )

        evaluation = config["evaluation"]
        if "checkpoint" not in evaluation or evaluation["checkpoint"] is None:
            raise ValueError("evaluation.checkpoint is required for evaluate mode")


def is_v2_config(config: Dict[str, Any]) -> bool:
    """Check if configuration is in V2 format.

    Args:
        config: Configuration dictionary

    Returns:
        True if V2 format, False if V1 format
    """
    # V2 has 'compute' and 'mode' keys, V1 doesn't
    return "compute" in config and "mode" in config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate classifier configuration (auto-detects V1 or V2).

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing
    """
    if is_v2_config(config):
        validate_config_v2(config)
    else:
        # V1 config - issue deprecation warning
        warnings.warn(
            "Using V1 configuration format. Please migrate to V2 format. "
            "See docs/refactoring/20260214_classifier-config-optimization-plan.md",
            DeprecationWarning,
            stacklevel=2,
        )
        # Validate using original V1 validation
        _validate_config_v1(config)


def _validate_config_v1(config: Dict[str, Any]) -> None:
    """Validate V1 classifier configuration (legacy).

    Args:
        config: Configuration dictionary to validate (V1 format)

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing
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

    # Validate model configuration - strict required fields
    model = config["model"]
    required_model_fields = ["name", "pretrained", "num_classes", "freeze_backbone"]
    for field in required_model_fields:
        if field not in model:
            raise KeyError(f"Missing required field: model.{field}")
        if model[field] is None:
            raise ValueError(f"model.{field} cannot be None")

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

    # Validate data configuration - strict required fields
    data = config["data"]
    required_data_fields = [
        "train_path",
        "batch_size",
        "num_workers",
        "image_size",
        "crop_size",
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

    if not isinstance(data["crop_size"], int) or data["crop_size"] < 1:
        raise ValueError("crop_size must be a positive integer")

    valid_normalize = ["imagenet", "cifar10", "none", None]
    if data.get("normalize") not in valid_normalize:
        raise ValueError(
            f"Invalid normalize option: {data.get('normalize')}. "
            f"Must be one of {valid_normalize}"
        )

    # Validate training configuration - strict required fields
    training = config["training"]
    required_training_fields = ["epochs", "learning_rate", "optimizer", "device"]
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

    valid_optimizers = ["adam", "sgd", "adamw"]
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

    valid_devices = ["cuda", "cpu"]
    if training["device"] not in valid_devices:
        raise ValueError(
            f"Invalid device: {training['device']}. Must be one of {valid_devices}"
        )

    # Validate output configuration - strict required fields
    output = config["output"]
    required_output_fields = ["checkpoint_dir", "log_dir"]
    for field in required_output_fields:
        if field not in output:
            raise KeyError(f"Missing required field: output.{field}")
        if output[field] is None:
            raise ValueError(f"output.{field} cannot be None")
