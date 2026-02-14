"""Classifier Configuration

This module provides default configuration values for classifier experiments.
It defines sensible defaults for training, data loading, model selection, and logging.

Default values are loaded from YAML as the single source of truth.
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

    Returns:
        Dictionary containing default configuration values from YAML file

    Raises:
        FileNotFoundError: If default.yaml is not found
        yaml.YAMLError: If YAML is invalid

    Example:
        >>> config = get_default_config()
        >>> print(config["training"]["epochs"])
        100
    """
    return get_default_config_from_module(__file__)


# Note: validate_config() is defined later in this file.


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


def validate_config(config: Dict[str, Any]) -> None:
    """Validate classifier configuration.

    Checks that all required fields are present and have valid values.

    Args:
        config: Configuration dictionary to validate

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
    validate_compute_section(config)

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
    validate_data_loading_section(data)

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
    validate_output_section(config)

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

        validate_optimizer_section(optimizer, valid_types=["adam", "adamw", "sgd"])

        # Validate scheduler
        validate_scheduler_section(training["scheduler"])

    elif config["mode"] == "evaluate":
        if "evaluation" not in config:
            raise KeyError(
                "Missing required section: evaluation (required for evaluate mode)"
            )

        evaluation = config["evaluation"]
        if "checkpoint" not in evaluation or evaluation["checkpoint"] is None:
            raise ValueError("evaluation.checkpoint is required for evaluate mode")
