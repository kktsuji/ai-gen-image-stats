"""Classifier Configuration

This module provides configuration validation for classifier experiments.
Strict validation: all parameters must be explicitly specified in the config file.
"""

from typing import Any, Dict

from src.utils.config import (
    validate_checkpointing_section,
    validate_compute_section,
    validate_data_loading_section,
    validate_experiment_section,
    validate_optimizer_section,
    validate_output_section,
    validate_scheduler_section,
    validate_split_file,
    validate_training_epochs,
)

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

    # Validate experiment type and mode
    validate_experiment_section(config, "classifier", ["train", "evaluate"])

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
    required_data_sections = ["split_file", "loading", "preprocessing", "augmentation"]
    for section in required_data_sections:
        if section not in data:
            raise KeyError(f"Missing required field: data.{section}")

    # Validate split_file
    validate_split_file(data)

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

        validate_training_epochs(training)

        # Validate optimizer
        optimizer = training["optimizer"]
        if "type" not in optimizer:
            raise KeyError("Missing required field: training.optimizer.type")
        if "learning_rate" not in optimizer:
            raise KeyError("Missing required field: training.optimizer.learning_rate")

        validate_optimizer_section(optimizer, valid_types=["adam", "adamw", "sgd"])

        # Validate scheduler
        validate_scheduler_section(training["scheduler"])

        # Validate checkpointing
        if "checkpointing" in training:
            validate_checkpointing_section(training["checkpointing"])

    # Validate synthetic_augmentation (optional section)
    syn_aug = data.get("synthetic_augmentation", {})
    if syn_aug.get("enabled"):
        if not isinstance(syn_aug.get("split_file"), str) or not syn_aug["split_file"]:
            raise ValueError(
                "data.synthetic_augmentation.split_file must be a non-empty string "
                "when synthetic_augmentation is enabled"
            )

        limit = syn_aug.get("limit", {})
        limit_mode = limit.get("mode")
        valid_limit_modes = [None, "max_ratio", "max_samples"]
        if limit_mode not in valid_limit_modes:
            raise ValueError(
                f"Invalid synthetic_augmentation limit.mode: {limit_mode}. "
                f"Must be one of {valid_limit_modes}"
            )

        if limit_mode == "max_ratio":
            max_ratio = limit.get("max_ratio")
            if not isinstance(max_ratio, (int, float)) or max_ratio <= 0:
                raise ValueError(
                    "synthetic_augmentation limit.max_ratio must be a positive number"
                )

        if limit_mode == "max_samples":
            max_samples = limit.get("max_samples")
            if not isinstance(max_samples, int) or max_samples <= 0:
                raise ValueError(
                    "synthetic_augmentation limit.max_samples must be a positive integer"
                )

    elif config["mode"] == "evaluate":
        if "evaluation" not in config:
            raise KeyError(
                "Missing required section: evaluation (required for evaluate mode)"
            )

        evaluation = config["evaluation"]
        if "checkpoint" not in evaluation or evaluation["checkpoint"] is None:
            raise ValueError("evaluation.checkpoint is required for evaluate mode")
