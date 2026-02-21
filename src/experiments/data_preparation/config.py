"""Data Preparation Configuration

This module provides configuration validation for the data preparation experiment.
Validates class directories, split parameters, and output settings.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from src.utils.config import get_default_config_from_module

logger = logging.getLogger(__name__)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration by loading default.yaml.

    Returns:
        Dictionary containing default configuration values from YAML file

    Raises:
        FileNotFoundError: If default.yaml is not found
        yaml.YAMLError: If YAML is invalid
    """
    return get_default_config_from_module(__file__)


def validate_config(config: Dict[str, Any]) -> None:
    """Validate data preparation configuration.

    Checks that all required fields are present and have valid values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing
    """
    # Validate experiment type
    if config.get("experiment") != "data_preparation":
        raise ValueError(
            f"Invalid experiment type: {config.get('experiment')}. "
            "Must be 'data_preparation'"
        )

    # Validate classes section
    if "classes" not in config:
        raise KeyError("Missing required config key: classes")

    classes = config["classes"]
    if not isinstance(classes, dict) or len(classes) == 0:
        raise ValueError("'classes' must be a non-empty dictionary")

    for class_name, class_path in classes.items():
        if not isinstance(class_path, str) or not class_path:
            raise ValueError(f"Class '{class_name}' must have a non-empty string path")

    # Validate split section
    if "split" not in config:
        raise KeyError("Missing required config key: split")

    split = config["split"]

    # Validate seed (int or null)
    if "seed" in split and split["seed"] is not None:
        if not isinstance(split["seed"], int):
            raise ValueError("split.seed must be an integer or null")

    # Validate train_ratio
    if "train_ratio" not in split:
        raise KeyError("Missing required field: split.train_ratio")

    train_ratio = split["train_ratio"]
    if not isinstance(train_ratio, (int, float)):
        raise ValueError("split.train_ratio must be a number")
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError(
            f"split.train_ratio must be between 0.0 and 1.0 (exclusive), "
            f"got {train_ratio}"
        )

    # Validate save_dir
    if "save_dir" not in split:
        raise KeyError("Missing required field: split.save_dir")
    if not isinstance(split["save_dir"], str) or not split["save_dir"]:
        raise ValueError("split.save_dir must be a non-empty string")

    # Validate split_file
    if "split_file" not in split:
        raise KeyError("Missing required field: split.split_file")
    if not isinstance(split["split_file"], str) or not split["split_file"]:
        raise ValueError("split.split_file must be a non-empty string")
    if not split["split_file"].endswith(".json"):
        raise ValueError("split.split_file must end with '.json'")

    # Validate force (optional, defaults to false)
    if "force" in split and not isinstance(split["force"], bool):
        raise ValueError("split.force must be a boolean")
