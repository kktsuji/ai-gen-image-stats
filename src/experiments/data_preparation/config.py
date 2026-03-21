"""Data Preparation Configuration

This module provides configuration validation for the data preparation experiment.
Validates class directories, split parameters, and output settings.
Strict validation: all parameters must be explicitly specified in the config file.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


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

    for class_name, class_entry in classes.items():
        if not isinstance(class_entry, dict):
            raise ValueError(
                f"Class '{class_name}' must be a dict with 'path' and 'label' keys"
            )
        if "path" not in class_entry:
            raise KeyError(f"Class '{class_name}' missing required key: 'path'")
        if not isinstance(class_entry["path"], str) or not class_entry["path"]:
            raise ValueError(f"Class '{class_name}' path must be a non-empty string")
        if "label" not in class_entry:
            raise KeyError(f"Class '{class_name}' missing required key: 'label'")
        if not isinstance(class_entry["label"], int) or class_entry["label"] < 0:
            raise ValueError(
                f"Class '{class_name}' label must be a non-negative integer"
            )

    labels = [entry["label"] for entry in classes.values()]
    if len(set(labels)) != len(labels):
        raise ValueError("Class labels must be unique")
    expected = set(range(len(labels)))
    if set(labels) != expected:
        raise ValueError(
            f"Class labels must form a contiguous range 0..{len(labels) - 1}, "
            f"got {sorted(labels)}"
        )

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
