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
        unexpected_keys = set(class_entry) - {"path", "label"}
        if unexpected_keys:
            raise ValueError(
                f"Class '{class_name}' has unsupported keys: {sorted(unexpected_keys)}. "
                "Only 'path' and 'label' are allowed"
            )
        if "path" not in class_entry:
            raise KeyError(f"Class '{class_name}' missing required key: 'path'")
        if not isinstance(class_entry["path"], str) or not class_entry["path"]:
            raise ValueError(f"Class '{class_name}' path must be a non-empty string")
        if "label" not in class_entry:
            raise KeyError(f"Class '{class_name}' missing required key: 'label'")
        if (
            isinstance(class_entry["label"], bool)
            or not isinstance(class_entry["label"], int)
            or class_entry["label"] < 0
        ):
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

    # Validate split mode (optional, defaults to the historical ratio split).
    # "kfold" enables repeated stratified k-fold cross-validation, emitting one
    # split JSON per (repeat, fold) for honest cross-split confidence intervals.
    mode = split.get("mode", "ratio")
    if mode not in ("ratio", "kfold"):
        raise ValueError(f"split.mode must be 'ratio' or 'kfold', got {mode!r}")

    # Validate save_dir (shared by both modes)
    if "save_dir" not in split:
        raise KeyError("Missing required field: split.save_dir")
    if not isinstance(split["save_dir"], str) or not split["save_dir"]:
        raise ValueError("split.save_dir must be a non-empty string")

    # Validate force (optional, defaults to false; shared by both modes)
    if "force" in split and not isinstance(split["force"], bool):
        raise ValueError("split.force must be a boolean")

    if mode == "kfold":
        _validate_kfold_split(split)
        return

    _validate_ratio_split(split)


def _validate_ratio_split(split: Dict[str, Any]) -> None:
    """Validate the classic single train/val/test ratio split."""
    # Validate seed (int or null)
    if "seed" in split and split["seed"] is not None:
        if not isinstance(split["seed"], int):
            raise ValueError("split.seed must be an integer or null")

    # Validate train_ratio (must be > 0, samples are required for training)
    if "train_ratio" not in split:
        raise KeyError("Missing required field: split.train_ratio")
    train_ratio = split["train_ratio"]
    if not isinstance(train_ratio, (int, float)) or isinstance(train_ratio, bool):
        raise ValueError("split.train_ratio must be a number")
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError(
            f"split.train_ratio must be between 0.0 and 1.0 (exclusive), "
            f"got {train_ratio}"
        )

    # Validate val_ratio (must be > 0, samples are required for validation)
    if "val_ratio" not in split:
        raise KeyError("Missing required field: split.val_ratio")
    val_ratio = split["val_ratio"]
    if not isinstance(val_ratio, (int, float)) or isinstance(val_ratio, bool):
        raise ValueError("split.val_ratio must be a number")
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        raise ValueError(
            f"split.val_ratio must be between 0.0 and 1.0 (exclusive), got {val_ratio}"
        )

    # Validate test_ratio (may be 0.0 to fall back to a train/val-only split)
    if "test_ratio" not in split:
        raise KeyError("Missing required field: split.test_ratio")
    test_ratio = split["test_ratio"]
    if not isinstance(test_ratio, (int, float)) or isinstance(test_ratio, bool):
        raise ValueError("split.test_ratio must be a number")
    if test_ratio < 0.0 or test_ratio >= 1.0:
        raise ValueError(
            f"split.test_ratio must be between 0.0 (inclusive) and 1.0 "
            f"(exclusive), got {test_ratio}"
        )

    # The three ratios must sum to 1.0 (small floating-point tolerance allowed)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "split.train_ratio + split.val_ratio + split.test_ratio must sum "
            f"to 1.0, got {ratio_sum}"
        )

    # Validate split_file
    if "split_file" not in split:
        raise KeyError("Missing required field: split.split_file")
    if not isinstance(split["split_file"], str) or not split["split_file"]:
        raise ValueError("split.split_file must be a non-empty string")
    if not split["split_file"].endswith(".json"):
        raise ValueError("split.split_file must end with '.json'")


def _validate_kfold_split(split: Dict[str, Any]) -> None:
    """Validate the repeated stratified k-fold split configuration.

    Emits ``n_folds * n_repeats`` split JSONs. Within each repeat the test folds
    partition the data (every sample is tested exactly once), so the across-split
    spread of metrics reflects honest data-resampling variance rather than only
    weight-initialisation noise.
    """
    # n_folds: integer >= 2
    if "n_folds" not in split:
        raise KeyError("Missing required field: split.n_folds (mode=kfold)")
    n_folds = split["n_folds"]
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("split.n_folds must be an integer >= 2")

    # n_repeats: integer >= 1
    if "n_repeats" not in split:
        raise KeyError("Missing required field: split.n_repeats (mode=kfold)")
    n_repeats = split["n_repeats"]
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("split.n_repeats must be an integer >= 1")

    # repeat_seeds: explicit list of ints, one per repeat (reproducibility)
    if "repeat_seeds" not in split:
        raise KeyError("Missing required field: split.repeat_seeds (mode=kfold)")
    repeat_seeds = split["repeat_seeds"]
    if not isinstance(repeat_seeds, list) or len(repeat_seeds) != n_repeats:
        raise ValueError(
            "split.repeat_seeds must be a list of length n_repeats "
            f"({n_repeats}), got {repeat_seeds!r}"
        )
    for s in repeat_seeds:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError("split.repeat_seeds entries must be integers")
    if len(set(repeat_seeds)) != len(repeat_seeds):
        raise ValueError("split.repeat_seeds entries must be unique")

    # val_fraction: float in (0, 1) — carved from each fold's training pool
    if "val_fraction" not in split:
        raise KeyError("Missing required field: split.val_fraction (mode=kfold)")
    val_fraction = split["val_fraction"]
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)):
        raise ValueError("split.val_fraction must be a number")
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError(
            f"split.val_fraction must be between 0.0 and 1.0 (exclusive), "
            f"got {val_fraction}"
        )

    # split_file must template the split index so each (repeat, fold) gets a file
    if "split_file" not in split:
        raise KeyError("Missing required field: split.split_file")
    split_file = split["split_file"]
    if not isinstance(split_file, str) or not split_file:
        raise ValueError("split.split_file must be a non-empty string")
    if not split_file.endswith(".json"):
        raise ValueError("split.split_file must end with '.json'")
    if "{index}" not in split_file:
        raise ValueError(
            "split.split_file must contain the '{index}' placeholder in mode=kfold "
            f"(it is expanded to 0..{n_folds * n_repeats - 1}), got {split_file!r}"
        )
