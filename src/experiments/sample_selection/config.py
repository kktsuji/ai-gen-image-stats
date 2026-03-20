"""Sample Selection Configuration

This module provides configuration validation for the sample selection experiment.
Validates feature extraction, data sources, scoring, selection, and output settings.
Strict validation: all parameters must be explicitly specified in the config file.
"""

from typing import Any, Dict

from src.utils.config import (
    validate_compute_section,
    validate_experiment_section,
    validate_output_section,
)

VALID_FEATURE_MODELS = ["inceptionv3", "resnet50", "resnet101", "resnet152"]
VALID_REAL_SOURCES = ["split_file", "directory"]
VALID_SELECTION_MODES = ["top_k", "percentile", "threshold"]


def validate_config(config: Dict[str, Any]) -> None:
    """Validate sample selection configuration.

    Checks that all required fields are present and have valid values.
    Validation is mode-aware: select and evaluate modes have different
    required sections.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing
    """
    # Validate experiment type and mode
    validate_experiment_section(config, "sample_selection", ["select", "evaluate"])

    mode = config["mode"]

    # Validate compute section
    validate_compute_section(config)

    # Validate feature_extraction section (both modes)
    _validate_feature_extraction_section(config)

    # Validate logging section (both modes)
    _validate_logging_section(config)

    if mode == "select":
        # Validate output section
        validate_output_section(
            config, required_subdirs=["logs", "reports", "selected"]
        )

        # Validate data section (select mode)
        _validate_data_section_select(config)

        # Validate scoring section
        _validate_scoring_section(config)

        # Validate selection section
        _validate_selection_section(config)

        # Validate dataset_metrics section
        _validate_dataset_metrics_section(config)

    elif mode == "evaluate":
        # Validate output section (no selected subdir needed)
        validate_output_section(config, required_subdirs=["logs", "reports"])

        # Validate data section (evaluate mode)
        _validate_data_section_evaluate(config)

        # Validate evaluation section
        _validate_evaluation_section(config)


def _validate_feature_extraction_section(config: Dict[str, Any]) -> None:
    """Validate feature_extraction configuration section.

    Args:
        config: Full configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "feature_extraction" not in config:
        raise KeyError("Missing required config key: feature_extraction")

    fe = config["feature_extraction"]

    if "model" not in fe:
        raise KeyError("Missing required field: feature_extraction.model")
    if fe["model"] not in VALID_FEATURE_MODELS:
        raise ValueError(
            f"Invalid feature_extraction.model: '{fe['model']}'. "
            f"Must be one of {VALID_FEATURE_MODELS}"
        )

    if "batch_size" not in fe:
        raise KeyError("Missing required field: feature_extraction.batch_size")
    if (
        isinstance(fe["batch_size"], bool)
        or not isinstance(fe["batch_size"], int)
        or fe["batch_size"] < 1
    ):
        raise ValueError("feature_extraction.batch_size must be a positive integer")

    if "image_size" not in fe:
        raise KeyError("Missing required field: feature_extraction.image_size")
    if (
        isinstance(fe["image_size"], bool)
        or not isinstance(fe["image_size"], int)
        or fe["image_size"] < 1
    ):
        raise ValueError("feature_extraction.image_size must be a positive integer")

    if "num_workers" not in fe:
        raise KeyError("Missing required field: feature_extraction.num_workers")
    if (
        isinstance(fe["num_workers"], bool)
        or not isinstance(fe["num_workers"], int)
        or fe["num_workers"] < 0
    ):
        raise ValueError(
            "feature_extraction.num_workers must be a non-negative integer"
        )


def _validate_data_section_select(config: Dict[str, Any]) -> None:
    """Validate data configuration section for select mode.

    Args:
        config: Full configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "data" not in config:
        raise KeyError("Missing required config key: data")

    data = config["data"]

    # Validate data.real section
    if "real" not in data:
        raise KeyError("Missing required config key: data.real")

    real = data["real"]

    if "source" not in real:
        raise KeyError("Missing required field: data.real.source")
    if real["source"] not in VALID_REAL_SOURCES:
        raise ValueError(
            f"Invalid data.real.source: '{real['source']}'. "
            f"Must be one of {VALID_REAL_SOURCES}"
        )

    if real["source"] == "split_file":
        if "split_file" not in real:
            raise KeyError(
                "Missing required field: data.real.split_file "
                "(required when source='split_file')"
            )
        if not isinstance(real["split_file"], str) or not real["split_file"]:
            raise ValueError("data.real.split_file must be a non-empty string")

        if "split" not in real:
            raise KeyError(
                "Missing required field: data.real.split "
                "(required when source='split_file')"
            )
        if real["split"] not in ("train", "val"):
            raise ValueError(
                f"Invalid data.real.split: '{real['split']}'. Must be 'train' or 'val'"
            )

        if "class_label" not in real:
            raise KeyError(
                "Missing required field: data.real.class_label "
                "(required when source='split_file')"
            )
        class_label = real["class_label"]
        if class_label is not None:
            if not isinstance(class_label, int) or isinstance(class_label, bool):
                raise ValueError(
                    f"data.real.class_label must be null or a non-negative integer, "
                    f"got {type(class_label).__name__}: {class_label!r}"
                )
            if class_label < 0:
                raise ValueError(
                    f"data.real.class_label must be null or a non-negative integer, "
                    f"got {class_label}"
                )

    elif real["source"] == "directory":
        if "directory" not in real:
            raise KeyError(
                "Missing required field: data.real.directory "
                "(required when source='directory')"
            )
        if not isinstance(real["directory"], str) or not real["directory"]:
            raise ValueError("data.real.directory must be a non-empty string")

    # Validate data.generated section
    if "generated" not in data:
        raise KeyError("Missing required config key: data.generated")

    generated = data["generated"]

    if "directory" not in generated:
        raise KeyError("Missing required field: data.generated.directory")
    if not isinstance(generated["directory"], str) or not generated["directory"]:
        raise ValueError("data.generated.directory must be a non-empty string")

    # Validate label and class_name for output metadata
    if "label" not in data:
        raise KeyError("Missing required field: data.label")
    if (
        isinstance(data["label"], bool)
        or not isinstance(data["label"], int)
        or data["label"] < 0
    ):
        raise ValueError("data.label must be a non-negative integer")

    if "class_name" not in data:
        raise KeyError("Missing required field: data.class_name")
    if not isinstance(data["class_name"], str) or not data["class_name"]:
        raise ValueError("data.class_name must be a non-empty string")


def _validate_scoring_section(config: Dict[str, Any]) -> None:
    """Validate scoring configuration section.

    Args:
        config: Full configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "scoring" not in config:
        raise KeyError("Missing required config key: scoring")

    scoring = config["scoring"]

    if "k" not in scoring:
        raise KeyError("Missing required field: scoring.k")
    if (
        isinstance(scoring["k"], bool)
        or not isinstance(scoring["k"], int)
        or scoring["k"] < 1
    ):
        raise ValueError("scoring.k must be a positive integer")

    if "require_realism" not in scoring:
        raise KeyError("Missing required field: scoring.require_realism")
    if not isinstance(scoring["require_realism"], bool):
        raise ValueError("scoring.require_realism must be a boolean")


def _validate_selection_section(config: Dict[str, Any]) -> None:
    """Validate selection configuration section.

    Args:
        config: Full configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "selection" not in config:
        raise KeyError("Missing required config key: selection")

    selection = config["selection"]

    if "mode" not in selection:
        raise KeyError("Missing required field: selection.mode")
    if selection["mode"] not in VALID_SELECTION_MODES:
        raise ValueError(
            f"Invalid selection.mode: '{selection['mode']}'. "
            f"Must be one of {VALID_SELECTION_MODES}"
        )

    if "value" not in selection:
        raise KeyError("Missing required field: selection.value")

    value = selection["value"]
    mode = selection["mode"]

    if mode == "top_k":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "selection.value must be a positive integer for top_k mode"
            )
    elif mode == "percentile":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            or value > 100
        ):
            raise ValueError(
                "selection.value must be a number in (0, 100] for percentile mode"
            )
    elif mode == "threshold":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                "selection.value must be a positive number for threshold mode"
            )


def _validate_dataset_metrics_section(config: Dict[str, Any]) -> None:
    """Validate dataset_metrics configuration section.

    Args:
        config: Full configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "dataset_metrics" not in config:
        raise KeyError("Missing required config key: dataset_metrics")

    dm = config["dataset_metrics"]

    if "enabled" not in dm:
        raise KeyError("Missing required field: dataset_metrics.enabled")
    if not isinstance(dm["enabled"], bool):
        raise ValueError("dataset_metrics.enabled must be a boolean")


def _validate_logging_section(config: Dict[str, Any]) -> None:
    """Validate logging configuration section.

    Args:
        config: Full configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "logging" not in config:
        raise KeyError("Missing required config key: logging")

    log = config["logging"]

    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    if "console_level" not in log:
        raise KeyError("Missing required field: logging.console_level")
    if log["console_level"] not in valid_log_levels:
        raise ValueError(
            f"Invalid logging.console_level: '{log['console_level']}'. "
            f"Must be one of {valid_log_levels}"
        )
    if "file_level" not in log:
        raise KeyError("Missing required field: logging.file_level")
    if log["file_level"] not in valid_log_levels:
        raise ValueError(
            f"Invalid logging.file_level: '{log['file_level']}'. "
            f"Must be one of {valid_log_levels}"
        )


def _validate_data_section_evaluate(config: Dict[str, Any]) -> None:
    """Validate data configuration section for evaluate mode.

    Requires data.real and data.generated (same rules as select mode for
    those subsections). Does NOT require data.label or data.class_name.
    Optional data.selected: if present, requires split_file and split.

    Args:
        config: Full configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "data" not in config:
        raise KeyError("Missing required config key: data")

    data = config["data"]

    # Validate data.real section (same rules as select mode)
    if "real" not in data:
        raise KeyError("Missing required config key: data.real")

    real = data["real"]

    if "source" not in real:
        raise KeyError("Missing required field: data.real.source")
    if real["source"] not in VALID_REAL_SOURCES:
        raise ValueError(
            f"Invalid data.real.source: '{real['source']}'. "
            f"Must be one of {VALID_REAL_SOURCES}"
        )

    if real["source"] == "split_file":
        if "split_file" not in real:
            raise KeyError(
                "Missing required field: data.real.split_file "
                "(required when source='split_file')"
            )
        if not isinstance(real["split_file"], str) or not real["split_file"]:
            raise ValueError("data.real.split_file must be a non-empty string")

        if "split" not in real:
            raise KeyError(
                "Missing required field: data.real.split "
                "(required when source='split_file')"
            )
        if real["split"] not in ("train", "val"):
            raise ValueError(
                f"Invalid data.real.split: '{real['split']}'. Must be 'train' or 'val'"
            )

        if "class_label" not in real:
            raise KeyError(
                "Missing required field: data.real.class_label "
                "(required when source='split_file')"
            )
        class_label = real["class_label"]
        if class_label is not None:
            if not isinstance(class_label, int) or isinstance(class_label, bool):
                raise ValueError(
                    f"data.real.class_label must be null or a non-negative integer, "
                    f"got {type(class_label).__name__}: {class_label!r}"
                )
            if class_label < 0:
                raise ValueError(
                    f"data.real.class_label must be null or a non-negative integer, "
                    f"got {class_label}"
                )

    elif real["source"] == "directory":
        if "directory" not in real:
            raise KeyError(
                "Missing required field: data.real.directory "
                "(required when source='directory')"
            )
        if not isinstance(real["directory"], str) or not real["directory"]:
            raise ValueError("data.real.directory must be a non-empty string")

    # Validate data.generated section
    if "generated" not in data:
        raise KeyError("Missing required config key: data.generated")

    generated = data["generated"]

    if "directory" not in generated:
        raise KeyError("Missing required field: data.generated.directory")
    if not isinstance(generated["directory"], str) or not generated["directory"]:
        raise ValueError("data.generated.directory must be a non-empty string")

    # Validate optional data.selected section
    if "selected" in data:
        selected = data["selected"]

        if "split_file" not in selected:
            raise KeyError("Missing required field: data.selected.split_file")
        if not isinstance(selected["split_file"], str) or not selected["split_file"]:
            raise ValueError("data.selected.split_file must be a non-empty string")

        if "split" not in selected:
            raise KeyError("Missing required field: data.selected.split")
        if selected["split"] != "train":
            raise ValueError(
                f"Invalid data.selected.split: '{selected['split']}'. Must be 'train'"
            )


def _validate_evaluation_section(config: Dict[str, Any]) -> None:
    """Validate evaluation configuration section.

    Requires evaluation.k as a positive integer.

    Args:
        config: Full configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "evaluation" not in config:
        raise KeyError("Missing required config key: evaluation")

    evaluation = config["evaluation"]

    if "k" not in evaluation:
        raise KeyError("Missing required field: evaluation.k")
    if (
        isinstance(evaluation["k"], bool)
        or not isinstance(evaluation["k"], int)
        or evaluation["k"] < 1
    ):
        raise ValueError("evaluation.k must be a positive integer")
