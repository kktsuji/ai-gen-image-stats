"""Configuration management utilities.

This module provides utilities for loading and merging configuration files.
All parameters must be explicitly specified in the config file (strict mode).
Priority order: CLI overrides > Config file
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_config(config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the config file is not valid YAML
        ValueError: If config_path is empty or None
    """
    if not config_path:
        raise ValueError("Config path cannot be empty or None")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(
    base_config: Optional[Dict[str, Any]], override_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively.

    The override_config takes precedence over base_config. Nested dictionaries
    are merged recursively. Lists and other types are replaced entirely.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary (takes precedence)

    Returns:
        Merged configuration dictionary

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 99}, "e": 4}
        >>> merge_configs(base, override)
        {"a": 1, "b": {"c": 99, "d": 3}, "e": 4}
    """
    if base_config is None:
        return override_config.copy() if override_config else {}

    if override_config is None:
        return base_config.copy() if base_config else {}

    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override value (for non-dict types or new keys)
            result[key] = value

    return result


def load_and_merge_configs(
    config_path: Optional[str] = None,
    defaults: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a config file and merge it with defaults and overrides.

    Priority order: overrides > config_file > defaults

    Args:
        config_path: Optional path to YAML configuration file
        defaults: Optional dictionary of default values
        overrides: Optional dictionary of override values (highest priority)

    Returns:
        Merged configuration dictionary

    Example:
        >>> defaults = {"epochs": 10, "batch_size": 32}
        >>> config = load_and_merge_configs(
        ...     config_path="experiment.yaml",
        ...     defaults=defaults,
        ...     overrides={"batch_size": 64}
        ... )
    """
    # Start with defaults
    result = defaults.copy() if defaults else {}

    # Merge config file if provided
    if config_path:
        file_config = load_config(config_path)
        result = merge_configs(result, file_config)

    # Apply overrides (highest priority)
    if overrides:
        result = merge_configs(result, overrides)

    return result


def save_config(
    config: Optional[Dict[str, Any]],
    output_path: Optional[Union[str, Path]],
    indent: int = 2,
) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the YAML file
        indent: Number of spaces for indentation (default: 2)

    Raises:
        ValueError: If config is None or output_path is empty
    """
    if config is None:
        raise ValueError("Config cannot be None")

    if not output_path:
        raise ValueError("Output path cannot be empty or None")

    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=indent)


# ==============================================================================
# Configuration Helper Functions
# ==============================================================================


def resolve_output_path(config: Dict[str, Any], subdir_key: str) -> Path:
    """Resolve output path from base_dir + subdirs.

    Args:
        config: Full configuration dictionary
        subdir_key: Key in output.subdirs (e.g., "logs", "checkpoints")

    Returns:
        Resolved Path object

    Raises:
        KeyError: If required keys are missing

    Example:
        >>> config = {"output": {"base_dir": "outputs", "subdirs": {"logs": "logs"}}}
        >>> resolve_output_path(config, "logs")
        Path("outputs/logs")
    """
    if "output" not in config:
        raise KeyError("Missing required config key: output")

    output = config["output"]

    if "base_dir" not in output:
        raise KeyError("Missing required config key: output.base_dir")

    if "subdirs" not in output:
        raise KeyError("Missing required config key: output.subdirs")

    if subdir_key not in output["subdirs"]:
        raise KeyError(f"Missing required config key: output.subdirs.{subdir_key}")

    base_dir = Path(output["base_dir"])
    subdir = output["subdirs"][subdir_key]
    return base_dir / subdir


def derive_image_size_from_model(config: Dict[str, Any]) -> int:
    """Derive image_size from model configuration.

    The image_size is defined in model.architecture.

    Args:
        config: Full configuration dictionary

    Returns:
        Image size from model.architecture.image_size

    Raises:
        KeyError: If required keys are missing

    Example:
        >>> config = {"model": {"architecture": {"image_size": 64}}}
        >>> derive_image_size_from_model(config)
        64
    """
    if "model" not in config:
        raise KeyError("Missing required config key: model")

    if "architecture" not in config["model"]:
        raise KeyError("Missing required config key: model.architecture")

    if "image_size" not in config["model"]["architecture"]:
        raise KeyError("Missing required config key: model.architecture.image_size")

    return config["model"]["architecture"]["image_size"]


def derive_return_labels_from_model(config: Dict[str, Any]) -> bool:
    """Derive return_labels from model conditioning configuration (V2).

    In V2 config, return_labels is derived from conditioning.type rather than
    being duplicated in the data section.

    Args:
        config: Full configuration dictionary (V2 format)

    Returns:
        True if model uses class conditioning, False otherwise

    Example:
        >>> config = {"model": {"conditioning": {"type": "class"}}}
        >>> derive_return_labels_from_model(config)
        True
        >>> config = {"model": {"conditioning": {"type": None}}}
        >>> derive_return_labels_from_model(config)
        False
    """
    if "model" not in config:
        return False

    if "conditioning" not in config["model"]:
        return False

    conditioning_type = config["model"]["conditioning"].get("type")
    return conditioning_type == "class"


# ==============================================================================
# Common Configuration Validation Functions
# ==============================================================================


def validate_compute_section(
    config: Dict[str, Any], valid_devices: Optional[List[str]] = None
) -> None:
    """Validate compute configuration section.

    Args:
        config: Full configuration dictionary
        valid_devices: List of valid device values (default: ["cuda", "cpu", "auto"])

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "compute" not in config:
        raise KeyError("Missing required config key: compute")

    compute = config["compute"]

    # Validate device
    if valid_devices is None:
        valid_devices = ["cuda", "cpu", "auto"]

    device = compute.get("device", "cuda")
    if device not in valid_devices:
        raise ValueError(f"Invalid device: {device}. Must be one of {valid_devices}")

    # Validate seed (if present)
    seed = compute.get("seed")
    if seed is not None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("compute.seed must be None or a non-negative integer")


def validate_output_section(
    config: Dict[str, Any], required_subdirs: Optional[List[str]] = None
) -> None:
    """Validate output configuration section.

    Args:
        config: Full configuration dictionary
        required_subdirs: List of required subdirectory keys (default: ["logs", "checkpoints"])

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "output" not in config:
        raise KeyError("Missing required config key: output")

    output = config["output"]

    if "base_dir" not in output or output["base_dir"] is None:
        raise ValueError("output.base_dir is required and cannot be None")

    if "subdirs" not in output:
        raise KeyError("Missing required config key: output.subdirs")

    subdirs = output["subdirs"]

    if required_subdirs is None:
        required_subdirs = ["logs", "checkpoints"]

    for subdir in required_subdirs:
        if subdir not in subdirs or subdirs[subdir] is None:
            raise ValueError(f"output.subdirs.{subdir} is required and cannot be None")


def validate_data_loading_section(data_config: Dict[str, Any]) -> None:
    """Validate data.loading configuration section.

    Args:
        data_config: The data section of configuration dictionary

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "loading" not in data_config:
        raise KeyError("Missing required config key: data.loading")

    loading = data_config["loading"]
    required_loading_fields = ["batch_size", "num_workers"]

    for field in required_loading_fields:
        if field not in loading:
            raise KeyError(f"Missing required field: data.loading.{field}")
        if loading[field] is None:
            raise ValueError(f"data.loading.{field} cannot be None")

    if not isinstance(loading["batch_size"], int) or loading["batch_size"] < 1:
        raise ValueError("data.loading.batch_size must be a positive integer")

    if not isinstance(loading["num_workers"], int) or loading["num_workers"] < 0:
        raise ValueError("data.loading.num_workers must be a non-negative integer")


def validate_optimizer_section(
    optimizer_config: Dict[str, Any], valid_types: Optional[List[str]] = None
) -> None:
    """Validate training.optimizer configuration section.

    Args:
        optimizer_config: The optimizer section of configuration dictionary
        valid_types: List of valid optimizer types (default: ["adam", "adamw", "sgd"])

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "type" not in optimizer_config:
        raise KeyError("Missing required field: optimizer.type")

    if "learning_rate" not in optimizer_config:
        raise KeyError("Missing required field: optimizer.learning_rate")

    if valid_types is None:
        valid_types = ["adam", "adamw", "sgd"]

    opt_type = optimizer_config["type"]
    if opt_type not in valid_types:
        raise ValueError(
            f"Invalid optimizer type: {opt_type}. Must be one of {valid_types}"
        )

    learning_rate = optimizer_config["learning_rate"]
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("optimizer.learning_rate must be a positive number")

    # Validate gradient_clip_norm if present
    if "gradient_clip_norm" in optimizer_config:
        grad_clip = optimizer_config["gradient_clip_norm"]
        if grad_clip is not None:
            if not isinstance(grad_clip, (int, float)) or grad_clip <= 0:
                raise ValueError(
                    "optimizer.gradient_clip_norm must be a positive number or None"
                )


def validate_scheduler_section(
    scheduler_config: Dict[str, Any],
    valid_types: Optional[List[Optional[str]]] = None,
) -> None:
    """Validate training.scheduler configuration section.

    Args:
        scheduler_config: The scheduler section of configuration dictionary
        valid_types: List of valid scheduler types (default: ["cosine", "step", "plateau", None])

    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    if "type" not in scheduler_config:
        raise KeyError("Missing required field: scheduler.type")

    if valid_types is None:
        valid_types = ["cosine", "step", "plateau", None, "none"]

    sched_type = scheduler_config["type"]
    normalized = sched_type.lower() if isinstance(sched_type, str) else sched_type
    if normalized not in valid_types:
        raise ValueError(
            f"Invalid scheduler type: {sched_type}. Must be one of {valid_types}"
        )


def validate_checkpointing_section(checkpointing_config: Dict[str, Any]) -> None:
    """Validate training.checkpointing configuration section.

    Args:
        checkpointing_config: The checkpointing section of configuration dictionary

    Raises:
        ValueError: If values are invalid
    """
    if "save_frequency" in checkpointing_config:
        sf = checkpointing_config["save_frequency"]
        if isinstance(sf, bool) or not isinstance(sf, int) or sf < 1:
            raise ValueError(
                "training.checkpointing.save_frequency must be a positive integer"
            )
    if "save_latest" in checkpointing_config:
        if not isinstance(checkpointing_config["save_latest"], bool):
            raise ValueError("training.checkpointing.save_latest must be a boolean")
    if "save_best_only" in checkpointing_config:
        if not isinstance(checkpointing_config["save_best_only"], bool):
            raise ValueError("training.checkpointing.save_best_only must be a boolean")


def validate_validation_section(validation_config: Dict[str, Any]) -> None:
    """Validate training.validation configuration section.

    Args:
        validation_config: The validation section of configuration dictionary

    Raises:
        ValueError: If values are invalid
    """
    if "frequency" in validation_config:
        freq = validation_config["frequency"]
        if isinstance(freq, bool) or not isinstance(freq, int) or freq < 1:
            raise ValueError("training.validation.frequency must be a positive integer")
    if "metric" in validation_config:
        if not isinstance(validation_config["metric"], str):
            raise ValueError("training.validation.metric must be a string")


def validate_training_epochs(training_config: Dict[str, Any]) -> None:
    """Validate training.epochs configuration.

    Args:
        training_config: The training section of configuration dictionary

    Raises:
        ValueError: If epochs is missing, None, or invalid
    """
    if "epochs" not in training_config or training_config["epochs"] is None:
        raise ValueError("training.epochs is required and cannot be None")

    if (
        isinstance(training_config["epochs"], bool)
        or not isinstance(training_config["epochs"], int)
        or training_config["epochs"] < 1
    ):
        raise ValueError("training.epochs must be a positive integer")


def validate_split_file(data_config: Dict[str, Any]) -> None:
    """Validate data.split_file configuration.

    Args:
        data_config: The data section of configuration dictionary

    Raises:
        ValueError: If split_file is missing, None, or invalid
    """
    if "split_file" not in data_config or data_config["split_file"] is None:
        raise ValueError("data.split_file is required and cannot be None")

    if not isinstance(data_config["split_file"], str) or not data_config["split_file"]:
        raise ValueError("data.split_file must be a non-empty string")


def validate_experiment_section(
    config: Dict[str, Any],
    expected_type: str,
    valid_modes: List[str],
) -> None:
    """Validate the experiment type and mode fields in a config dict.

    Args:
        config: The merged configuration dictionary.
        expected_type: The expected value of config["experiment"].
        valid_modes: List of allowed values for config["mode"].

    Raises:
        ValueError: If experiment type or mode is invalid.
    """
    if config.get("experiment") != expected_type:
        raise ValueError(
            f"Invalid experiment type: '{config.get('experiment')}'. "
            f"Expected '{expected_type}'."
        )
    if config.get("mode") not in valid_modes:
        raise ValueError(
            f"Invalid mode: '{config.get('mode')}'. Valid modes: {valid_modes}"
        )
