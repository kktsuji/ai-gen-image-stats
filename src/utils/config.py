"""Configuration management utilities.

This module provides utilities for loading and merging configuration files.
Priority order: CLI arguments > Config file > Code defaults
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
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
    base_config: Dict[str, Any], override_config: Dict[str, Any]
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


def save_config(config: Dict[str, Any], output_path: str, indent: int = 2) -> None:
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
