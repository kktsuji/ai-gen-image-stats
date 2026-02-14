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
# V2 Configuration Helper Functions
# ==============================================================================


def resolve_output_path(config: Dict[str, Any], subdir_key: str) -> Path:
    """Resolve output path from base_dir + subdirs (V2 config).

    Args:
        config: Full configuration dictionary (V2 format)
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
    """Derive image_size from model configuration (V2).

    In V2 config, image_size is only defined in model.architecture,
    eliminating the duplication that existed in V1.

    Args:
        config: Full configuration dictionary (V2 format)

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


def migrate_config_v1_to_v2(v1_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate V1 configuration to V2 structure.

    Provides backward compatibility by automatically converting V1 configs.
    A warning is issued to encourage migration.

    Args:
        v1_config: Configuration in V1 format

    Returns:
        Configuration in V2 format

    Raises:
        ValueError: If v1_config structure is invalid

    Example:
        >>> v1 = {"device": "cuda", "model": {"image_size": 64, ...}}
        >>> v2 = migrate_config_v1_to_v2(v1)
        >>> print(v2["compute"]["device"])
        cuda
    """
    import warnings

    # Check if already V2 (has compute key)
    if "compute" in v1_config:
        warnings.warn("Configuration appears to be V2 format already")
        return v1_config.copy()

    warnings.warn(
        "Using V1 configuration format. Please migrate to V2 format. "
        "See docs/research/diffusion-config-migration-guide.md",
        DeprecationWarning,
        stacklevel=2,
    )

    v2_config = {
        "experiment": v1_config.get("experiment", "diffusion"),
        "mode": v1_config.get("mode", "train"),
    }

    # Migrate compute
    v2_config["compute"] = {
        "device": v1_config.get("device", "cuda"),
        "seed": v1_config.get("seed"),
    }

    # Migrate model
    if "model" not in v1_config:
        raise ValueError("V1 config missing required 'model' section")

    v1_model = v1_config["model"]
    v2_config["model"] = {
        "architecture": {
            "image_size": v1_model.get("image_size", 40),
            "in_channels": v1_model.get("in_channels", 3),
            "model_channels": v1_model.get("model_channels", 64),
            "channel_multipliers": v1_model.get("channel_multipliers", [1, 2, 4]),
            "use_attention": v1_model.get("use_attention", [False, False, True]),
        },
        "diffusion": {
            "num_timesteps": v1_model.get("num_timesteps", 1000),
            "beta_schedule": v1_model.get("beta_schedule", "cosine"),
            "beta_start": v1_model.get("beta_start", 0.0001),
            "beta_end": v1_model.get("beta_end", 0.02),
        },
        "conditioning": {
            "type": "class" if v1_model.get("num_classes") else None,
            "num_classes": v1_model.get("num_classes"),
            "class_dropout_prob": v1_model.get("class_dropout_prob", 0.1),
        },
    }

    # Migrate data
    if "data" not in v1_config:
        raise ValueError("V1 config missing required 'data' section")

    v1_data = v1_config["data"]
    v2_config["data"] = {
        "paths": {
            "train": v1_data.get("train_path", "data/train"),
            "val": v1_data.get("val_path"),
        },
        "loading": {
            "batch_size": v1_data.get("batch_size", 32),
            "num_workers": v1_data.get("num_workers", 4),
            "pin_memory": v1_data.get("pin_memory", True),
            "shuffle_train": v1_data.get("shuffle_train", True),
            "drop_last": v1_data.get("drop_last", False),
        },
        "augmentation": {
            "horizontal_flip": v1_data.get("horizontal_flip", True),
            "rotation_degrees": v1_data.get("rotation_degrees", 0),
            "color_jitter": {
                "enabled": v1_data.get("color_jitter", False),
                "strength": v1_data.get("color_jitter_strength", 0.1),
            },
        },
    }

    # Migrate output
    v1_output = v1_config.get("output", {})
    v2_config["output"] = {
        "base_dir": "outputs",
        "subdirs": {
            "logs": "logs",
            "checkpoints": "checkpoints",
            "samples": "samples",
            "generated": "generated",
        },
    }

    # Migrate training
    if "training" in v1_config:
        v1_training = v1_config["training"]
        v2_config["training"] = {
            "epochs": v1_training.get("epochs", 200),
            "optimizer": {
                "type": v1_training.get("optimizer", "adam"),
                "learning_rate": v1_training.get("learning_rate", 0.0001),
                **v1_training.get("optimizer_kwargs", {}),
            },
            "scheduler": {
                "type": v1_training.get("scheduler"),
                **v1_training.get("scheduler_kwargs", {}),
            },
            "ema": {
                "enabled": v1_training.get("use_ema", True),
                "decay": v1_training.get("ema_decay", 0.9999),
            },
            "checkpointing": {
                "save_frequency": v1_training.get("save_frequency", 10),
                "save_best_only": v1_training.get("save_best_only", False),
                "save_optimizer": True,
            },
            "validation": v1_training.get(
                "validation", {"enabled": True, "frequency": 1, "metric": "loss"}
            ),
            "visualization": v1_training.get(
                "visualization",
                {
                    "enabled": True,
                    "interval": 10,
                    "num_samples": 8,
                    "guidance_scale": 3.0,
                },
            ),
            "performance": {
                "use_amp": v1_training.get("use_amp", False),
                "use_tf32": True,
                "cudnn_benchmark": True,
                "compile_model": False,
            },
            "resume": {
                "enabled": False,
                "checkpoint": None,
                "reset_optimizer": False,
                "reset_scheduler": False,
            },
        }

    # Migrate generation
    if "generation" in v1_config:
        v1_gen = v1_config["generation"]
        v2_config["generation"] = {
            "checkpoint": v1_gen.get("checkpoint"),
            "sampling": {
                "num_samples": v1_gen.get("num_samples", 100),
                "guidance_scale": v1_gen.get("guidance_scale", 3.0),
                "use_ema": v1_gen.get("use_ema", True),
            },
            "output": {
                "save_individual": True,
                "save_grid": v1_gen.get("save_grid", True),
                "grid_nrow": v1_gen.get("grid_nrow", 10),
            },
        }

    return v2_config
