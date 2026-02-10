"""Command-line interface utilities.

This module provides CLI argument parsing with support for configuration
file overrides. Priority order: CLI arguments > Config file > Code defaults
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import load_and_merge_configs


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for the project.

    Returns:
        Configured ArgumentParser instance

    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(['--experiment', 'classifier', '--epochs', '10'])
        >>> args.experiment
        'classifier'
    """
    parser = argparse.ArgumentParser(
        description="AI Image Generation and Statistics Training Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["classifier", "diffusion", "gan"],
        help="Type of experiment to run",
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "sgd", "adamw"],
        help="Optimizer type",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture name",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=None,
        help="Use pretrained weights",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes for classification",
    )

    # Data parameters
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to training data directory",
    )

    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Path to validation data directory",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saving outputs",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for saving logs",
    )

    # Experiment modes
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate", "evaluate"],
        help="Experiment mode",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for loading/resuming",
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to generate",
    )

    # Device parameters
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device to use for training/inference",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Misc
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    return parser


def args_to_dict(args: argparse.Namespace, exclude_none: bool = True) -> Dict[str, Any]:
    """Convert argument namespace to a nested configuration dictionary.

    Converts flat CLI arguments to nested config structure matching the
    expected format for config files. Arguments with underscores or hyphens
    are converted to nested dictionaries where appropriate.

    Args:
        args: Parsed argument namespace from argparse
        exclude_none: If True, exclude arguments that are None (default: True)

    Returns:
        Nested dictionary suitable for config merging

    Example:
        >>> args = argparse.Namespace(experiment='classifier', epochs=10, batch_size=32)
        >>> args_to_dict(args)
        {'experiment': 'classifier', 'training': {'epochs': 10}, 'data': {'batch_size': 32}}
    """
    # Map CLI arguments to config structure
    # Arguments are organized into sections matching config file format
    arg_mapping = {
        # Top-level fields
        "experiment": ("experiment",),
        "mode": ("mode",),
        # Model section
        "model": ("model", "name"),
        "pretrained": ("model", "pretrained"),
        "num_classes": ("model", "num_classes"),
        # Data section
        "train_path": ("data", "train_path"),
        "val_path": ("data", "val_path"),
        "batch_size": ("data", "batch_size"),
        "num_workers": ("data", "num_workers"),
        # Training section
        "epochs": ("training", "epochs"),
        "lr": ("training", "learning_rate"),
        "optimizer": ("training", "optimizer"),
        "seed": ("training", "seed"),
        # Output section
        "output_dir": ("output", "output_dir"),
        "checkpoint_dir": ("output", "checkpoint_dir"),
        "log_dir": ("output", "log_dir"),
        "checkpoint": ("output", "checkpoint"),
        # Generation section
        "num_samples": ("generation", "num_samples"),
        # Device section
        "device": ("device",),
        "verbose": ("verbose",),
    }

    args_dict = vars(args)
    config: Dict[str, Any] = {}

    for arg_name, arg_value in args_dict.items():
        # Skip None values if requested
        if exclude_none and arg_value is None:
            continue

        # Convert hyphen to underscore for lookup
        lookup_name = arg_name.replace("-", "_")

        # Skip config file path (handled separately)
        if lookup_name == "config":
            continue

        # Get the config path for this argument
        if lookup_name not in arg_mapping:
            # Unknown argument, add to top level
            config[lookup_name] = arg_value
            continue

        config_path = arg_mapping[lookup_name]

        # Set the value in the nested config structure
        current = config
        for i, key in enumerate(config_path[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[config_path[-1]] = arg_value

    return config


def parse_args(
    args: Optional[List[str]] = None, defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Parse command-line arguments and merge with config file and defaults.

    This is the main entry point for CLI processing. It:
    1. Parses CLI arguments
    2. Loads config file if specified
    3. Merges everything with correct priority: CLI > Config file > Defaults

    Args:
        args: List of argument strings (default: sys.argv)
        defaults: Optional dictionary of default configuration values

    Returns:
        Merged configuration dictionary with all settings

    Raises:
        FileNotFoundError: If specified config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON

    Example:
        >>> config = parse_args(['--experiment', 'classifier', '--epochs', '10'])
        >>> config['experiment']
        'classifier'
        >>> config['training']['epochs']
        10
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Convert CLI args to config dict (excluding None values)
    cli_overrides = args_to_dict(parsed_args, exclude_none=True)

    # Get config file path if specified
    config_path = parsed_args.config

    # Merge everything with correct priority
    final_config = load_and_merge_configs(
        config_path=config_path, defaults=defaults, overrides=cli_overrides
    )

    return final_config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration dictionary.

    Checks that required fields are present and have valid values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required fields are missing or have invalid values
    """
    # Check required fields
    if "experiment" not in config:
        raise ValueError("'experiment' field is required in config")

    if config["experiment"] not in ["classifier", "diffusion", "gan"]:
        raise ValueError(
            f"Invalid experiment type: {config['experiment']}. "
            f"Must be one of: classifier, diffusion, gan"
        )

    # Validate paths exist if specified
    path_fields = [
        ("data", "train_path"),
        ("data", "val_path"),
        ("output", "checkpoint"),
    ]

    for section, field in path_fields:
        if section in config and field in config[section]:
            path_value = config[section][field]
            if path_value is not None:
                path = Path(path_value)
                # Only check training/validation paths in train mode
                if field in ["train_path", "val_path"]:
                    mode = config.get("mode", "train")
                    if mode == "train" and not path.exists():
                        raise ValueError(f"Path does not exist: {path_value}")
                # Check checkpoint path for loading
                elif field == "checkpoint" and not path.exists():
                    raise ValueError(f"Checkpoint file does not exist: {path_value}")

    # Validate numeric ranges
    if "training" in config:
        training = config["training"]
        if "epochs" in training and training["epochs"] is not None:
            if training["epochs"] <= 0:
                raise ValueError(f"epochs must be positive, got {training['epochs']}")

        if "learning_rate" in training and training["learning_rate"] is not None:
            if training["learning_rate"] <= 0:
                raise ValueError(
                    f"learning_rate must be positive, got {training['learning_rate']}"
                )

    if "data" in config:
        data = config["data"]
        if "batch_size" in data and data["batch_size"] is not None:
            if data["batch_size"] <= 0:
                raise ValueError(
                    f"batch_size must be positive, got {data['batch_size']}"
                )

        if "num_workers" in data and data["num_workers"] is not None:
            if data["num_workers"] < 0:
                raise ValueError(
                    f"num_workers must be non-negative, got {data['num_workers']}"
                )
