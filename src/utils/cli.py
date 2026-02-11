"""Command-line interface utilities.

This module provides CLI argument parsing with config-only mode.
All configuration must be provided via JSON config file.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import load_config


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for the project.

    Config-only mode: accepts only a config file path as positional argument.

    Returns:
        Configured ArgumentParser instance

    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(['configs/classifier/baseline.json'])
        >>> args.config_path
        'configs/classifier/baseline.json'
    """
    parser = argparse.ArgumentParser(
        description="AI Image Generation and Statistics Training Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required positional argument: config file path
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to JSON configuration file",
    )

    # Optional verbose flag
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    return parser


def parse_args(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse command-line arguments and load configuration from file.

    This is the main entry point for CLI processing. It:
    1. Parses CLI arguments (config file path)
    2. Loads and validates the config file
    3. Returns the configuration dictionary

    Args:
        args: List of argument strings (default: sys.argv)

    Returns:
        Configuration dictionary loaded from file

    Raises:
        FileNotFoundError: If specified config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If experiment type is invalid or missing

    Example:
        >>> config = parse_args(['configs/classifier/baseline.json'])
        >>> config['experiment']
        'classifier'
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Get config file path from positional argument
    config_path = parsed_args.config_path

    # Verify config file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config file
    config = load_config(config_path)

    # Verify experiment field exists
    if "experiment" not in config:
        raise ValueError(
            "Missing required 'experiment' field in config file. "
            "Must be one of: classifier, diffusion, gan"
        )

    # Validate experiment type
    valid_experiments = ["classifier", "diffusion", "gan"]
    if config["experiment"] not in valid_experiments:
        raise ValueError(
            f"Invalid experiment type: {config['experiment']}. "
            f"Must be one of: {valid_experiments}"
        )

    # Add verbose flag to config if needed
    config["verbose"] = parsed_args.verbose

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate basic configuration structure.

    Performs lightweight validation - detailed validation is done by
    experiment-specific validators.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required top-level fields are missing or invalid
    """
    # Check required fields
    if "experiment" not in config:
        raise ValueError("'experiment' field is required in config")

    valid_experiments = ["classifier", "diffusion", "gan"]
    if config["experiment"] not in valid_experiments:
        raise ValueError(
            f"Invalid experiment type: {config['experiment']}. "
            f"Must be one of: {valid_experiments}"
        )
