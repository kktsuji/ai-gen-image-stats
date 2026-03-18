"""Command-line interface utilities.

This module provides CLI argument parsing with config file and optional
parameter overrides (top-level or dot-notation).
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import load_config, merge_configs


def infer_type(value: str) -> Any:
    """Convert a CLI string value to the appropriate Python type.

    Conversion order: quoted string > bool > None > int > float > str.

    Values wrapped in matching quotes (single or double) are always treated
    as strings with the quotes stripped. This allows forcing string type for
    values that would otherwise be inferred as another type.

    Args:
        value: String value from CLI argument

    Returns:
        Converted value with inferred type

    Example:
        >>> infer_type("42")
        42
        >>> infer_type("3.14")
        3.14
        >>> infer_type("true")
        True
        >>> infer_type("none")
        None
        >>> infer_type("hello")
        'hello'
        >>> infer_type("'42'")
        '42'
        >>> infer_type('"true"')
        'true'
    """
    # Quoted string — strip quotes and return as-is
    if len(value) >= 2 and (
        (value.startswith("'") and value.endswith("'"))
        or (value.startswith('"') and value.endswith('"'))
    ):
        return value[1:-1]

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # None
    if value.lower() in ("null", "none"):
        return None

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        f = float(value)
        if not math.isfinite(f):
            return value  # treat inf/nan as strings
        return f
    except ValueError:
        pass

    # String (default)
    return value


def dot_notation_to_dict(key: str, value: Any) -> Dict[str, Any]:
    """Convert a dot-notation key and value into a nested dictionary.

    Args:
        key: Key string, either a single segment (e.g., "mode") or
            dot-separated for nested access (e.g., "model.architecture.image_size")
        value: Value to set at the leaf

    Returns:
        Nested dictionary

    Example:
        >>> dot_notation_to_dict("mode", "generate")
        {'mode': 'generate'}
        >>> dot_notation_to_dict("model.architecture.image_size", 60)
        {'model': {'architecture': {'image_size': 60}}}
    """
    parts = key.split(".")
    result: Dict[str, Any] = {}
    current = result
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return result


def parse_override_args(remaining: List[str]) -> Dict[str, Any]:
    """Parse remaining CLI arguments as config overrides.

    Each override must be a ``--key value`` pair. Keys may use dot-notation
    for nested values (e.g., ``--model.architecture.image_size 60``) or be
    top-level keys (e.g., ``--mode generate``).

    Args:
        remaining: List of unrecognized CLI arguments from parse_known_args

    Returns:
        Nested dictionary of override values

    Raises:
        ValueError: If arguments are malformed (missing value)

    Note:
        Values starting with ``--`` are treated as the next override key, not
        as a value. For example, ``--output.prefix "--exp"`` will raise a
        "Missing value" error because ``"--exp"`` is interpreted as a new key.

    Example:
        >>> parse_override_args(["--mode", "generate"])
        {'mode': 'generate'}
        >>> parse_override_args(["--model.architecture.image_size", "60"])
        {'model': {'architecture': {'image_size': 60}}}
    """
    if not remaining:
        return {}

    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(remaining):
        arg = remaining[i]

        if not arg.startswith("--"):
            if arg.startswith("-") and "." in arg:
                raise ValueError(
                    f"Invalid override: '{arg}'. "
                    f"Use double dashes (e.g., --{arg.lstrip('-')})"
                )
            raise ValueError(
                f"Unexpected argument: '{arg}'. "
                f"Override keys must start with '--' "
                f"(e.g., --mode generate or --model.architecture.image_size 60)"
            )

        key = arg[2:]  # Strip leading --

        # Check that a value follows
        if i + 1 >= len(remaining) or remaining[i + 1].startswith("--"):
            raise ValueError(
                f"Missing value for override key: '{arg}'. "
                f"Each override must have a value "
                f"(e.g., --model.architecture.image_size 60)"
            )

        raw_value = remaining[i + 1]
        typed_value = infer_type(raw_value)
        override_dict = dot_notation_to_dict(key, typed_value)
        overrides = merge_configs(overrides, override_dict)

        i += 2

    return overrides


def validate_override_keys(
    config: Dict[str, Any], overrides: Dict[str, Any], prefix: str = ""
) -> None:
    """Validate that all override keys exist in the base config.

    Catches typos like ``--model.architectur.image_size`` (missing 'e') that
    would silently create a new key instead of overriding the intended one.

    Args:
        config: Base configuration dictionary to validate against
        overrides: Nested dictionary of override values
        prefix: Dot-separated path for error messages (used in recursion)

    Raises:
        ValueError: If an override key does not exist in the base config

    Note:
        This function validates key existence only, not value types. Type
        validation is delegated to experiment-level config validators.
    """
    for key, value in overrides.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in config:
            raise ValueError(
                f"Unknown config key: '{full_key}'. "
                f"Check for typos. "
                f"Available keys at '{prefix or '<root>'}': "
                f"{sorted(config.keys())}"
            )
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            validate_override_keys(config[key], value, full_key)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for the project.

    Accepts a config file path as positional argument, an optional --verbose
    flag, and any config overrides (parsed separately).

    Returns:
        Configured ArgumentParser instance

    Example:
        >>> parser = create_parser()
        >>> args, _ = parser.parse_known_args(
        ...     ['configs/diffusion.yaml', '--model.architecture.image_size', '60']
        ... )
        >>> args.config_path
        'configs/diffusion.yaml'
    """
    parser = argparse.ArgumentParser(
        description="AI Image Generation and Statistics Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Config overrides:\n"
            "  python -m src.main configs/diffusion.yaml "
            "--mode generate\n"
            "  python -m src.main configs/diffusion.yaml "
            "--model.architecture.image_size 60\n"
            "  python -m src.main configs/diffusion.yaml "
            "--training.epochs 50 --data.loading.batch_size 16\n"
        ),
    )

    # Required positional argument: config file path
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file",
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
    """Parse command-line arguments, load config, and apply overrides.

    This is the main entry point for CLI processing. It:
    1. Parses known CLI arguments (config file path, --verbose)
    2. Loads the config file
    3. Parses remaining arguments as config overrides
    4. Deep-merges overrides on top of the loaded config
    5. Returns the final configuration dictionary

    Args:
        args: List of argument strings (default: sys.argv)

    Returns:
        Configuration dictionary with CLI overrides applied

    Raises:
        FileNotFoundError: If specified config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If overrides are malformed

    Example:
        >>> config = parse_args(['config.yaml', '--model.architecture.image_size', '60'])
        >>> config['model']['architecture']['image_size']
        60
    """
    parser = create_parser()
    parsed_args, remaining = parser.parse_known_args(args)

    # Get config file path from positional argument
    config_path = parsed_args.config_path

    # Verify config path points to an existing regular file
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config file
    config = load_config(config_path)

    # Apply overrides from CLI
    if remaining:
        cli_overrides = parse_override_args(remaining)
        validate_override_keys(config, cli_overrides)
        config = merge_configs(config, cli_overrides)

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

    valid_experiments = [
        "classifier",
        "diffusion",
        "gan",
        "data_preparation",
        "sample_selection",
    ]
    if config["experiment"] not in valid_experiments:
        raise ValueError(
            f"Invalid experiment type: {config['experiment']}. "
            f"Must be one of: {valid_experiments}"
        )
