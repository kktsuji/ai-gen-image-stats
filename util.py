"""Utility functions for the project."""

import argparse
import json
import os
from typing import Any, Dict


def save_args(
    args: argparse.Namespace, output_dir: str, filename: str = "args.json"
) -> None:
    """Save command line arguments to a JSON file.

    Args:
        args: argparse.Namespace containing the arguments
        output_dir: Directory where the arguments file will be saved
        filename: Name of the output file (default: args.json)
    """
    os.makedirs(output_dir, exist_ok=True)

    args_dict = vars(args)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)

    print(f"Arguments saved to {output_path}")
