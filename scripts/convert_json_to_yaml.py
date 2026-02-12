#!/usr/bin/env python3
"""Convert JSON configuration files to YAML format.

This script converts all JSON config files in the project to YAML format,
preserving the structure and data.

Usage:
    python scripts/convert_json_to_yaml.py
    python scripts/convert_json_to_yaml.py --path configs/
    python scripts/convert_json_to_yaml.py --file configs/classifier/baseline.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import yaml


def convert_file(json_path: Path, delete_json: bool = False) -> Optional[Path]:
    """Convert a single JSON file to YAML.

    Args:
        json_path: Path to JSON file
        delete_json: Whether to delete JSON file after conversion

    Returns:
        Path to created YAML file, or None if conversion failed
    """
    try:
        # Load JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        # Save as YAML
        yaml_path = json_path.with_suffix(".yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Converted: {json_path.name} → {yaml_path.name}")

        # Delete JSON if requested
        if delete_json:
            json_path.unlink()
            print(f"  Deleted: {json_path.name}")

        return yaml_path

    except Exception as e:
        print(f"✗ Error converting {json_path}: {e}")
        return None


def convert_directory(
    dir_path: Path, delete_json: bool = False, recursive: bool = True
):
    """Convert all JSON files in a directory.

    Args:
        dir_path: Path to directory
        delete_json: Whether to delete JSON files after conversion
        recursive: Whether to search subdirectories
    """
    pattern = "**/*.json" if recursive else "*.json"
    json_files = list(dir_path.glob(pattern))

    if not json_files:
        print(f"No JSON files found in {dir_path}")
        return

    print(f"Found {len(json_files)} JSON file(s) in {dir_path}")
    print()

    converted = 0
    for json_file in json_files:
        if convert_file(json_file, delete_json):
            converted += 1

    print()
    print(f"Converted {converted}/{len(json_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON config files to YAML format"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Directory to convert (default: configs/)",
        default="configs/",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Single file to convert (overrides --path)",
    )
    parser.add_argument(
        "--delete-json",
        action="store_true",
        help="Delete JSON files after successful conversion",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )

    args = parser.parse_args()

    if args.file:
        # Convert single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1

        convert_file(file_path, args.delete_json)

    else:
        # Convert directory
        dir_path = Path(args.path)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            return 1

        convert_directory(dir_path, args.delete_json, not args.no_recursive)

    return 0


if __name__ == "__main__":
    exit(main())
