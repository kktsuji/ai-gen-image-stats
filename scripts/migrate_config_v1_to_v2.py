#!/usr/bin/env python3
"""Migrate diffusion configuration from V1 to V2 format.

Usage:
    python scripts/migrate_config_v1_to_v2.py --input old_config.yaml --output new_config.yaml

    # Dry run (print without saving)
    python scripts/migrate_config_v1_to_v2.py --input old_config.yaml --output new_config.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import migrate_config_v1_to_v2


def main():
    parser = argparse.ArgumentParser(
        description="Migrate diffusion configuration from V1 to V2 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a configuration file
  python scripts/migrate_config_v1_to_v2.py -i old.yaml -o new.yaml
  
  # Dry run to preview changes
  python scripts/migrate_config_v1_to_v2.py -i old.yaml -o new.yaml --dry-run
  
For more information, see:
  docs/research/diffusion-config-migration-guide.md
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to V1 configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to save V2 configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print V2 config without saving",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    args = parser.parse_args()

    # Load V1 config
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading V1 config from: {input_path}")
    try:
        with open(input_path, "r") as f:
            v1_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

    # Migrate to V2
    print("Migrating to V2 format...")
    try:
        v2_config = migrate_config_v1_to_v2(v1_config)
    except Exception as e:
        print(f"Error during migration: {e}")
        sys.exit(1)

    # Save or print
    if args.dry_run:
        print("\n" + "=" * 80)
        print("MIGRATED V2 CONFIG (DRY RUN - NOT SAVED)")
        print("=" * 80 + "\n")
        print(yaml.dump(v2_config, default_flow_style=False, sort_keys=False))
        print("\n" + "=" * 80)
        print("To save this configuration, run without --dry-run")
        print("=" * 80)
    else:
        output_path = Path(args.output)

        # Check if output exists
        if output_path.exists() and not args.force:
            print(f"Error: Output file already exists: {output_path}")
            print("Use --force to overwrite, or choose a different output path")
            sys.exit(1)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save V2 config
        try:
            with open(output_path, "w") as f:
                yaml.dump(
                    v2_config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            print(f"\n✓ V2 config saved to: {output_path}")
        except Exception as e:
            print(f"Error saving config file: {e}")
            sys.exit(1)

        print("\nNext steps:")
        print("  1. Review the migrated configuration")
        print("  2. Test with: python -m src.main", str(output_path))
        print(
            "  3. See migration guide: docs/research/diffusion-config-migration-guide.md"
        )


if __name__ == "__main__":
    main()
