#!/usr/bin/env python3
"""Migration Script: Classifier Config V1 to V2

This script converts classifier configuration files from V1 format to V2 format.

V2 Changes:
- Added compute section (device, seed moved from training)
- Added mode field (train, evaluate)
- Restructured model config (architecture, initialization, regularization)
- Restructured data config (paths, loading, preprocessing, augmentation)
- Unified output directory structure (base_dir + subdirs)
- Grouped optimizer parameters (type, learning_rate, etc.)
- Grouped scheduler parameters (type, T_max, etc.)
- Moved checkpointing parameters to training.checkpointing
- Moved validation parameters to training.validation
- Added performance section (use_amp, use_tf32, etc.)
- Added resume section

Usage:
    python scripts/migrate_classifier_config_v1_to_v2.py --input old_config.yaml --output new_config.yaml
    python scripts/migrate_classifier_config_v1_to_v2.py --input configs/classifier/legacy.yaml --output configs/classifier/migrated.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def migrate_v1_to_v2(v1_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert V1 classifier config to V2 format.

    Args:
        v1_config: Configuration in V1 format

    Returns:
        Configuration in V2 format

    Raises:
        ValueError: If V1 config structure is invalid
    """
    # Check if already V2
    if "compute" in v1_config and "mode" in v1_config:
        print("Warning: Config appears to be in V2 format already")
        return v1_config.copy()

    # Initialize V2 config
    v2_config: Dict[str, Any] = {}

    # Basic fields
    v2_config["experiment"] = v1_config.get("experiment", "classifier")
    v2_config["mode"] = "train"

    # Migrate compute (from training section in V1)
    training_v1 = v1_config.get("training", {})
    v2_config["compute"] = {
        "device": training_v1.get("device", "cuda"),
        "seed": training_v1.get("seed"),
    }

    # Migrate model
    if "model" not in v1_config:
        raise ValueError("V1 config missing required 'model' section")

    model_v1 = v1_config["model"]
    v2_config["model"] = {
        "architecture": {
            "name": model_v1.get("name", "resnet50"),
            "num_classes": model_v1.get("num_classes", 2),
        },
        "initialization": {
            "pretrained": model_v1.get("pretrained", True),
            "freeze_backbone": model_v1.get("freeze_backbone", False),
            "trainable_layers": model_v1.get("trainable_layers"),
        },
        "regularization": {
            "dropout": model_v1.get("dropout", 0.5),
        },
    }

    # Migrate data
    if "data" not in v1_config:
        raise ValueError("V1 config missing required 'data' section")

    data_v1 = v1_config["data"]

    # Handle color_jitter - V1 is boolean, V2 is nested dict
    color_jitter_v1 = data_v1.get("color_jitter", False)
    if isinstance(color_jitter_v1, bool):
        color_jitter_v2 = {
            "enabled": color_jitter_v1,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        }
    else:
        # Already a dict (shouldn't happen in V1)
        color_jitter_v2 = color_jitter_v1

    v2_config["data"] = {
        "paths": {
            "train": data_v1.get("train_path", "data/train"),
            "val": data_v1.get("val_path", "data/val"),
        },
        "loading": {
            "batch_size": data_v1.get("batch_size", 32),
            "num_workers": data_v1.get("num_workers", 4),
            "pin_memory": data_v1.get("pin_memory", True),
            "shuffle_train": data_v1.get("shuffle_train", True),
            "drop_last": data_v1.get("drop_last", False),
        },
        "preprocessing": {
            "image_size": data_v1.get("image_size", 256),
            "crop_size": data_v1.get("crop_size", 224),
            "normalize": data_v1.get("normalize", "imagenet"),
        },
        "augmentation": {
            "horizontal_flip": data_v1.get("horizontal_flip", True),
            "rotation_degrees": data_v1.get("rotation_degrees", 0),
            "color_jitter": color_jitter_v2,
        },
    }

    # Migrate output
    output_v1 = v1_config.get("output", {})
    v2_config["output"] = {
        "base_dir": "outputs",
        "subdirs": {
            "logs": "logs",
            "checkpoints": "checkpoints",
        },
    }

    # Migrate training
    if "training" in v1_config:
        training_v1 = v1_config["training"]

        # Migrate optimizer
        optimizer_name = training_v1.get("optimizer", "adam")
        optimizer_kwargs = training_v1.get("optimizer_kwargs", {})
        optimizer_v2 = {
            "type": optimizer_name,
            "learning_rate": training_v1.get("learning_rate", 0.001),
            "gradient_clip_norm": training_v1.get("gradient_clip"),
        }
        # Merge optimizer kwargs
        for key, value in optimizer_kwargs.items():
            if key not in optimizer_v2:
                optimizer_v2[key] = value

        # Migrate scheduler
        scheduler_name = training_v1.get("scheduler", "cosine")
        scheduler_kwargs = training_v1.get("scheduler_kwargs", {})
        scheduler_v2 = {
            "type": scheduler_name if scheduler_name not in ["none", None] else None,
        }
        # Merge scheduler kwargs, handle T_max auto
        for key, value in scheduler_kwargs.items():
            if key == "T_max" and value == training_v1.get("epochs"):
                scheduler_v2[key] = "auto"
            else:
                scheduler_v2[key] = value

        # Migrate validation
        validation_v1 = v1_config.get("validation", {})
        validation_v2 = {
            "enabled": True if validation_v1 else True,  # Default to enabled
            "frequency": validation_v1.get("frequency", 1),
            "metric": validation_v1.get("metric", "accuracy"),
            "early_stopping_patience": training_v1.get("early_stopping_patience"),
        }

        # Build training section
        v2_config["training"] = {
            "epochs": training_v1.get("epochs", 100),
            "optimizer": optimizer_v2,
            "scheduler": scheduler_v2,
            "checkpointing": {
                "save_frequency": output_v1.get("save_frequency", 10),
                "save_best_only": output_v1.get("save_best_only", True),
                "save_optimizer": True,
            },
            "validation": validation_v2,
            "performance": {
                "use_amp": training_v1.get("mixed_precision", False),
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

    # Add evaluation section (default, empty)
    v2_config["evaluation"] = {
        "checkpoint": None,
        "data": {
            "test_path": "data/test",
            "batch_size": 32,
        },
        "output": {
            "save_predictions": True,
            "save_confusion_matrix": True,
            "save_metrics": True,
        },
    }

    return v2_config


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate classifier config from V1 to V2 format"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input V1 config file path"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output V2 config file path"
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite output file if exists"
    )

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Error: Output file already exists: {output_path}")
        print("Use --force to overwrite")
        sys.exit(1)

    # Load V1 config
    print(f"Loading V1 config from: {input_path}")
    with open(input_path) as f:
        v1_config = yaml.safe_load(f)

    # Migrate to V2
    print("Migrating to V2 format...")
    try:
        v2_config = migrate_v1_to_v2(v1_config)
    except Exception as e:
        print(f"Error during migration: {e}")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save V2 config
    print(f"Saving V2 config to: {output_path}")
    with open(output_path, "w") as f:
        yaml.dump(v2_config, f, default_flow_style=False, sort_keys=False, indent=2)

    print("Migration completed successfully!")
    print("\nKey changes:")
    print("- Added 'compute' section (device, seed)")
    print("- Added 'mode' field")
    print(
        "- Restructured 'model' section (architecture, initialization, regularization)"
    )
    print("- Restructured 'data' section (paths, loading, preprocessing, augmentation)")
    print("- Unified 'output' directory structure (base_dir + subdirs)")
    print("- Grouped optimizer and scheduler parameters")
    print("- Added performance and resume sections")
    print("\nPlease review the migrated config and adjust as needed.")


if __name__ == "__main__":
    main()
