"""Data Preparation - Split Generation Logic

This module implements the core logic for creating reproducible train/val splits
from class directories. It scans image files, performs stratified splitting with
a configurable seed, and saves the result as a JSON file with metadata.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Supported image extensions (same as ImageFolderDataset)
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def _scan_image_files(directory: str) -> List[str]:
    """Scan a directory for image files.

    Args:
        directory: Path to directory to scan

    Returns:
        Sorted list of image file paths (relative to project root)

    Raises:
        FileNotFoundError: If directory does not exist
        ValueError: If directory contains no image files
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Class directory not found: {directory}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))

    if not image_files:
        raise ValueError(f"No image files found in: {directory}")

    # Convert to relative paths (from project root) and sort for determinism
    relative_paths = sorted(str(p) for p in image_files)
    return relative_paths


def _split_list(items: List[str], train_ratio: float, rng: random.Random) -> tuple:
    """Split a list into train and val portions.

    Args:
        items: List of items to split
        train_ratio: Fraction of items for training
        rng: Random number generator instance

    Returns:
        Tuple of (train_items, val_items)
    """
    shuffled = items.copy()
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    # Ensure at least 1 item in each split if possible
    if split_idx == 0 and len(shuffled) > 1:
        split_idx = 1
    elif split_idx == len(shuffled) and len(shuffled) > 1:
        split_idx = len(shuffled) - 1

    return shuffled[:split_idx], shuffled[split_idx:]


def prepare_split(config: Dict[str, Any]) -> str:
    """Generate a train/val split from class directories.

    Scans each class directory for image files, performs a stratified split
    with the configured seed, and saves the result as a JSON file.

    Args:
        config: Configuration dictionary with 'classes' and 'split' sections

    Returns:
        Path to the generated split JSON file

    Raises:
        FileNotFoundError: If class directories don't exist
        ValueError: If configuration is invalid or directories are empty
        FileExistsError: If split file exists and force=false
    """
    classes_config = config["classes"]
    split_config = config["split"]

    seed = split_config.get("seed")
    train_ratio = split_config["train_ratio"]
    save_dir = split_config["save_dir"]
    split_file = split_config["split_file"]
    force = split_config.get("force", False)

    # Build output path
    output_dir = Path(save_dir)
    output_path = output_dir / split_file

    # Check if file exists and skip if force=false
    if output_path.exists() and not force:
        logger.info(f"Split file already exists: {output_path}")
        logger.info("Skipping regeneration (set split.force=true to overwrite)")
        return str(output_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create RNG instance (isolated from global random state)
    rng = random.Random(seed)

    # Build class-to-label mapping (sorted by class name for determinism)
    sorted_class_names = sorted(classes_config.keys())
    class_to_label = {name: idx for idx, name in enumerate(sorted_class_names)}

    logger.info("=" * 60)
    logger.info("DATA PREPARATION - Train/Val Split")
    logger.info("=" * 60)
    logger.info(f"Seed: {seed}")
    logger.info(f"Train ratio: {train_ratio}")
    logger.info(f"Classes: {sorted_class_names}")

    # Scan and split each class
    all_train = []
    all_val = []
    class_samples = {}
    source_paths = {}

    for class_name in sorted_class_names:
        class_path = classes_config[class_name]
        label = class_to_label[class_name]
        source_paths[class_name] = class_path

        logger.info(f"Scanning class '{class_name}': {class_path}")

        # Scan for image files
        image_files = _scan_image_files(class_path)
        logger.info(f"  Found {len(image_files)} images")

        # Split this class
        train_files, val_files = _split_list(image_files, train_ratio, rng)

        # Record per-class statistics
        class_samples[class_name] = {
            "total": len(image_files),
            "train": len(train_files),
            "val": len(val_files),
        }

        # Add to combined lists with labels
        for path in train_files:
            all_train.append({"path": path, "label": label})
        for path in val_files:
            all_val.append({"path": path, "label": label})

        logger.info(f"  Train: {len(train_files)}, Val: {len(val_files)}")

    # Build JSON structure
    total_samples = len(all_train) + len(all_val)
    split_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "train_ratio": train_ratio,
            "total_samples": total_samples,
            "train_samples": len(all_train),
            "val_samples": len(all_val),
            "classes": class_to_label,
            "class_samples": class_samples,
            "source_paths": source_paths,
        },
        "train": all_train,
        "val": all_val,
    }

    # Write JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info(f"Split file saved: {output_path}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Train samples: {len(all_train)}")
    logger.info(f"Val samples: {len(all_val)}")
    logger.info("=" * 60)

    return str(output_path)
