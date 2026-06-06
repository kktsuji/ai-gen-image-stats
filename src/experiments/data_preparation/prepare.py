"""Data Preparation - Split Generation Logic

This module implements the core logic for creating reproducible train/val/test
splits from class directories. It scans image files, performs stratified splitting
with a configurable seed, and saves the result as a JSON file with metadata.
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


def _split_list(
    items: List[str], train_ratio: float, val_ratio: float, rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    """Split a list into train, val, and test portions.

    Args:
        items: List of items to split
        train_ratio: Fraction of items for training
        val_ratio: Fraction of items for validation (test gets the remainder)
        rng: Random number generator instance

    Returns:
        Tuple of (train_items, val_items, test_items)

    Notes:
        Splits are stratified per-class when this function is applied per class.
        For very small lists the train and val portions are guaranteed at least
        one item each when possible, but the test portion may be empty.
    """
    shuffled = items.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)

    # Ensure at least 1 item in train and val if possible (test takes remainder)
    if n > 1:
        if train_idx == 0:
            train_idx = 1
        if val_idx <= train_idx:
            val_idx = min(train_idx + 1, n)

    return shuffled[:train_idx], shuffled[train_idx:val_idx], shuffled[val_idx:]


def _kfold_chunks(items: List[str], n_folds: int) -> List[List[str]]:
    """Partition an (already shuffled) list into ``n_folds`` contiguous chunks.

    Chunk sizes differ by at most one, mirroring scikit-learn's ``KFold`` so the
    folds together partition ``items`` (every element appears in exactly one
    chunk). Applied per class, this yields a stratified fold assignment.
    """
    n = len(items)
    folds: List[List[str]] = []
    start = 0
    for k in range(n_folds):
        # Distribute the remainder across the first (n % n_folds) folds.
        size = n // n_folds + (1 if k < n % n_folds else 0)
        folds.append(items[start : start + size])
        start += size
    return folds


def _build_split_dict(
    all_train: List[Dict[str, Any]],
    all_val: List[Dict[str, Any]],
    all_test: List[Dict[str, Any]],
    class_to_label: Dict[str, int],
    class_samples: Dict[str, Dict[str, int]],
    source_paths: Dict[str, str],
    extra_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the split JSON structure shared by both split modes."""
    total_samples = len(all_train) + len(all_val) + len(all_test)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_samples": total_samples,
        "train_samples": len(all_train),
        "val_samples": len(all_val),
        "test_samples": len(all_test),
        "classes": class_to_label,
        "class_samples": class_samples,
        "source_paths": source_paths,
    }
    metadata.update(extra_metadata)
    return {
        "metadata": metadata,
        "train": all_train,
        "val": all_val,
        "test": all_test,
    }


def prepare_split(config: Dict[str, Any]) -> str:
    """Generate a single stratified train/val/test split (ratio mode).

    Scans each class directory for image files, performs a stratified split
    with the configured seed, and saves the result as a JSON file. For the
    repeated stratified k-fold mode (``split.mode: kfold``) use
    :func:`prepare_kfold_splits` instead (``src/main.py`` dispatches on the mode).

    Args:
        config: Configuration dictionary with 'classes' and 'split' sections

    Returns:
        Path to the generated split JSON file.

    Raises:
        FileNotFoundError: If class directories don't exist
        ValueError: If configuration is invalid or directories are empty
        FileExistsError: If split file exists and force=false
    """
    classes_config = config["classes"]
    split_config = config["split"]

    seed = split_config.get("seed")
    train_ratio = split_config["train_ratio"]
    val_ratio = split_config["val_ratio"]
    test_ratio = split_config["test_ratio"]
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

    # Build class-to-label mapping from explicit config labels
    class_to_label = {name: entry["label"] for name, entry in classes_config.items()}
    # Sort by label for deterministic iteration order
    sorted_class_names = sorted(classes_config.keys(), key=lambda n: class_to_label[n])

    logger.info("=" * 60)
    logger.info("DATA PREPARATION - Train/Val/Test Split")
    logger.info("=" * 60)
    logger.info(f"Seed: {seed}")
    logger.info(f"Train ratio: {train_ratio}")
    logger.info(f"Val ratio: {val_ratio}")
    logger.info(f"Test ratio: {test_ratio}")
    logger.info(f"Classes: {sorted_class_names}")

    # Scan and split each class
    all_train = []
    all_val = []
    all_test = []
    class_samples = {}
    source_paths = {}

    for class_name in sorted_class_names:
        class_path = classes_config[class_name]["path"]
        label = class_to_label[class_name]
        source_paths[class_name] = class_path

        logger.info(f"Scanning class '{class_name}': {class_path}")

        # Scan for image files
        image_files = _scan_image_files(class_path)
        logger.info(f"  Found {len(image_files)} images")

        # Split this class
        train_files, val_files, test_files = _split_list(
            image_files, train_ratio, val_ratio, rng
        )

        # Warn when a non-zero test fraction was requested but the class had
        # too few samples to populate it (the min-guarantee for train/val
        # consumes the remainder). Otherwise the empty "test" entries are silent.
        if test_ratio > 0 and len(test_files) == 0:
            logger.warning(
                f"  Class '{class_name}' has too few samples "
                f"({len(image_files)}) to populate the test split; "
                "its 'test' entries will be empty despite test_ratio > 0"
            )

        # Record per-class statistics
        class_samples[class_name] = {
            "total": len(image_files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

        # Add to combined lists with labels
        for path in train_files:
            all_train.append({"path": path, "label": label})
        for path in val_files:
            all_val.append({"path": path, "label": label})
        for path in test_files:
            all_test.append({"path": path, "label": label})

        logger.info(
            f"  Train: {len(train_files)}, Val: {len(val_files)}, "
            f"Test: {len(test_files)}"
        )

    # Build JSON structure
    split_data = _build_split_dict(
        all_train,
        all_val,
        all_test,
        class_to_label,
        class_samples,
        source_paths,
        {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
    )

    # Write JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    total_samples = split_data["metadata"]["total_samples"]
    logger.info("")
    logger.info(f"Split file saved: {output_path}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Train samples: {len(all_train)}")
    logger.info(f"Val samples: {len(all_val)}")
    logger.info(f"Test samples: {len(all_test)}")
    logger.info("=" * 60)

    return str(output_path)


def prepare_kfold_splits(config: Dict[str, Any]) -> List[str]:
    """Generate repeated stratified k-fold splits (kfold mode).

    Scans each class once, then for every (repeat, fold) emits a stratified
    train/val/test JSON. Within a repeat the test folds partition the data, so
    each sample is tested exactly once per repeat. ``val`` is carved
    deterministically (stratified) from each fold's training pool.

    Returns:
        List of generated split-file paths, ordered by split index
        (``index = repeat * n_folds + fold``).
    """
    classes_config = config["classes"]
    split_config = config["split"]

    n_folds = split_config["n_folds"]
    n_repeats = split_config["n_repeats"]
    repeat_seeds = split_config["repeat_seeds"]
    val_fraction = split_config["val_fraction"]
    save_dir = split_config["save_dir"]
    split_file = split_config["split_file"]
    force = split_config.get("force", False)

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_label = {name: entry["label"] for name, entry in classes_config.items()}
    sorted_class_names = sorted(classes_config.keys(), key=lambda n: class_to_label[n])

    logger.info("=" * 60)
    logger.info("DATA PREPARATION - Repeated Stratified K-Fold")
    logger.info("=" * 60)
    logger.info(f"n_folds: {n_folds}, n_repeats: {n_repeats}")
    logger.info(f"repeat_seeds: {repeat_seeds}")
    logger.info(f"val_fraction: {val_fraction}")
    logger.info(f"Classes: {sorted_class_names}")

    # Scan every class once (reused across all repeats/folds).
    class_files: Dict[str, List[str]] = {}
    source_paths: Dict[str, str] = {}
    for class_name in sorted_class_names:
        class_path = classes_config[class_name]["path"]
        source_paths[class_name] = class_path
        logger.info(f"Scanning class '{class_name}': {class_path}")
        files = _scan_image_files(class_path)
        logger.info(f"  Found {len(files)} images")
        class_files[class_name] = files

        # Stratified k-fold needs at least one sample per fold to populate every
        # test fold for this class. With fewer, _kfold_chunks leaves some folds
        # empty, so the class silently drops out of those test sets -- shrinking
        # the across-split n the robustness CI is meant to report. Warn (don't
        # raise) since a legitimately tiny class is a valid, if degraded, input.
        if len(files) < n_folds:
            logger.warning(
                f"  Class '{class_name}' has fewer samples ({len(files)}) than "
                f"n_folds ({n_folds}); {n_folds - len(files)} test fold(s) will "
                "have no samples of this class (incomplete stratification)"
            )

    output_paths: List[str] = []

    for repeat, repeat_seed in enumerate(repeat_seeds):
        rng = random.Random(repeat_seed)
        # Pre-compute per-class fold partitions for this repeat's shuffle.
        class_folds: Dict[str, List[List[str]]] = {}
        for class_name in sorted_class_names:
            shuffled = class_files[class_name].copy()
            rng.shuffle(shuffled)
            class_folds[class_name] = _kfold_chunks(shuffled, n_folds)

        for fold in range(n_folds):
            index = repeat * n_folds + fold
            output_path = output_dir / split_file.format(index=index)

            if output_path.exists() and not force:
                logger.info(
                    f"Split file already exists: {output_path} (skipping; "
                    "set split.force=true to overwrite)"
                )
                output_paths.append(str(output_path))
                continue

            all_train: List[Dict[str, Any]] = []
            all_val: List[Dict[str, Any]] = []
            all_test: List[Dict[str, Any]] = []
            class_samples: Dict[str, Dict[str, int]] = {}

            for class_name in sorted_class_names:
                label = class_to_label[class_name]
                folds_c = class_folds[class_name]
                test_files = folds_c[fold]
                # Training pool = all other folds, in deterministic fold order.
                pool: List[str] = []
                for j in range(n_folds):
                    if j != fold:
                        pool.extend(folds_c[j])

                # Carve a stratified val portion from the training pool.
                val_n = round(len(class_files[class_name]) * val_fraction)
                if len(pool) > 1:
                    val_n = max(1, min(val_n, len(pool) - 1))
                else:
                    val_n = 0
                val_files = pool[:val_n]
                train_files = pool[val_n:]

                class_samples[class_name] = {
                    "total": len(class_files[class_name]),
                    "train": len(train_files),
                    "val": len(val_files),
                    "test": len(test_files),
                }
                for path in train_files:
                    all_train.append({"path": path, "label": label})
                for path in val_files:
                    all_val.append({"path": path, "label": label})
                for path in test_files:
                    all_test.append({"path": path, "label": label})

            split_data = _build_split_dict(
                all_train,
                all_val,
                all_test,
                class_to_label,
                class_samples,
                source_paths,
                {
                    "mode": "kfold",
                    "split_index": index,
                    "repeat": repeat,
                    "fold": fold,
                    "n_folds": n_folds,
                    "n_repeats": n_repeats,
                    "repeat_seed": repeat_seed,
                    "val_fraction": val_fraction,
                },
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"[split {index}] repeat={repeat} fold={fold} -> {output_path} "
                f"(train {len(all_train)}, val {len(all_val)}, test {len(all_test)})"
            )
            output_paths.append(str(output_path))

    logger.info("")
    logger.info(f"Generated {len(output_paths)} k-fold split files in {output_dir}")
    logger.info("=" * 60)

    return output_paths
