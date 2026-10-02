"""Anomaly Detection - Extended Split Generation

Derives anomaly-detection split files from the binary CV split files
(``cv_binary_split{N}.json``, abnormal vs suspicious) without modifying them.

The binary ``train``/``val``/``test`` entries are inherited unchanged, so the
held-out test fold is identical to the classifier experiments and results can
be paired per split. The remaining normal subclasses (red, green, junk, blur,
blue), which never appear in the binary task, are partitioned with the same
repeated k-fold geometry (``repeat_seed``/``fold``/``n_folds``/``val_fraction``
read from each source file) and stored as ``normal_extra_{train,val,test}``.
Every entry is tagged with its ``subclass`` for per-subclass score analysis.

Run:
    python -m src.experiments.anomaly_detection.splits \\
        --src-dir /path/to/splits/cv-binary \\
        --normal-dir data/in-house/normal \\
        --path-remap data/ data/in-house/ \\
        --out-dir outputs/splits/cv-binary-ad
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.experiments.data_preparation.prepare import _kfold_chunks, _scan_image_files

logger = logging.getLogger(__name__)

DEFAULT_EXTRA_SUBCLASSES = ["red", "green", "junk", "blur", "blue"]
EXTRA_SPLIT_KEYS = ("normal_extra_train", "normal_extra_val", "normal_extra_test")


def _remap_path(path: str, remap: Optional[Tuple[str, str]]) -> str:
    """Replace a leading ``remap[0]`` prefix of ``path`` with ``remap[1]``."""
    if remap is None:
        return path
    old, new = remap
    if not path.startswith(old):
        raise ValueError(f"Path '{path}' does not start with remap prefix '{old}'")
    return new + path[len(old) :]


def _carve_val(pool: List[str], val_fraction: float) -> Tuple[List[str], List[str]]:
    """Split a training pool into (train, val), mirroring ``prepare_kfold_splits``."""
    val_n = round(len(pool) * val_fraction)
    if len(pool) > 1:
        val_n = max(1, min(val_n, len(pool) - 1))
    else:
        val_n = 0
    return pool[val_n:], pool[:val_n]


def split_extra_subclass(
    files: List[str],
    subclass: str,
    repeat_seed: int,
    fold: int,
    n_folds: int,
    val_fraction: float,
) -> Dict[str, List[str]]:
    """Partition one extra normal subclass into train/val/test for a given fold.

    The shuffle is seeded per (repeat_seed, subclass), so the fold assignment is
    deterministic and the test folds of one repeat partition the subclass.

    Returns:
        Dict with keys ``train``, ``val``, ``test`` (lists of file paths).
    """
    shuffled = sorted(files)
    random.Random(f"{repeat_seed}-{subclass}").shuffle(shuffled)
    folds = _kfold_chunks(shuffled, n_folds)
    pool: List[str] = []
    for j in range(n_folds):
        if j != fold:
            pool.extend(folds[j])
    train, val = _carve_val(pool, val_fraction)
    return {"train": train, "val": val, "test": folds[fold]}


def build_extended_split(
    binary_split: Dict[str, Any],
    extra_files: Dict[str, List[str]],
    path_remap: Optional[Tuple[str, str]] = None,
    source_file: str = "",
) -> Dict[str, Any]:
    """Build an anomaly-detection split dict from one binary CV split dict.

    Args:
        binary_split: Parsed binary split JSON (kfold mode).
        extra_files: Mapping subclass name -> list of image paths.
        path_remap: Optional (old_prefix, new_prefix) applied to inherited paths.
        source_file: Source split file path, recorded in metadata.

    Returns:
        Extended split dict with inherited ``train``/``val``/``test`` entries,
        ``normal_extra_{train,val,test}`` entries and an ``anomaly_detection``
        metadata block.
    """
    metadata = binary_split["metadata"]
    if metadata.get("mode") != "kfold":
        raise ValueError(
            f"Expected a kfold split file, got mode={metadata.get('mode')!r}"
        )
    label_to_class = {label: name for name, label in metadata["classes"].items()}
    # Binary convention of the CV splits: label 1 = abnormal, label 0 = normal.
    if sorted(label_to_class) != [0, 1]:
        raise ValueError(
            f"Expected a binary split with labels {{0, 1}}, got {metadata['classes']}"
        )
    normal_label = 0

    extended: Dict[str, Any] = {}
    for key in ("train", "val", "test"):
        extended[key] = [
            {
                "path": _remap_path(entry["path"], path_remap),
                "label": entry["label"],
                "subclass": label_to_class[entry["label"]],
            }
            for entry in binary_split[key]
        ]

    extra_counts: Dict[str, Dict[str, int]] = {}
    for key in EXTRA_SPLIT_KEYS:
        extended[key] = []
    for subclass in sorted(extra_files):
        parts = split_extra_subclass(
            extra_files[subclass],
            subclass,
            repeat_seed=metadata["repeat_seed"],
            fold=metadata["fold"],
            n_folds=metadata["n_folds"],
            val_fraction=metadata["val_fraction"],
        )
        extra_counts[subclass] = {part: len(paths) for part, paths in parts.items()}
        for part, paths in parts.items():
            extended[f"normal_extra_{part}"].extend(
                {"path": p, "label": normal_label, "subclass": subclass} for p in paths
            )

    _check_no_leakage(extended)

    extended["metadata"] = {
        **metadata,
        "anomaly_detection": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_split_file": source_file,
            "path_remap": list(path_remap) if path_remap else None,
            "extra_subclasses": extra_counts,
        },
    }
    return extended


def _check_no_leakage(extended: Dict[str, Any]) -> None:
    """Raise if any image path appears in more than one partition."""
    seen: Dict[str, str] = {}
    for key in ("train", "val", "test", *EXTRA_SPLIT_KEYS):
        for entry in extended[key]:
            path = entry["path"]
            if path in seen:
                raise ValueError(
                    f"Leakage: '{path}' appears in both '{seen[path]}' and '{key}'"
                )
            seen[path] = key


def _check_paths_exist(extended: Dict[str, Any]) -> None:
    """Raise if any referenced image file is missing."""
    missing = [
        entry["path"]
        for key in ("train", "val", "test", *EXTRA_SPLIT_KEYS)
        for entry in extended[key]
        if not Path(entry["path"]).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image file(s) referenced by the split do not exist, "
            f"e.g. {missing[0]} (check --path-remap)"
        )


def generate_extended_splits(
    src_files: Sequence[str],
    normal_dir: str,
    out_dir: str,
    extra_subclasses: Sequence[str] = tuple(DEFAULT_EXTRA_SUBCLASSES),
    path_remap: Optional[Tuple[str, str]] = None,
    force: bool = False,
    check_paths: bool = True,
) -> List[str]:
    """Generate one extended split file per binary split file.

    Output files are named after the source with ``cv_binary_`` replaced by
    ``cv_binary_ad_`` (other names get an ``ad_`` prefix). Existing outputs are
    kept unless ``force`` is set.

    Returns:
        List of output file paths (in the order of ``src_files``).
    """
    if not src_files:
        raise ValueError("No source split files given")

    extra_files = {
        subclass: _scan_image_files(str(Path(normal_dir) / subclass))
        for subclass in extra_subclasses
    }
    for subclass, files in extra_files.items():
        logger.info(f"Extra normal subclass '{subclass}': {len(files)} images")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[str] = []
    for src in src_files:
        src_name = Path(src).name
        if src_name.startswith("cv_binary_"):
            out_name = "cv_binary_ad_" + src_name[len("cv_binary_") :]
        else:
            out_name = "ad_" + src_name
        out_path = output_dir / out_name

        if out_path.exists() and not force:
            logger.info(f"Extended split exists: {out_path} (skipping; use --force)")
            outputs.append(str(out_path))
            continue

        with open(src, encoding="utf-8") as f:
            binary_split = json.load(f)
        extended = build_extended_split(
            binary_split, extra_files, path_remap=path_remap, source_file=str(src)
        )
        if check_paths:
            _check_paths_exist(extended)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(extended, f, indent=2, ensure_ascii=False)

        counts = {
            key: len(extended[key])
            for key in ("train", "val", "test", *EXTRA_SPLIT_KEYS)
        }
        logger.info(f"{src_name} -> {out_path} {counts}")
        outputs.append(str(out_path))

    return outputs


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Derive anomaly-detection splits from binary CV splits"
    )
    parser.add_argument(
        "--src-dir", required=True, help="Directory with binary split JSONs"
    )
    parser.add_argument(
        "--pattern",
        default="cv_binary_split*.json",
        help="Glob for source split files (default: cv_binary_split*.json)",
    )
    parser.add_argument(
        "--normal-dir",
        required=True,
        help="Directory holding the normal subclass folders",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--extra-subclasses",
        nargs="+",
        default=DEFAULT_EXTRA_SUBCLASSES,
        help="Normal subclasses to add (default: red green junk blur blue)",
    )
    parser.add_argument(
        "--path-remap",
        nargs=2,
        metavar=("OLD", "NEW"),
        default=None,
        help="Rewrite the leading OLD prefix of inherited paths to NEW",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite outputs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    src_files = sorted(Path(args.src_dir).glob(args.pattern))
    generate_extended_splits(
        src_files=[str(p) for p in src_files],
        normal_dir=args.normal_dir,
        out_dir=args.out_dir,
        extra_subclasses=args.extra_subclasses,
        path_remap=tuple(args.path_remap) if args.path_remap else None,
        force=args.force,
    )


if __name__ == "__main__":
    main()
