"""Anomaly Detection - Normal-Subclass Classification Splits

Derives, from each extended AD split (``cv_binary_ad_split{N}.json``, see
``splits.py``), a split file for classifying the *normal* subclasses only
(suspicious, red, green, junk, blur, blue). A backbone fine-tuned on this task
learns cell morphology from normal labels alone; no abnormal image enters its
training, validation or test partitions, so it can serve as a representation
for one-class anomaly detection without using any CTC label.

Partitions follow the AD split, so nothing held out there is trained on here:

- ``train`` = the suspicious entries of ``train`` + ``normal_extra_train``
- ``val``   = the suspicious entries of ``val``   + ``normal_extra_val``
- ``test``  = the suspicious entries of ``test``  + ``normal_extra_test``

Labels are class indices in alphabetical order of the subclass names, recorded
in ``metadata.classes``. The output is ``SplitFileDataset``-compatible.

Run:
    python -m src.experiments.anomaly_detection.normal_subclass_split \\
        --src-dir work/anomaly-detection/shared/splits \\
        --out-dir work/anomaly-detection/shared/splits-normal-subclass
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PARTITIONS = {
    "train": ("train", "normal_extra_train"),
    "val": ("val", "normal_extra_val"),
    "test": ("test", "normal_extra_test"),
}
# Metadata of the source split carried over unchanged (fold geometry).
CARRIED_METADATA = (
    "split_index",
    "repeat",
    "fold",
    "n_folds",
    "n_repeats",
    "repeat_seed",
    "val_fraction",
)


# Binary convention of the CV splits (enforced by ``splits.py``): label 1 is the
# abnormal class, label 0 the normal one. Abnormal images are dropped by label,
# never by name, so a differently named abnormal class cannot slip through.
ABNORMAL_LABEL = 1
NORMAL_LABEL = 0


def build_normal_subclass_split(ad_split: Dict[str, Any]) -> Dict[str, Any]:
    """Build the normal-subclass split from one extended AD split.

    Abnormal images are the label-1 entries of ``train``/``val``/``test``; the
    name of that class comes from ``metadata.classes``. ``normal_extra_*``
    entries must all be label 0.

    Args:
        ad_split: Parsed ``cv_binary_ad_split{N}.json``.

    Returns:
        Split dict with ``train``/``val``/``test`` and ``metadata``.

    Raises:
        ValueError: If the source is not a binary split with label 1 as its
            abnormal class, an abnormal entry's tag does not match that class,
            a ``normal_extra_*`` entry is not label 0, an entry lacks a
            subclass tag, a partition ends up empty, a subclass is missing from
            train, or a path appears in more than one partition.
    """
    binary_classes = ad_split.get("metadata", {}).get("classes", {})
    label_to_name = {label: name for name, label in binary_classes.items()}
    if sorted(label_to_name) != [NORMAL_LABEL, ABNORMAL_LABEL]:
        raise ValueError(
            "Expected a binary source split with labels {0, 1} in "
            f"metadata.classes, got {binary_classes}"
        )
    abnormal_name = label_to_name[ABNORMAL_LABEL]

    selected: Dict[str, List[Dict[str, Any]]] = {}
    for part, source_keys in PARTITIONS.items():
        entries = []
        for key in source_keys:
            for entry in ad_split[key]:
                if "subclass" not in entry:
                    raise ValueError(f"Entry without a subclass tag in '{key}'")
                if key.startswith("normal_extra_"):
                    if entry["label"] != NORMAL_LABEL:
                        raise ValueError(
                            f"{entry['path']} in '{key}' has label {entry['label']}, "
                            f"expected {NORMAL_LABEL}"
                        )
                elif entry["label"] == ABNORMAL_LABEL:
                    if entry["subclass"] != abnormal_name:
                        raise ValueError(
                            f"{entry['path']} in '{key}' has label "
                            f"{ABNORMAL_LABEL} but subclass "
                            f"'{entry['subclass']}', expected '{abnormal_name}'"
                        )
                    continue
                if entry["subclass"] == abnormal_name:
                    raise ValueError(
                        f"{entry['path']} in '{key}' is tagged '{abnormal_name}' "
                        f"but has label {entry['label']}"
                    )
                entries.append(entry)
        if not entries:
            raise ValueError(f"No normal entries for partition '{part}'")
        selected[part] = entries

    names = sorted({e["subclass"] for entries in selected.values() for e in entries})
    # A class absent from train would get a label the classifier never learns.
    untrained = sorted(set(names) - {e["subclass"] for e in selected["train"]})
    if untrained:
        raise ValueError(f"Subclasses {untrained} appear in val/test but not in train")
    classes = {name: i for i, name in enumerate(names)}
    out: Dict[str, Any] = {
        part: [
            {
                "path": e["path"],
                "label": classes[e["subclass"]],
                "subclass": e["subclass"],
            }
            for e in entries
        ]
        for part, entries in selected.items()
    }
    _check_no_overlap(out)

    source_meta = ad_split.get("metadata", {})
    out["metadata"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "classes": classes,
        "class_samples": {
            part: dict(sorted(Counter(e["subclass"] for e in out[part]).items()))
            for part in PARTITIONS
        },
        "train_samples": len(out["train"]),
        "val_samples": len(out["val"]),
        "test_samples": len(out["test"]),
        "dropped_subclass": abnormal_name,
        **{k: source_meta[k] for k in CARRIED_METADATA if k in source_meta},
    }
    return out


def _check_no_overlap(split: Dict[str, Any]) -> None:
    seen: Dict[str, str] = {}
    for part in PARTITIONS:
        for entry in split[part]:
            other = seen.setdefault(entry["path"], part)
            if other != part:
                raise ValueError(f"{entry['path']} is in both '{other}' and '{part}'")


def generate_normal_subclass_splits(
    src_files: List[str], out_dir: str, force: bool = False
) -> List[Path]:
    """Write ``normal_subclass_split{N}.json`` for each extended AD split file.

    All splits are built and checked before anything is written, so a failure
    leaves the output directory untouched: every split must map the same
    subclasses to the same labels (a label means one subclass across splits),
    and without ``force`` no output may exist yet. The source file and its
    SHA-256 are recorded in the metadata.

    Raises:
        ValueError: If there are no source files, a name does not end in a
            split index, two sources share an index, or the splits' class
            mappings differ.
        FileExistsError: If an output exists and ``force`` is False.
    """
    if not src_files:
        raise ValueError("No source split files given")
    out_path = Path(out_dir)
    planned: List[Tuple[Path, Dict[str, Any]]] = []
    for src in src_files:
        stem = Path(src).stem  # cv_binary_ad_split{N}
        index = stem.rsplit("split", 1)[-1]
        if not index.isdigit():
            raise ValueError(f"Cannot read the split index from {src}")
        target = out_path / f"normal_subclass_split{index}.json"
        if any(t == target for t, _ in planned):
            raise ValueError(f"Two source files map to {target.name}")
        raw = Path(src).read_bytes()
        split = build_normal_subclass_split(json.loads(raw))
        split["metadata"]["source_split_file"] = str(src)
        split["metadata"]["source_sha256"] = hashlib.sha256(raw).hexdigest()
        planned.append((target, split))

    reference = planned[0][1]["metadata"]["classes"]
    for target, split in planned[1:]:
        if split["metadata"]["classes"] != reference:
            raise ValueError(
                f"{target.name} maps classes {split['metadata']['classes']}, "
                f"but {planned[0][0].name} maps {reference}"
            )
    existing = [str(t) for t, _ in planned if t.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Outputs exist (use --force to overwrite; nothing was written): {existing}"
        )

    out_path.mkdir(parents=True, exist_ok=True)
    for target, split in planned:
        target.write_text(json.dumps(split, indent=2), encoding="utf-8")
        logger.info(
            f"{target.name}: train {len(split['train'])}, val {len(split['val'])}, "
            f"test {len(split['test'])}, classes {split['metadata']['classes']}"
        )
    return [t for t, _ in planned]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Derive normal-subclass classification splits from AD splits"
    )
    parser.add_argument("--src-dir", required=True, help="Directory with AD splits")
    parser.add_argument(
        "--pattern",
        default="cv_binary_ad_split*.json",
        help="Glob for source split files (default: cv_binary_ad_split*.json)",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    src_files = sorted(str(p) for p in Path(args.src_dir).glob(args.pattern))
    generate_normal_subclass_splits(src_files, args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
