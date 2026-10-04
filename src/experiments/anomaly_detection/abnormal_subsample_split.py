"""Anomaly Detection - Splits with k Abnormal Training Images

Derives, from each extended AD split (``cv_binary_ad_split{N}.json``, see
``splits.py``), copies whose ``train`` partition keeps only ``k`` of its
abnormal (label-1) images. Everything else is unchanged: the normal ``train``
entries, ``val``, ``test`` and ``normal_extra_*``. This gives a learning curve
over ``k`` on the same test folds as the full-data runs.

``val`` keeps all of its abnormal images, so validation metrics stay defined
(and AD thresholds keep their normal validation images). A run that claims to
use only ``k`` labelled abnormal images must therefore not select its
checkpoint by a validation metric: use the last epoch (``final_model.pth``),
not ``best_model.pth``, or the validation abnormals add labelled images.

Which images are kept is random, so each ``k`` comes in several *draws*. Draws
are nested: draw ``d`` shuffles the split's abnormal training images once
(sorted by path, then ``random.Random("<repeat_seed>-split<N>-draw<d>")``) and
each ``k`` keeps the first ``k`` of that order, so the 5 images of ``k=5`` are
among the 10 of ``k=10`` in the same draw. Differences along ``k`` then come
from the added images, not from a new random choice.

Outputs are ``cv_binary_ad_split{N}_k{K}_d{D}.json``; each records ``k``, the
draw, the kept paths and its source file with SHA-256 under
``metadata.abnormal_subsample``.

Run:
    python -m src.experiments.anomaly_detection.abnormal_subsample_split \\
        --src-dir work/anomaly-detection/shared/splits \\
        --out-dir work/anomaly-detection/shared/splits-kctc \\
        --k 1 2 5 10 20 40 --draws 3
"""

import argparse
import copy
import hashlib
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Binary convention of the CV splits (enforced by ``splits.py``).
ABNORMAL_LABEL = 1
SOURCE_NAME = re.compile(r"split(\d+)$")


def draw_order(split: Dict[str, Any], draw: int) -> List[str]:
    """The abnormal training paths of a split in the random order of one draw.

    Raises:
        ValueError: If the split lacks ``train``, ``metadata.repeat_seed`` or
            ``metadata.split_index``, or ``draw`` is negative.
    """
    if draw < 0:
        raise ValueError(f"draw must be >= 0, got {draw}")
    meta = split.get("metadata", {})
    for key in ("repeat_seed", "split_index"):
        if key not in meta:
            raise ValueError(f"Source split has no metadata.{key}")
    if "train" not in split:
        raise ValueError("Source split is missing the 'train' partition")
    paths = sorted(e["path"] for e in split["train"] if e["label"] == ABNORMAL_LABEL)
    rng = random.Random(f"{meta['repeat_seed']}-split{meta['split_index']}-draw{draw}")
    rng.shuffle(paths)
    return paths


def build_subsampled_split(split: Dict[str, Any], k: int, draw: int) -> Dict[str, Any]:
    """Copy of ``split`` whose ``train`` keeps ``k`` abnormal images of a draw.

    Raises:
        ValueError: If ``k`` is not a positive integer, the split has fewer
            than ``k`` abnormal training images, or ``draw_order`` rejects it.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    order = draw_order(split, draw)
    if k > len(order):
        raise ValueError(
            f"k={k} exceeds the {len(order)} abnormal training images of split "
            f"{split['metadata']['split_index']}"
        )
    kept = set(order[:k])
    out = copy.deepcopy(split)
    out["train"] = [
        e for e in split["train"] if e["label"] != ABNORMAL_LABEL or e["path"] in kept
    ]
    out["metadata"]["abnormal_subsample"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "draw": draw,
        "n_abnormal_train_available": len(order),
        "kept_paths": order[:k],
    }
    return out


def generate_subsampled_splits(
    src_files: List[str],
    out_dir: str,
    ks: Sequence[int],
    draws: int,
    force: bool = False,
) -> List[Path]:
    """Write ``cv_binary_ad_split{N}_k{K}_d{D}.json`` for every source, k, draw.

    Everything is built and checked before anything is written, so a failure
    leaves the output directory untouched.

    Raises:
        ValueError: If there are no sources, ``ks`` is empty or repeats a
            value, ``draws`` < 1, a source name does not end in a split index,
            two sources share an index, or a split has too few abnormal
            training images for the largest ``k``.
        FileExistsError: If an output exists and ``force`` is False.
    """
    if not src_files:
        raise ValueError("No source split files given")
    if not ks or len(set(ks)) != len(ks):
        raise ValueError(f"ks must be a non-empty list without repeats, got {ks}")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError(f"draws must be a positive integer, got {draws!r}")

    out_path = Path(out_dir)
    planned: List[Tuple[Path, Dict[str, Any]]] = []
    indices = set()
    for src in src_files:
        match = SOURCE_NAME.search(Path(src).stem)
        if not match:
            raise ValueError(f"Cannot read the split index from {src}")
        index = str(int(match.group(1)))  # split01 and split1 are the same split
        if index in indices:
            raise ValueError(f"Two source files have split index {index}")
        indices.add(index)
        raw = Path(src).read_bytes()
        split = json.loads(raw)
        provenance = {
            "source_split_file": str(src),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        }
        for k in ks:
            for draw in range(draws):
                out = build_subsampled_split(split, k, draw)
                out["metadata"]["abnormal_subsample"].update(provenance)
                target = out_path / f"cv_binary_ad_split{index}_k{k}_d{draw}.json"
                planned.append((target, out))

    existing = [str(t) for t, _ in planned if t.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Outputs exist (use --force to overwrite; nothing was written): {existing}"
        )
    out_path.mkdir(parents=True, exist_ok=True)
    for target, split in planned:
        target.write_text(json.dumps(split, indent=2), encoding="utf-8")
    logger.info(
        f"Wrote {len(planned)} files: {len(src_files)} splits x k {list(ks)} x "
        f"{draws} draws -> {out_path}"
    )
    return [t for t, _ in planned]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Derive AD splits whose train keeps k abnormal images"
    )
    parser.add_argument("--src-dir", required=True, help="Directory with AD splits")
    parser.add_argument(
        "--pattern",
        default="cv_binary_ad_split*.json",
        help="Glob for source split files (default: cv_binary_ad_split*.json); "
        "only names ending in the split index are read",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--k", type=int, nargs="+", required=True, help="Abnormal training counts"
    )
    parser.add_argument("--draws", type=int, required=True, help="Draws per k")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Only names ending in the split index: this skips earlier outputs
    # (``..._k{K}_d{D}.json``), which the default pattern also matches.
    src_files = sorted(
        str(p)
        for p in Path(args.src_dir).glob(args.pattern)
        if SOURCE_NAME.search(p.stem)
    )
    generate_subsampled_splits(
        src_files, args.out_dir, args.k, args.draws, force=args.force
    )


if __name__ == "__main__":
    main()
