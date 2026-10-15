"""Cell catalog: the dataset definition for ROI extraction.

Reads the CAD coordinate CSVs (one directory per sample, header-less
``img_no,x,y`` rows) and writes one JSON entry per cell. When a class layout
directory is given (e.g. the existing training images, ``<class path>/<roi
name>.png``), each cell also records the class paths its image is filed under,
so the layout can be rebuilt from the catalog alone.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiments.roi_extraction.common import (
    file_sha256,
    provenance,
    write_json,
)

logger = logging.getLogger(__name__)

COORDINATE_SYSTEM = (
    "img_no = tile number (_sNNN_ in the TIFF file names); x = column, y = row, "
    "in pixels, origin at the top-left of the tile"
)


def roi_name(sample: str, img_no: int, x: int, y: int) -> str:
    """ROI file stem, as named by the original roi-generator."""
    return f"{sample.replace(' ', '-')}_img-no{img_no:03d}_y{y}_x{x}"


def read_coordinates(path: Path) -> List[Dict[str, int]]:
    """Parse a header-less ``img_no,x,y`` CSV (blank lines are skipped)."""
    rows = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(",")
        try:
            img_no, x, y = (int(v) for v in fields)
        except ValueError:
            raise ValueError(
                f"{path}:{line_no}: expected three integers 'img_no,x,y', got {line!r}"
            ) from None
        rows.append({"img_no": img_no, "x": x, "y": y})
    return rows


def scan_class_layout(layout_dir: str) -> Dict[str, List[str]]:
    """Map each image stem under ``layout_dir`` to the directories holding it."""
    root = Path(layout_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Class layout directory not found: {layout_dir}")
    layout: Dict[str, List[str]] = {}
    for path in sorted(root.rglob("*.png")):
        class_path = path.parent.relative_to(root).as_posix()
        layout.setdefault(path.stem, []).append(class_path)
    if not layout:
        raise ValueError(f"No PNG images found under: {layout_dir}")
    return layout


def build_catalog(
    coordinates_root: str,
    sources: Dict[str, str],
    class_layout_dir: Optional[str],
) -> Dict[str, Any]:
    """Build the catalog (metadata + cells) from the coordinate CSVs."""
    root = Path(coordinates_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_root}")
    samples = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not samples:
        raise ValueError(f"No sample directories in: {coordinates_root}")

    cells: List[Dict[str, Any]] = []
    source_files: Dict[str, str] = {}
    for sample in samples:
        for pool, csv_name in sources.items():
            csv_path = root / sample / csv_name
            if not csv_path.is_file():
                raise FileNotFoundError(f"Coordinate file not found: {csv_path}")
            source_files[f"{sample}/{csv_name}"] = file_sha256(csv_path)
            for row in read_coordinates(csv_path):
                cells.append(
                    {
                        "sample": sample,
                        **row,
                        "source": pool,
                        "roi_name": roi_name(sample, **row),
                        "class_paths": [],
                    }
                )

    duplicates = [n for n, c in Counter(c["roi_name"] for c in cells).items() if c > 1]
    if duplicates:
        raise ValueError(
            f"ROI names must be unique across sources, duplicated: {duplicates[:5]}"
        )

    class_counts: Dict[str, int] = {}
    if class_layout_dir is not None:
        layout = scan_class_layout(class_layout_dir)
        by_name = {c["roi_name"]: c for c in cells}
        unknown = sorted(set(layout) - set(by_name))
        if unknown:
            raise ValueError(
                f"{len(unknown)} image(s) in {class_layout_dir} match no cell of "
                f"the catalog, e.g. {unknown[:5]}"
            )
        for name, class_paths in layout.items():
            by_name[name]["class_paths"] = class_paths
            for class_path in class_paths:
                class_counts[class_path] = class_counts.get(class_path, 0) + 1

    cell_counts = Counter(c["source"] for c in cells)
    unassigned = Counter(c["source"] for c in cells if not c["class_paths"])
    return {
        "metadata": {
            **provenance(),
            "coordinates_root": coordinates_root,
            "coordinate_system": COORDINATE_SYSTEM,
            "sources": dict(sources),
            "source_files_sha256": source_files,
            "class_layout_dir": class_layout_dir,
            "cell_counts": {pool: cell_counts[pool] for pool in sources},
            "class_counts": dict(sorted(class_counts.items())),
            "unassigned_counts": (
                {pool: unassigned[pool] for pool in sources}
                if class_layout_dir is not None
                else None
            ),
        },
        "cells": cells,
    }


def run_catalog(config: Dict[str, Any]) -> Dict[str, Any]:
    """Write the catalog to ``data.catalog`` (``mode: catalog``)."""
    out_path = Path(config["data"]["catalog"])
    if out_path.exists() and not config["output"]["force"]:
        raise FileExistsError(
            f"Catalog already exists: {out_path} (set output.force: true to replace it)"
        )
    catalog = build_catalog(
        config["data"]["coordinates_root"],
        config["sources"],
        config["catalog"]["class_layout_dir"],
    )
    write_json(out_path, catalog)
    meta = catalog["metadata"]
    logger.info(f"Catalog: {len(catalog['cells'])} cells -> {out_path}")
    logger.info(f"  cells per source: {meta['cell_counts']}")
    if meta["unassigned_counts"] is not None:
        logger.info(f"  images per class path: {meta['class_counts']}")
        logger.info(f"  cells without a class path: {meta['unassigned_counts']}")
    return catalog
