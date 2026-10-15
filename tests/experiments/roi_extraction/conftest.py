"""Synthetic inputs for the roi_extraction tests (two samples, 16-bit tiles)."""

import numpy as np
import pytest
from PIL import Image

from tests.experiments.roi_extraction.test_config import make_config

TILE_SHAPE = (12, 16)  # rows, columns
SUFFIXES = {"b": "ch01", "g": "ch02", "r": "ch03"}

# sample -> source CSV name -> rows (img_no, x, y)
COORDINATES = {
    "S 1": {
        "groundTruthCTCs.csv": [(1, 5, 6), (2, 0, 0)],
        "normalCells.csv": [(1, 15, 11), (3, 8, 2)],
    },
    "S 2": {
        "groundTruthCTCs.csv": [],
        "normalCells.csv": [(1, 7, 7)],
    },
}


def tile_values(sample_index: int, img_no: int, channel_index: int) -> np.ndarray:
    """Distinct 16-bit values per (sample, tile, channel, pixel)."""
    rows, cols = TILE_SHAPE
    base = sample_index * 20000 + img_no * 4000 + channel_index * 1000
    return (base + np.arange(rows * cols).reshape(rows, cols) * 3).astype(np.uint16)


def write_inputs(root) -> dict:
    coords_root = root / "coords"
    raw_root = root / "raw"
    for sample_index, (sample, csvs) in enumerate(COORDINATES.items()):
        (coords_root / sample).mkdir(parents=True)
        for csv_name, rows in csvs.items():
            text = "".join(f"{n},{x},{y}\n" for n, x, y in rows)
            (coords_root / sample / csv_name).write_text(text)
        tile_dir = raw_root / sample / "2 (after)"
        tile_dir.mkdir(parents=True)
        (raw_root / sample / "1 (before)").mkdir()
        for img_no in (1, 2, 3):
            for channel_index in range(4):
                values = tile_values(sample_index, img_no, channel_index)
                name = f"{sample} after.lif_s{img_no:03d}_ch{channel_index:02d}.tif"
                Image.fromarray(values).save(tile_dir / name)
    return {"coords": coords_root, "raw": raw_root}


@pytest.fixture
def roi_inputs(tmp_path):
    """Synthetic inputs plus a config pointing at them (``mode`` to be set)."""
    paths = write_inputs(tmp_path)
    config = make_config()
    config["data"] = {
        "coordinates_root": str(paths["coords"]),
        "image_root": str(paths["raw"]),
        "catalog": str(tmp_path / "out" / "cells.json"),
        "raw16_dir": str(tmp_path / "out" / "raw16"),
    }
    config["rendering"]["output_dir"] = str(tmp_path / "out" / "rendered")
    return config
