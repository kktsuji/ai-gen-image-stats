"""ROI extraction: catalog cells -> lossless per-channel 16-bit ROIs.

Each cell's ROI is cut from the 16-bit tile TIFFs of its sample and saved as
one 16-bit grayscale PNG per channel (``<roi_name>_{b,g,r}.png``). Nothing is
rescaled, so every rendering method starts from the same pixels.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from src.experiments.roi_extraction.common import (
    file_sha256,
    load_catalog,
    prepare_output_dir,
    provenance,
    write_json,
)
from src.experiments.roi_extraction.config import CHANNEL_KEYS

logger = logging.getLogger(__name__)


def roi_window(center: int, size: int, length: int) -> Tuple[int, int]:
    """Window ``[start, stop)`` of ``size`` around ``center`` along one axis.

    Near an image edge the window is shifted inward so it keeps its size, as
    in the original roi-generator. A center outside the axis is an error
    (e.g. swapped x/y), not clamped to the edge.
    """
    if not 0 <= center < length:
        raise ValueError(f"ROI center {center} lies outside an axis of {length}")
    half = size // 2
    start = max(center - half, 0)
    stop = min(center + half, length)
    if stop - start < size:
        if start == 0:
            stop = min(stop + (size - (stop - start)), length)
        else:
            start = max(start - (size - (stop - start)), 0)
    if stop - start != size:
        raise ValueError(
            f"Cannot fit a {size}-pixel ROI around {center} in an axis of {length}"
        )
    return start, stop


def find_image_dir(image_root: str, sample: str, pattern: str) -> Path:
    """The single directory of ``sample`` matching ``pattern`` (e.g. ``*after*``)."""
    matches = sorted(p for p in (Path(image_root) / sample).glob(pattern) if p.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{pattern}' directory for sample '{sample}' "
            f"under {image_root}, found {[str(m) for m in matches]}"
        )
    return matches[0]


def find_tile(image_dir: Path, img_no: int, suffix: str) -> Path:
    """The tile TIFF ``*_sNNN_<suffix>.tif`` of ``image_dir``."""
    matches = sorted(image_dir.glob(f"*_s{img_no:03d}_{suffix}.tif"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one tile '*_s{img_no:03d}_{suffix}.tif' in "
            f"{image_dir}, found {len(matches)}"
        )
    return matches[0]


def read_uint16(path: Path) -> np.ndarray:
    """Read a single-channel 16-bit image (either byte order) as native uint16."""
    with Image.open(path) as img:
        array = np.array(img)
    if array.dtype.kind != "u" or array.dtype.itemsize != 2 or array.ndim != 2:
        raise ValueError(
            f"Expected a single-channel 16-bit image: {path} "
            f"(got dtype {array.dtype}, shape {array.shape})"
        )
    return array.astype(np.uint16, copy=False)


def write_uint16_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(array, dtype=np.uint16)).save(path)


def run_extract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Cut every catalog cell into ``data.raw16_dir`` (``mode: extract``)."""
    data, extraction = config["data"], config["extraction"]
    # Sorted by tile, so each tile is read once even when several sources share it
    cells = sorted(
        load_catalog(data["catalog"])["cells"],
        key=lambda c: (c["sample"], c["img_no"]),
    )
    out_dir = prepare_output_dir(
        data["raw16_dir"],
        config["output"]["force"],
        inputs=[data["catalog"], data["image_root"], data["coordinates_root"]],
    )
    size = extraction["roi_size"]
    channels = extraction["channels"]

    image_dirs: Dict[str, Path] = {}
    tiles: Dict[str, np.ndarray] = {}
    tile_key = None
    for i, cell in enumerate(cells, 1):
        sample = cell["sample"]
        if sample not in image_dirs:
            image_dirs[sample] = find_image_dir(
                data["image_root"], sample, extraction["image_dir_glob"]
            )
        if tile_key != (sample, cell["img_no"]):
            tile_key = (sample, cell["img_no"])
            tiles = {
                key: read_uint16(find_tile(image_dirs[sample], cell["img_no"], suffix))
                for key, suffix in channels.items()
            }
            shapes = {tile.shape for tile in tiles.values()}
            if len(shapes) != 1:
                raise ValueError(
                    f"Channel tiles of sample '{sample}', tile {cell['img_no']} "
                    f"differ in shape: { {k: t.shape for k, t in tiles.items()} }"
                )
        height, width = tiles[CHANNEL_KEYS[0]].shape
        y1, y2 = roi_window(cell["y"], size, height)
        x1, x2 = roi_window(cell["x"], size, width)
        for key in CHANNEL_KEYS:
            write_uint16_png(
                out_dir / f"{cell['roi_name']}_{key}.png", tiles[key][y1:y2, x1:x2]
            )
        if i % 500 == 0:
            logger.info(f"  {i}/{len(cells)} ROIs")

    report = {
        **provenance(),
        "catalog": data["catalog"],
        "catalog_sha256": file_sha256(data["catalog"]),
        "image_root": data["image_root"],
        "image_dirs": {
            s: d.relative_to(Path(data["image_root"])).as_posix()
            for s, d in image_dirs.items()
        },
        "roi_size": size,
        "channels": dict(channels),
        "cell_counts": dict(Counter(c["source"] for c in cells)),
    }
    write_json(out_dir / "extraction.json", report)
    logger.info(f"Extracted {len(cells)} ROIs -> {out_dir}")
    return report
