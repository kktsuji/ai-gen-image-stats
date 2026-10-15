"""ROI rendering: lossless 16-bit ROIs -> training images.

``rendering.method`` names the intensity transform, so other transforms can be
added later without touching the catalog or the extracted ROIs.

``pseudo_rgb8_vstretch`` is the transform of the in-house training images:
the b/g/r channels are scaled linearly from 16 to 8 bits (round half up), then,
when ``rendering.vstretch`` is set, the HSV value channel of each image is
stretched between two of its own percentiles (OpenCV HSV conversion, so hue
and saturation, i.e. the channel ratios, are kept).
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

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
from src.experiments.roi_extraction.extract import read_uint16

logger = logging.getLogger(__name__)

_MAX_IN = 2**16 - 1
_MAX_OUT = 2**8 - 1


def to_uint8(bgr16: np.ndarray) -> np.ndarray:
    """Scale 16-bit values linearly to 8 bits, rounding half up."""
    scaled = bgr16.astype(np.float64) * _MAX_OUT / _MAX_IN
    scaled = np.floor(scaled + 0.5)
    return np.clip(scaled, 0, _MAX_OUT).astype(np.uint8)


def vstretch(
    bgr8: np.ndarray, lower_percentile: float, upper_percentile: float
) -> np.ndarray:
    """Stretch the HSV value channel between two of the image's percentiles."""
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "rendering.vstretch needs OpenCV: pip install -r requirements-data.txt"
        ) from e

    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lower = np.percentile(v, lower_percentile)
    upper = np.percentile(v, upper_percentile)
    if upper <= lower:
        raise ValueError(
            f"Cannot stretch: value percentiles {lower_percentile} and "
            f"{upper_percentile} are equal ({lower})"
        )
    v_stretched = np.clip((v - lower) * (255.0 / (upper - lower)), 0, 255)
    v_stretched = v_stretched.astype(np.uint8)
    return np.asarray(cv2.cvtColor(cv2.merge([h, s, v_stretched]), cv2.COLOR_HSV2BGR))


def render_pseudo_rgb8_vstretch(
    bgr16: np.ndarray, stretch: Optional[Dict[str, Any]]
) -> np.ndarray:
    """Apply ``pseudo_rgb8_vstretch`` to a BGR uint16 image; returns BGR uint8."""
    bgr8 = to_uint8(bgr16)
    if stretch is None:
        return bgr8
    return vstretch(bgr8, stretch["lower_percentile"], stretch["upper_percentile"])


def load_bgr16(raw16_dir: Path, name: str) -> np.ndarray:
    """Stack the per-channel ROIs of ``name`` into a BGR uint16 image."""
    return np.stack(
        [read_uint16(raw16_dir / f"{name}_{key}.png") for key in CHANNEL_KEYS],
        axis=-1,
    )


def check_extraction(raw16_dir: Path, catalog_path: str) -> None:
    """Require ``raw16_dir`` to have been extracted from this exact catalog."""
    report_path = raw16_dir / "extraction.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"No extraction report in {raw16_dir} (run with --mode extract first)"
        )
    with open(report_path, encoding="utf-8") as f:
        extracted_from = json.load(f)["catalog_sha256"]
    if extracted_from != file_sha256(catalog_path):
        raise ValueError(
            f"{raw16_dir} was extracted from another version of {catalog_path}; "
            "re-run --mode extract"
        )


def run_render(config: Dict[str, Any]) -> Dict[str, Any]:
    """Render every catalog cell into ``rendering.output_dir`` (``mode: render``)."""
    data, rendering = config["data"], config["rendering"]
    catalog = load_catalog(data["catalog"])
    cells = catalog["cells"]
    raw16_dir = Path(data["raw16_dir"])
    if (
        rendering["layout"] == "class_paths"
        and catalog["metadata"]["class_layout_dir"] is None
    ):
        raise ValueError(
            "rendering.layout: class_paths needs a catalog built with "
            "catalog.class_layout_dir (this one has no class paths); use "
            "layout: pool instead"
        )
    check_extraction(raw16_dir, data["catalog"])
    out_dir = prepare_output_dir(
        rendering["output_dir"],
        config["output"]["force"],
        inputs=[data["catalog"], data["raw16_dir"]]
        + [p for p in [catalog["metadata"]["class_layout_dir"]] if p],
    )

    written: Counter = Counter()
    skipped = []
    for cell in cells:
        if rendering["layout"] == "pool":
            targets = [cell["source"]]
        else:
            targets = cell["class_paths"]
        if not targets:
            skipped.append(cell["roi_name"])
            continue
        bgr8 = render_pseudo_rgb8_vstretch(
            load_bgr16(raw16_dir, cell["roi_name"]), rendering["vstretch"]
        )
        rgb8 = Image.fromarray(np.ascontiguousarray(bgr8[..., ::-1]))
        for target in targets:
            target_dir = out_dir / target
            target_dir.mkdir(parents=True, exist_ok=True)
            rgb8.save(target_dir / f"{cell['roi_name']}.png")
            written[target] += 1

    report = {
        **provenance(),
        "method": rendering["method"],
        "vstretch": rendering["vstretch"],
        "layout": rendering["layout"],
        "catalog": data["catalog"],
        "catalog_sha256": file_sha256(data["catalog"]),
        "raw16_dir": data["raw16_dir"],
        "written_counts": dict(sorted(written.items())),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    write_json(out_dir / "rendering.json", report)
    logger.info(
        f"Rendered {sum(written.values())} images -> {out_dir} "
        f"({len(skipped)} cells without a target skipped)"
    )
    return report
