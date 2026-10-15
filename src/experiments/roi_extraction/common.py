"""Helpers shared by the roi_extraction modes."""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.utils.checkpoint import file_sha256
from src.utils.git import get_git_commit_hash

__all__ = [
    "check_no_overlap",
    "file_sha256",
    "load_catalog",
    "prepare_output_dir",
    "protected_paths",
    "rois_sha256",
    "provenance",
    "write_json",
]


def provenance() -> Dict[str, Any]:
    """Creation time and commit, recorded in every output's metadata."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit_hash(),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_catalog(path: str) -> Dict[str, Any]:
    """Read a catalog (``metadata`` + ``cells``) written by ``mode: catalog``."""
    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Catalog not found: {path} (run with --mode catalog first)"
        )
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def rois_sha256(cells: List[Dict[str, Any]]) -> str:
    """Hash of what the extraction depends on (each cell's position and name).

    Order-independent; rebuilding an unchanged catalog, or changing only its
    class paths, keeps this hash, so the extracted ROIs stay valid.
    """
    keys = ("sample", "img_no", "x", "y", "roi_name")
    rois = sorted([cell[k] for k in keys] for cell in cells)
    return hashlib.sha256(json.dumps(rois).encode("utf-8")).hexdigest()


def protected_paths(config: Dict[str, Any], output_key: str) -> Dict[str, str]:
    """Every path of the config other than the output ``output_key`` itself."""
    paths = {f"data.{k}": v for k, v in config["data"].items()}
    paths["rendering.output_dir"] = config["rendering"]["output_dir"]
    if config["catalog"]["class_layout_dir"] is not None:
        paths["catalog.class_layout_dir"] = config["catalog"]["class_layout_dir"]
    del paths[output_key]
    return paths


def check_no_overlap(path: str, protected: Dict[str, str]) -> None:
    """Refuse an output that is, contains or lies inside a ``protected`` path.

    Paths are compared after resolving symlinks.
    """
    resolved = Path(path).resolve()
    for key, other_path in protected.items():
        other = Path(other_path).resolve()
        if other.is_relative_to(resolved) or resolved.is_relative_to(other):
            raise ValueError(
                f"Output {path} overlaps {key} ({other_path}); "
                "choose a separate location"
            )


def prepare_output_dir(path: str, force: bool, protected: Dict[str, str]) -> Path:
    """Return an empty output directory, refusing to overwrite one unless forced.

    ``force`` empties the directory, so it must not be, contain or lie inside
    any ``protected`` path (compared after resolving symlinks). A symlinked
    output directory is emptied, not removed.
    """
    out_dir = Path(path)
    check_no_overlap(path, protected)
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {path}")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {path} (set output.force: true "
                "to replace it)"
            )
        for child in out_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
