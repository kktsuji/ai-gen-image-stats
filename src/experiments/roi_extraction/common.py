"""Helpers shared by the roi_extraction modes."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

from src.utils.checkpoint import file_sha256
from src.utils.git import get_git_commit_hash

__all__ = [
    "file_sha256",
    "load_catalog",
    "prepare_output_dir",
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


def prepare_output_dir(path: str, force: bool, inputs: Sequence[str]) -> Path:
    """Return an empty output directory, refusing to overwrite one unless forced.

    ``force`` deletes the directory, so it must not be, contain or lie inside
    any of ``inputs`` (paths compared after resolving symlinks).
    """
    out_dir = Path(path)
    resolved = out_dir.resolve()
    for input_path in inputs:
        other = Path(input_path).resolve()
        if other.is_relative_to(resolved) or resolved.is_relative_to(other):
            raise ValueError(
                f"Output directory {path} overlaps the input {input_path}; "
                "choose a separate directory"
            )
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {path} (set output.force: true "
                "to replace it)"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
