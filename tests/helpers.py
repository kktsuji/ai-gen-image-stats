"""Shared test helper functions.

Utility functions used across test modules. Unlike conftest.py (which is
reserved for pytest fixtures and hooks), this module holds plain helper
functions that tests import explicitly.
"""

import json
from pathlib import Path


def create_split_json(data_dir, split_json_path, include_val=True):
    """Create a split JSON file from a directory structure.

    Scans class directories (e.g. '0.Normal', '1.Abnormal') for images
    and writes a JSON split file compatible with SplitFileDataset.

    Args:
        data_dir: Root directory containing class subdirectories.
        split_json_path: Path where the JSON file will be written.
        include_val: If True, val split mirrors train; otherwise empty.

    Returns:
        str: Path to the created split JSON file.
    """
    entries = []
    data_path = Path(data_dir)
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            label = int(class_dir.name.split(".")[0])
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix in (".png", ".jpg", ".jpeg"):
                    entries.append({"path": str(img_file), "label": label})

    class_names = sorted(set(str(e["label"]) for e in entries))
    classes = {name: int(name) for name in class_names}

    split_data = {
        "metadata": {"classes": classes},
        "train": entries,
        "val": entries if include_val else [],
    }

    split_json_path = Path(split_json_path)
    split_json_path.write_text(json.dumps(split_data))
    return str(split_json_path)
