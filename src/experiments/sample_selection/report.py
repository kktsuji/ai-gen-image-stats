"""Sample Selection Report Generation

Produces CSV quality reports, accepted samples JSON (compatible with
SplitFileDataset), and summary JSON with dataset-level metrics.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def write_quality_report(
    output_path: Path,
    image_paths: List[str],
    knn_scores: np.ndarray,
    realism_flags: Optional[np.ndarray],
    selected_mask: np.ndarray,
) -> None:
    """Write per-sample quality report as CSV.

    Columns: path, knn_score, realism_flag, selected

    Args:
        output_path: Path to write the CSV file.
        image_paths: List of image file paths.
        knn_scores: Per-sample k-NN distance scores.
        realism_flags: Optional boolean realism flags (None if not computed).
        selected_mask: Boolean mask of selected samples.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "knn_score", "realism_flag", "selected"])

        for i, path in enumerate(image_paths):
            realism = bool(realism_flags[i]) if realism_flags is not None else ""
            writer.writerow(
                [path, f"{knn_scores[i]:.6f}", realism, bool(selected_mask[i])]
            )


def write_accepted_samples_json(
    output_path: Path,
    selected_paths: List[str],
    label: int,
    class_name: str,
    metadata: Dict[str, Any],
) -> None:
    """Write accepted samples in SplitFileDataset-compatible JSON format.

    Output format:
    {
      "metadata": {
        "created_at": "...",
        "source_experiment": "sample_selection",
        "selection_mode": "percentile",
        "total_candidates": 500,
        "total_selected": 400,
        "classes": {"normal": 0},
        "scoring": {"k": 5, "require_realism": false}
      },
      "train": [
        {"path": "outputs/diffusion/generated/sample_0001.png", "label": 0},
        ...
      ]
    }

    Uses "train" as the split key so SplitFileDataset can load it directly
    with split="train".

    Args:
        output_path: Path to write the JSON file.
        selected_paths: List of file paths for accepted samples.
        label: Integer class label for all samples.
        class_name: String class name.
        metadata: Additional metadata dict (selection_mode, scoring params, etc.).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_experiment": "sample_selection",
            "classes": {class_name: label},
            **metadata,
        },
        "train": [{"path": path, "label": label} for path in selected_paths],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def write_summary(
    output_path: Path,
    dataset_metrics: Optional[Dict[str, float]],
    selection_stats: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Write experiment summary as JSON.

    Includes dataset-level metrics (FID, precision, recall), selection
    statistics, and configuration snapshot.

    Args:
        output_path: Path to write the JSON file.
        dataset_metrics: Optional dict with FID/precision/recall metrics.
        selection_stats: Dict with selection statistics (counts, score stats).
        config: Full experiment configuration (for reference).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": selection_stats,
        "config": {
            "feature_extraction": config.get("feature_extraction"),
            "scoring": config.get("scoring"),
            "selection": config.get("selection"),
            "dataset_metrics": config.get("dataset_metrics"),
        },
    }

    if dataset_metrics is not None:
        summary["dataset_metrics"] = dataset_metrics

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
