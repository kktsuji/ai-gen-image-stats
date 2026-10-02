"""One-class anomaly detectors.

Every detector is fit on normal samples only and returns one score per sample,
where a higher score means more anomalous.
"""

from typing import Any, Dict, Optional, Protocol

import numpy as np

from src.experiments.anomaly_detection.methods.knn import KNNDetector
from src.experiments.anomaly_detection.methods.mahalanobis import (
    MahalanobisDetector,
)
from src.experiments.anomaly_detection.methods.patchcore import PatchCoreDetector


class AnomalyDetector(Protocol):
    def fit(self, normal_features: np.ndarray) -> "AnomalyDetector": ...

    def score(self, features: np.ndarray) -> np.ndarray: ...


def build_detector(
    method_config: Dict[str, Any], seed: Optional[int], device: str
) -> AnomalyDetector:
    """Instantiate the detector selected by ``method.type``."""
    method_type = method_config["type"]
    params = method_config[method_type]
    if method_type == "knn":
        return KNNDetector(k=params["k"])
    if method_type == "mahalanobis":
        return MahalanobisDetector()
    if method_type == "patchcore":
        return PatchCoreDetector(
            coreset_ratio=params["coreset_ratio"],
            projection_dim=params["projection_dim"],
            seed=seed,
            device=device,
        )
    raise ValueError(f"Unknown anomaly detection method: {method_type!r}")


__all__ = [
    "AnomalyDetector",
    "KNNDetector",
    "MahalanobisDetector",
    "PatchCoreDetector",
    "build_detector",
]
