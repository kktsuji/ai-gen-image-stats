"""k-NN anomaly detector: mean distance to the k nearest normal samples."""

from typing import Optional

import numpy as np

from src.experiments.sample_selection.selector import compute_knn_scores


class KNNDetector:
    """Score = mean Euclidean distance to the k nearest normal training samples."""

    def __init__(self, k: int) -> None:
        self.k = k
        self._normal: Optional[np.ndarray] = None

    def fit(self, normal_features: np.ndarray) -> "KNNDetector":
        self._normal = np.asarray(normal_features, dtype=np.float32)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._normal is None:
            raise RuntimeError("KNNDetector.score called before fit")
        return compute_knn_scores(
            self._normal, np.asarray(features, dtype=np.float32), k=self.k
        )
