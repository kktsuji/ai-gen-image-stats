"""Gaussian (Mahalanobis) anomaly detector with Ledoit-Wolf shrinkage."""

from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf


class MahalanobisDetector:
    """Score = squared Mahalanobis distance to the normal-class Gaussian.

    The covariance is Ledoit-Wolf shrunk, which keeps it well conditioned when
    the feature dimension (2048) exceeds or approaches the number of normals.
    """

    def __init__(self) -> None:
        self._model: Optional[LedoitWolf] = None

    def fit(self, normal_features: np.ndarray) -> "MahalanobisDetector":
        self._model = LedoitWolf().fit(np.asarray(normal_features, dtype=np.float64))
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MahalanobisDetector.score called before fit")
        return self._model.mahalanobis(np.asarray(features, dtype=np.float64))
