"""PatchCore anomaly detector (Roth et al., CVPR 2022).

A memory bank of normal patch embeddings is subsampled with greedy
(k-center) coreset selection; an image's score is the largest distance from
any of its patches to the nearest memory-bank patch.
"""

from typing import Optional

import numpy as np
import torch

# Rows of the query/candidate matrix processed per distance computation, which
# bounds peak memory at roughly CHUNK x memory-bank-size floats.
_CHUNK = 4096


def greedy_coreset(
    features: torch.Tensor,
    n_select: int,
    projection_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Greedy k-center coreset indices of ``features`` (N, D).

    Distances are computed on a random Gaussian projection to
    ``projection_dim`` dimensions (Johnson-Lindenstrauss), as in PatchCore.
    """
    n = features.shape[0]
    if n_select >= n:
        return torch.arange(n, device=features.device)

    dim = features.shape[1]
    if projection_dim < dim:
        projection = torch.randn(
            dim, projection_dim, generator=generator, device="cpu"
        ).to(features.device) / (projection_dim**0.5)
        reduced = features @ projection
    else:
        reduced = features

    start = int(torch.randint(n, (1,), generator=generator).item())
    selected = [start]
    min_dist = torch.linalg.norm(reduced - reduced[start], dim=1)
    for _ in range(n_select - 1):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        min_dist = torch.minimum(
            min_dist, torch.linalg.norm(reduced - reduced[idx], dim=1)
        )
    return torch.tensor(selected, device=features.device)


def nearest_distances(
    queries: torch.Tensor, bank: torch.Tensor, chunk: int = _CHUNK
) -> torch.Tensor:
    """Distance from each query row to its nearest memory-bank row."""
    out = []
    for start in range(0, queries.shape[0], chunk):
        dists = torch.cdist(queries[start : start + chunk], bank)
        out.append(dists.min(dim=1).values)
    return torch.cat(out)


class PatchCoreDetector:
    """Patch-level memory-bank detector over (N, P, D) patch embeddings."""

    def __init__(
        self,
        coreset_ratio: float,
        projection_dim: int,
        seed: Optional[int],
        device: str,
    ) -> None:
        self.coreset_ratio = coreset_ratio
        self.projection_dim = projection_dim
        self.seed = seed
        self.device = device
        self.memory_bank: Optional[torch.Tensor] = None

    def fit(self, normal_features: np.ndarray) -> "PatchCoreDetector":
        if normal_features.ndim != 3:
            raise ValueError(
                "PatchCore expects patch features of shape (N, P, D), "
                f"got {normal_features.shape}"
            )
        patches = torch.from_numpy(
            np.asarray(normal_features, dtype=np.float32).reshape(
                -1, normal_features.shape[-1]
            )
        ).to(self.device)
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        n_select = max(1, int(round(self.coreset_ratio * patches.shape[0])))
        with torch.no_grad():
            idx = greedy_coreset(patches, n_select, self.projection_dim, generator)
        self.memory_bank = patches[idx]
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.memory_bank is None:
            raise RuntimeError("PatchCoreDetector.score called before fit")
        n_images, n_patches, dim = features.shape
        queries = torch.from_numpy(
            np.asarray(features, dtype=np.float32).reshape(-1, dim)
        ).to(self.device)
        with torch.no_grad():
            dists = nearest_distances(queries, self.memory_bank)
        return dists.reshape(n_images, n_patches).max(dim=1).values.cpu().numpy()
