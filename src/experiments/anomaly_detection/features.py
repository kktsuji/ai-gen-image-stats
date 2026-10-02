"""Anomaly Detection - Frozen Feature Extraction with Caching

Extracts features from a frozen ImageNet backbone, either as one pooled vector
per image (k-NN / Mahalanobis) or as a grid of locally-aware patch vectors
(PatchCore). Frozen features do not depend on the split, so every image is
extracted once and cached on disk; each split then reads its subset.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.experiments.sample_selection.selector import (
    create_feature_model,
    extract_features_from_loader,
)
from src.utils.data.transforms import get_val_transforms

logger = logging.getLogger(__name__)


class PathListDataset(Dataset):
    """Unlabeled dataset over an explicit list of image paths."""

    def __init__(self, paths: Sequence[str], transform: Any) -> None:
        self.image_paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def build_feature_spec(config: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the feature representation a config needs (also the cache key)."""
    fe = config["feature_extraction"]
    spec: Dict[str, Any] = {
        "model": fe["model"],
        "image_size": fe["image_size"],
        "crop_size": fe["crop_size"],
        "normalize": "imagenet",
    }
    method = config["method"]
    if method["type"] == "patchcore":
        params = method["patchcore"]
        spec.update(
            kind="patch",
            layers=list(params["layers"]),
            grid_size=params["grid_size"],
            feature_dim=params["feature_dim"],
        )
    else:
        spec["kind"] = "pooled"
    return spec


def spec_cache_name(spec: Dict[str, Any]) -> str:
    """Stable, human-readable cache file name for a feature spec."""
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{spec['model']}_{spec['kind']}_{digest}.npz"


def _resize_map(fmap: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Resize a (B, C, H, W) feature map to (B, C, grid, grid)."""
    if fmap.shape[-1] >= grid_size:
        return F.adaptive_avg_pool2d(fmap, grid_size)
    return F.interpolate(
        fmap, size=(grid_size, grid_size), mode="bilinear", align_corners=False
    )


def patch_embeddings(
    feature_maps: List[torch.Tensor], grid_size: int, feature_dim: int
) -> torch.Tensor:
    """Combine intermediate feature maps into PatchCore patch embeddings.

    Each map gets 3x3 local average aggregation (neighbourhood context), is
    resized to a common grid, concatenated along channels, and the channel
    axis is reduced to ``feature_dim`` by adaptive average pooling.

    Returns:
        Tensor of shape (B, grid_size * grid_size, feature_dim).
    """
    resized = [
        _resize_map(F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1), grid_size)
        for fmap in feature_maps
    ]
    stacked = torch.cat(resized, dim=1)  # (B, C, g, g)
    batch, channels = stacked.shape[:2]
    patches = stacked.permute(0, 2, 3, 1).reshape(-1, 1, channels)
    reduced = F.adaptive_avg_pool1d(patches, feature_dim)
    return reduced.reshape(batch, grid_size * grid_size, feature_dim)


def extract_patch_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    layers: List[str],
    grid_size: int,
    feature_dim: int,
) -> np.ndarray:
    """Extract PatchCore patch embeddings via forward hooks on ``layers``.

    Returns:
        Array of shape (N, grid_size * grid_size, feature_dim), float16.
    """
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    for name in layers:
        module = getattr(model, name)

        def hook(_module, _inputs, output, name=name):
            captured[name] = output

        handles.append(module.register_forward_hook(hook))

    outputs = []
    try:
        with torch.no_grad():
            for images in loader:
                captured.clear()
                model.extract_features(images.to(device))  # type: ignore[operator]
                embeds = patch_embeddings(
                    [captured[name] for name in layers], grid_size, feature_dim
                )
                outputs.append(embeds.cpu().numpy().astype(np.float16))
    finally:
        for handle in handles:
            handle.remove()

    if not outputs:
        raise ValueError("No features extracted: the data loader yielded no batches")
    return np.concatenate(outputs, axis=0)


def _extract(
    paths: List[str],
    spec: Dict[str, Any],
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    transform = get_val_transforms(
        image_size=spec["image_size"],
        crop_size=spec["crop_size"],
        normalize=spec["normalize"],
    )
    dataset = PathListDataset(paths, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model = create_feature_model(spec["model"], device)
    if spec["kind"] == "patch":
        return extract_patch_features(
            model,
            loader,
            device,
            spec["layers"],
            spec["grid_size"],
            spec["feature_dim"],
        )
    features, _ = extract_features_from_loader(model, loader, device, dataset)
    return features.astype(np.float32)


def get_features(
    paths: Sequence[str],
    spec: Dict[str, Any],
    device: str,
    batch_size: int,
    num_workers: int,
    cache_dir: Optional[str],
) -> np.ndarray:
    """Return features for ``paths`` (row-aligned), extracting only uncached ones.

    Args:
        paths: Image paths; duplicates are allowed.
        spec: Feature spec from :func:`build_feature_spec`.
        cache_dir: Directory for the on-disk cache, or None to disable caching.

    Returns:
        Array whose first axis matches ``paths``.
    """
    unique = list(dict.fromkeys(paths))
    cached_paths: List[str] = []
    cached_feats: Optional[np.ndarray] = None
    cache_file = Path(cache_dir) / spec_cache_name(spec) if cache_dir else None

    if cache_file is not None and cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            stored_spec = json.loads(str(data["spec"]))
            if stored_spec != spec:
                raise ValueError(
                    f"Feature cache {cache_file} was built for a different spec"
                )
            cached_paths = [str(p) for p in data["paths"]]
            cached_feats = data["features"]
        logger.info(f"Loaded {len(cached_paths)} cached features from {cache_file}")

    index = {path: i for i, path in enumerate(cached_paths)}
    missing = [p for p in unique if p not in index]

    if missing:
        logger.info(f"Extracting features for {len(missing)} images ({spec})")
        new_feats = _extract(missing, spec, device, batch_size, num_workers)
        if cached_feats is None:
            cached_feats = new_feats
        else:
            cached_feats = np.concatenate([cached_feats, new_feats], axis=0)
        for path in missing:
            index[path] = len(cached_paths)
            cached_paths.append(path)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_file,
                spec=np.array(json.dumps(spec, sort_keys=True)),
                paths=np.array(cached_paths),
                features=cached_feats,
            )
            logger.info(f"Feature cache updated: {cache_file}")

    assert cached_feats is not None
    return cached_feats[[index[p] for p in paths]]
