"""Anomaly Detection - Frozen Feature Extraction with Caching

Extracts features from a frozen backbone, either as one pooled vector per image
(k-NN / Mahalanobis) or as a grid of locally-aware patch vectors (PatchCore).
The backbone has ImageNet weights, or the weights of a fine-tuned classifier
checkpoint (``feature_extraction.checkpoint``). ImageNet features do not depend
on the split, so every image is extracted once and cached on disk; each split
then reads its subset. A checkpoint is part of the cache key (by content hash),
so per-split checkpoints get separate caches.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.experiments.sample_selection.selector import (
    create_feature_model,
    extract_features_from_loader,
)
from src.utils.checkpoint import file_sha256, load_model_weights
from src.utils.data.transforms import get_val_transforms

logger = logging.getLogger(__name__)

# Classification head of ResNetClassifier / InceptionV3Classifier. Its shape
# depends on num_classes and it is not used by extract_features(), so it is not
# loaded from a checkpoint.
HEAD_PREFIX = "fc."


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


def load_backbone_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a classifier checkpoint's backbone weights into ``model``.

    The checkpoint must come from the same architecture (``save_checkpoint``
    format, ``model_state_dict``). The classification head (``fc.*``) is skipped;
    every other parameter and buffer must be present, so a checkpoint from a
    different backbone fails instead of silently keeping ImageNet weights.
    """
    load_model_weights(model, checkpoint_path, skip_prefixes=(HEAD_PREFIX,))
    model.eval()
    logger.info(f"Loaded backbone weights from {checkpoint_path}")


def build_feature_spec(config: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the feature representation a config needs (also the cache key).

    With ``feature_extraction.checkpoint`` set, the spec records the checkpoint's
    content hash (not its path, so a moved series keeps its caches).
    """
    fe = config["feature_extraction"]
    spec: Dict[str, Any] = {
        "model": fe["model"],
        "image_size": fe["image_size"],
        "crop_size": fe["crop_size"],
        "normalize": "imagenet",
    }
    if fe["checkpoint"] is not None:
        spec["checkpoint_sha256"] = file_sha256(fe["checkpoint"])
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
    model = spec["model"] + ("-ft" if "checkpoint_sha256" in spec else "")
    return f"{model}_{spec['kind']}_{digest}.npz"


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
    checkpoint: Optional[str] = None,
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
    if checkpoint is not None:
        if file_sha256(checkpoint) != spec.get("checkpoint_sha256"):
            raise ValueError(
                f"Checkpoint {checkpoint} changed after the spec was built"
            )
        load_backbone_weights(model, checkpoint)
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


def _load_cache(cache_file: Path, spec: Dict[str, Any]) -> Tuple[List[str], Any]:
    """Read a feature cache; returns ([], None) when the file does not exist."""
    if not cache_file.exists():
        return [], None
    with np.load(cache_file, allow_pickle=False) as data:
        stored_spec = json.loads(str(data["spec"]))
        if stored_spec != spec:
            raise ValueError(
                f"Feature cache {cache_file} was built for a different spec"
            )
        paths = [str(p) for p in data["paths"]]
        features = data["features"]
    logger.info(f"Loaded {len(paths)} cached features from {cache_file}")
    return paths, features


def _save_cache_atomic(
    cache_file: Path,
    spec: Dict[str, Any],
    paths: List[str],
    features: np.ndarray,
) -> None:
    """Write the cache to a temp file in the same directory, then rename it.

    The rename is atomic, so a reader never sees a partially written file.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=cache_file.parent, prefix=cache_file.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                spec=np.array(json.dumps(spec, sort_keys=True)),
                paths=np.array(paths),
                features=features,
            )
        os.replace(tmp_name, cache_file)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    logger.info(f"Feature cache updated: {cache_file}")


def get_features(
    paths: Sequence[str],
    spec: Dict[str, Any],
    device: str,
    batch_size: int,
    num_workers: int,
    cache_dir: Optional[str],
    checkpoint: Optional[str] = None,
) -> np.ndarray:
    """Return features for ``paths`` (row-aligned), extracting only uncached ones.

    The cache is shared by every run with the same spec (all methods, all
    splits). A file lock is held across read -> extract -> write so concurrent
    runs neither corrupt the file nor lose each other's rows, and runs needing
    the same images wait instead of extracting them twice.

    Args:
        paths: Image paths; duplicates are allowed.
        spec: Feature spec from :func:`build_feature_spec`.
        cache_dir: Directory for the on-disk cache, or None to disable caching.
        checkpoint: Classifier checkpoint whose backbone weights replace the
            ImageNet ones; its hash must be in ``spec`` (``build_feature_spec``).

    Returns:
        Array whose first axis matches ``paths``.
    """
    if not cache_dir:
        unique = list(dict.fromkeys(paths))
        logger.info(f"Extracting features for {len(unique)} images ({spec})")
        feats = _extract(unique, spec, device, batch_size, num_workers, checkpoint)
        index = {path: i for i, path in enumerate(unique)}
        return feats[[index[p] for p in paths]]

    cache_file = Path(cache_dir) / spec_cache_name(spec)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache_file) + ".lock"):
        cached_paths, cached_feats = _load_cache(cache_file, spec)
        index = {path: i for i, path in enumerate(cached_paths)}
        missing = [p for p in dict.fromkeys(paths) if p not in index]

        if missing:
            logger.info(f"Extracting features for {len(missing)} images ({spec})")
            new_feats = _extract(
                missing, spec, device, batch_size, num_workers, checkpoint
            )
            if cached_feats is None:
                cached_feats = new_feats
            else:
                cached_feats = np.concatenate([cached_feats, new_feats], axis=0)
            for path in missing:
                index[path] = len(cached_paths)
                cached_paths.append(path)
            _save_cache_atomic(cache_file, spec, cached_paths, cached_feats)

    return cached_feats[[index[p] for p in paths]]
