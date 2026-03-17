"""Sample Selection Pipeline

Core pipeline for selecting high-quality generated samples by comparing them
to real training data in feature space. Each run handles one class: real images
of that class vs generated images of that class.

The pipeline:
1. Loads real and generated images
2. Extracts features using a pretrained backbone (InceptionV3/ResNet)
3. Computes per-sample quality scores via k-NN distance to the real manifold
4. Ranks, filters, and copies selected samples to an output directory
5. Outputs a CSV quality report and dataset-level metrics
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from src.utils.config import resolve_output_path
from src.utils.data.datasets import SimpleImageDataset, SplitFileDataset
from src.utils.data.transforms import get_val_transforms

logger = logging.getLogger(__name__)


def run_sample_selection(config: Dict[str, Any], device: str, log_dir: Path) -> Path:
    """Main pipeline: load data -> extract features -> score -> filter -> copy -> report.

    Args:
        config: Full experiment configuration dictionary.
        device: Device string (e.g. "cpu" or "cuda").
        log_dir: Log directory path.

    Returns:
        Path to the output reports directory.
    """
    fe_config = config["feature_extraction"]
    scoring_config = config["scoring"]
    selection_config = config["selection"]
    data_config = config["data"]

    # Build transform for feature extraction
    image_size = fe_config["image_size"]
    transform = get_val_transforms(
        image_size=image_size,
        crop_size=image_size,
        normalize="imagenet",
    )

    # Load datasets
    real_dataset = _load_real_dataset(data_config, transform)
    gen_dataset = SimpleImageDataset(
        root=data_config["generated"]["directory"],
        transform=transform,
    )

    logger.info(f"Real images: {len(real_dataset)}")
    logger.info(f"Generated images: {len(gen_dataset)}")

    # Create data loaders (sequential, no shuffle)
    batch_size = fe_config["batch_size"]
    num_workers = fe_config["num_workers"]

    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Create feature extraction model
    model = _create_feature_model(fe_config["model"], device)

    # Extract features
    logger.info("Extracting features from real images...")
    real_features, real_paths = extract_features_from_loader(
        model, real_loader, device, real_dataset
    )
    logger.info(f"Real features shape: {real_features.shape}")

    logger.info("Extracting features from generated images...")
    gen_features, gen_paths = extract_features_from_loader(
        model, gen_loader, device, gen_dataset
    )
    logger.info(f"Generated features shape: {gen_features.shape}")

    # Compute quality scores
    k = scoring_config["k"]
    logger.info(f"Computing k-NN scores (k={k})...")
    knn_scores = compute_knn_scores(real_features, gen_features, k=k)

    # Compute realism flags
    require_realism = scoring_config["require_realism"]
    realism_flags: Optional[np.ndarray] = None
    if require_realism:
        logger.info("Computing realism flags...")
        realism_flags = compute_realism_flags(real_features, gen_features, k=k)
        in_manifold = int(np.sum(realism_flags))
        logger.info(
            f"Realism: {in_manifold}/{len(realism_flags)} samples within real manifold"
        )

    # Select samples
    selected_mask = select_samples(
        scores=knn_scores,
        mode=selection_config["mode"],
        value=selection_config["value"],
        realism_flags=realism_flags,
        require_realism=require_realism,
    )
    num_selected = int(np.sum(selected_mask))
    logger.info(
        f"Selected {num_selected}/{len(gen_paths)} samples "
        f"(mode={selection_config['mode']}, value={selection_config['value']})"
    )

    # Write reports
    reports_dir = resolve_output_path(config, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    from src.experiments.sample_selection.report import (
        write_accepted_samples_json,
        write_quality_report,
        write_summary,
    )

    # Quality CSV report
    write_quality_report(
        output_path=reports_dir / "quality_report.csv",
        image_paths=gen_paths,
        knn_scores=knn_scores,
        realism_flags=realism_flags,
        selected_mask=selected_mask,
    )
    logger.info(f"Quality report saved to: {reports_dir / 'quality_report.csv'}")

    # Accepted samples JSON (compatible with SplitFileDataset)
    selected_paths = [p for p, sel in zip(gen_paths, selected_mask) if sel]
    write_accepted_samples_json(
        output_path=reports_dir / "accepted_samples.json",
        selected_paths=selected_paths,
        label=data_config["label"],
        class_name=data_config["class_name"],
        metadata={
            "selection_mode": selection_config["mode"],
            "selection_value": selection_config["value"],
            "total_candidates": len(gen_paths),
            "total_selected": num_selected,
            "scoring": {
                "k": k,
                "require_realism": require_realism,
            },
        },
    )
    logger.info(
        f"Accepted samples JSON saved to: {reports_dir / 'accepted_samples.json'}"
    )

    # Compute dataset-level metrics if enabled
    dataset_metrics: Optional[Dict[str, float]] = None
    if config["dataset_metrics"]["enabled"]:
        logger.info("Computing dataset-level metrics (FID, precision, recall)...")
        from src.utils.metrics import evaluate_generative_model

        dataset_metrics = evaluate_generative_model(real_features, gen_features, k=k)
        logger.info(f"FID: {dataset_metrics['fid']:.4f}")
        logger.info(f"Precision: {dataset_metrics['precision']:.4f}")
        logger.info(f"Recall: {dataset_metrics['recall']:.4f}")

    # Summary JSON
    write_summary(
        output_path=reports_dir / "summary.json",
        dataset_metrics=dataset_metrics,
        selection_stats={
            "total_candidates": len(gen_paths),
            "total_selected": num_selected,
            "score_mean": float(np.mean(knn_scores)),
            "score_std": float(np.std(knn_scores)),
            "score_min": float(np.min(knn_scores)),
            "score_max": float(np.max(knn_scores)),
            "score_median": float(np.median(knn_scores)),
        },
        config=config,
    )
    logger.info(f"Summary saved to: {reports_dir / 'summary.json'}")

    # Copy selected samples to output directory
    selected_dir = resolve_output_path(config, "selected")
    selected_dir.mkdir(parents=True, exist_ok=True)
    _copy_selected_samples(selected_paths, selected_dir)
    logger.info(f"Copied {num_selected} selected samples to: {selected_dir}")

    return reports_dir


def _load_real_dataset(
    data_config: Dict[str, Any],
    transform: Any,
) -> Any:
    """Load real image dataset based on config source type.

    Args:
        data_config: Data section of configuration.
        transform: Image transform to apply.

    Returns:
        Dataset instance.
    """
    real_config = data_config["real"]
    source = real_config["source"]

    if source == "split_file":
        dataset = SplitFileDataset(
            split_file=real_config["split_file"],
            split=real_config["split"],
            transform=transform,
            return_labels=False,
        )
        # Filter to single class if class_label is specified
        class_label = real_config["class_label"]
        if class_label is not None:
            dataset = _filter_split_dataset_by_class(dataset, class_label)
        return dataset
    else:
        return SimpleImageDataset(
            root=real_config["directory"],
            transform=transform,
        )


def _filter_split_dataset_by_class(
    dataset: SplitFileDataset, class_label: int
) -> "_FilteredDataset":
    """Filter a SplitFileDataset to only include samples of a specific class.

    Creates a new dataset-like object containing only the samples matching
    the given class label. Uses a FilteredDataset wrapper.

    Args:
        dataset: The full SplitFileDataset.
        class_label: The integer class label to filter by.

    Returns:
        A filtered dataset containing only the matching samples.
    """
    matching_indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label == class_label
    ]
    if len(matching_indices) == 0:
        raise ValueError(
            f"No samples found for class_label={class_label} "
            f"in split file '{dataset.split_file}'"
        )
    return _FilteredDataset(dataset, matching_indices)


class _FilteredDataset:
    """Wraps a SplitFileDataset to expose only a subset of indices.

    This is a lightweight wrapper that presents a filtered view of the
    underlying dataset without copying data. It provides the same interface
    needed by extract_features_from_loader (len, getitem, image_paths-like access).
    """

    def __init__(self, dataset: SplitFileDataset, indices: List[int]) -> None:
        self._dataset = dataset
        self._indices = indices
        # Pre-compute the file paths for the filtered samples
        all_samples = dataset.samples
        self._filtered_paths = [all_samples[i][0] for i in indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[self._indices[index]]

    def get_file_paths(self) -> List[str]:
        """Get file paths for all samples in the filtered dataset."""
        return list(self._filtered_paths)


def _create_feature_model(model_name: str, device: str) -> torch.nn.Module:
    """Create a pretrained feature extraction model.

    Args:
        model_name: Model name (inceptionv3, resnet50, etc.).
        device: Device string.

    Returns:
        Model in eval mode on the specified device.
    """
    if model_name == "inceptionv3":
        from src.experiments.classifier.models import InceptionV3Classifier

        model = InceptionV3Classifier(
            num_classes=1,
            pretrained=True,
            freeze_backbone=True,
        )
    elif model_name in ("resnet50", "resnet101", "resnet152"):
        from src.experiments.classifier.models import ResNetClassifier

        model = ResNetClassifier(
            variant=model_name,
            num_classes=1,
            pretrained=True,
            freeze_backbone=True,
        )
    else:
        raise ValueError(f"Unknown feature extraction model: {model_name}")

    model = model.to(device)
    model.eval()
    return model


def extract_features_from_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    dataset: Any,
) -> Tuple[np.ndarray, List[str]]:
    """Batch feature extraction with torch.no_grad().

    Iterates through the loader in order (shuffle=False) to maintain
    correspondence between features and file paths.

    Args:
        model: Feature extraction model with extract_features() method.
        loader: DataLoader (must have shuffle=False).
        device: Device string.
        dataset: The dataset object, used to retrieve file paths.

    Returns:
        Tuple of (features array of shape (N, D), list of file path strings).
    """
    all_features = []

    with torch.no_grad():
        for batch in loader:
            # Handle both (image,) and (image, label) returns
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            features = model.extract_features(images)  # type: ignore[operator]
            all_features.append(features.cpu().numpy())

    features_array = np.concatenate(all_features, axis=0)

    # Get file paths from the dataset
    file_paths = _get_dataset_paths(dataset)

    return features_array, file_paths


def _get_dataset_paths(dataset: Any) -> List[str]:
    """Extract file paths from a dataset object.

    Supports SimpleImageDataset (image_paths), SplitFileDataset (samples),
    and _FilteredDataset (get_file_paths).

    Args:
        dataset: Dataset instance.

    Returns:
        List of file path strings.
    """
    if hasattr(dataset, "get_file_paths"):
        return dataset.get_file_paths()
    elif hasattr(dataset, "image_paths"):
        return [str(p) for p in dataset.image_paths]
    elif hasattr(dataset, "samples"):
        return [path for path, _ in dataset.samples]
    else:
        raise TypeError(
            f"Cannot extract file paths from dataset of type {type(dataset).__name__}"
        )


def compute_knn_scores(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Per-sample mean k-NN distance to the real manifold.

    For each generated sample, finds the k nearest real samples in feature
    space and returns the mean distance. Lower scores = more realistic.

    Args:
        real_features: Feature vectors from real images, shape (N, D).
        gen_features: Feature vectors from generated images, shape (M, D).
        k: Number of nearest neighbors.

    Returns:
        Array of scores, shape (M,). Lower is better.
    """
    # Clamp k to the number of real samples
    effective_k = min(k, real_features.shape[0])

    nbrs = NearestNeighbors(n_neighbors=effective_k).fit(real_features)
    distances, _ = nbrs.kneighbors(gen_features)
    # Mean distance to k nearest real neighbors
    scores = distances.mean(axis=1)
    return scores


def compute_realism_flags(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Boolean array: is each generated sample within the real manifold?

    Uses the precision criterion from Kynkäänniemi et al. (2019): a generated
    sample is "realistic" if its distance to the nearest real sample is less
    than or equal to that real sample's k-th nearest neighbor distance within
    the real distribution.

    Args:
        real_features: Feature vectors from real images, shape (N, D).
        gen_features: Feature vectors from generated images, shape (M, D).
        k: Number of nearest neighbors for manifold estimation.

    Returns:
        Boolean array of shape (M,). True = within real manifold.
    """
    effective_k = min(k, real_features.shape[0] - 1)
    if effective_k < 1:
        # Not enough real samples to compute manifold
        return np.ones(gen_features.shape[0], dtype=bool)

    # Compute k-th nearest neighbor distances within real distribution
    real_nbrs = NearestNeighbors(n_neighbors=effective_k + 1).fit(real_features)
    real_distances, _ = real_nbrs.kneighbors(real_features)
    # Index effective_k because index 0 is the point itself
    real_kth_distances = real_distances[:, effective_k]

    # For each generated sample, find nearest real sample
    gen_to_real_nbrs = NearestNeighbors(n_neighbors=1).fit(real_features)
    gen_to_real_distances, gen_to_real_indices = gen_to_real_nbrs.kneighbors(
        gen_features
    )
    gen_to_real_distances = gen_to_real_distances.flatten()
    gen_to_real_indices = gen_to_real_indices.flatten()

    # A generated sample is within the real manifold if its distance to the
    # nearest real sample <= that real sample's k-th neighbor distance
    flags = gen_to_real_distances <= real_kth_distances[gen_to_real_indices]
    return flags.astype(bool)


def select_samples(
    scores: np.ndarray,
    mode: str,
    value: Any,
    realism_flags: Optional[np.ndarray] = None,
    require_realism: bool = False,
) -> np.ndarray:
    """Select samples based on quality scores and optional realism filter.

    Args:
        scores: Per-sample quality scores (lower = better). Shape (M,).
        mode: Selection mode ('top_k', 'percentile', 'threshold').
        value: Selection parameter (N for top_k, percentage for percentile,
               distance threshold for threshold).
        realism_flags: Optional boolean array (M,) from compute_realism_flags.
        require_realism: If True, only consider samples that pass realism check.

    Returns:
        Boolean mask of selected samples, shape (M,).
    """
    n = len(scores)
    selected = np.zeros(n, dtype=bool)

    # Apply realism filter first if required
    if require_realism and realism_flags is not None:
        candidates = realism_flags.copy()
    else:
        candidates = np.ones(n, dtype=bool)

    candidate_indices = np.where(candidates)[0]
    if len(candidate_indices) == 0:
        return selected

    candidate_scores = scores[candidate_indices]

    if mode == "top_k":
        k = min(int(value), len(candidate_indices))
        # Select the k candidates with lowest scores
        top_indices = np.argsort(candidate_scores)[:k]
        selected[candidate_indices[top_indices]] = True

    elif mode == "percentile":
        # Keep top X% by quality (lowest scores)
        cutoff_idx = max(1, int(np.ceil(len(candidate_indices) * value / 100.0)))
        top_indices = np.argsort(candidate_scores)[:cutoff_idx]
        selected[candidate_indices[top_indices]] = True

    elif mode == "threshold":
        # Keep samples below distance threshold
        below_threshold = candidate_scores <= value
        selected[candidate_indices[below_threshold]] = True

    return selected


def _copy_selected_samples(selected_paths: List[str], output_dir: Path) -> None:
    """Copy selected sample files to the output directory.

    Args:
        selected_paths: List of source file paths.
        output_dir: Destination directory.
    """
    for path_str in selected_paths:
        src = Path(path_str)
        dst = output_dir / src.name
        shutil.copy2(str(src), str(dst))
