"""Sample Selection Evaluator

Dedicated evaluate mode for the sample selection experiment. Compares
distribution-level quality metrics across real, generated, and optionally
selected image sets to assess quality at each stage of the pipeline.

Metrics per pair: FID, Precision, Recall, ROC-AUC, PR-AUC
(via evaluate_generative_model from src/utils/metrics).
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from torch.utils.data import DataLoader

from src.experiments.sample_selection.report import write_evaluation_report
from src.experiments.sample_selection.selector import (
    create_feature_model,
    extract_features_from_loader,
    load_real_dataset,
)
from src.utils.config import resolve_output_path
from src.utils.data.datasets import SimpleImageDataset, SplitFileDataset
from src.utils.data.transforms import get_val_transforms
from src.utils.metrics import (
    calculate_fid,
    calculate_precision_recall,
    evaluate_generative_model,
)

logger = logging.getLogger(__name__)

# Minimum samples per set required for ROC-AUC/PR-AUC.  train_test_split
# with test_size=0.3 and stratify needs at least ceil(2/0.3)=7 samples
# per class to guarantee 2 samples in the test split.
_MIN_SAMPLES_FOR_AUC = 7


def run_sample_selection_evaluate(
    config: Dict[str, Any], device: str, log_dir: Path
) -> Path:
    """Evaluate distribution-level quality of generated and selected images.

    Compares real vs generated (always), and if data.selected is present,
    also compares real vs selected and generated vs selected.

    Args:
        config: Full experiment configuration dictionary.
        device: Device string (e.g. "cpu" or "cuda").
        log_dir: Log directory path.

    Returns:
        Path to the output reports directory.
    """
    fe_config = config["feature_extraction"]
    data_config = config["data"]
    eval_config = config["evaluation"]
    k = eval_config["k"]

    # Build transform for feature extraction
    image_size = fe_config["image_size"]
    transform = get_val_transforms(
        image_size=image_size,
        crop_size=image_size,
        normalize="imagenet",
    )

    batch_size = fe_config["batch_size"]
    num_workers = fe_config["num_workers"]

    # Load datasets
    real_dataset = load_real_dataset(data_config, transform)
    gen_dataset = SimpleImageDataset(
        root=data_config["generated"]["directory"],
        transform=transform,
    )

    if len(real_dataset) == 0:
        raise ValueError("Real dataset is empty — cannot compute metrics")
    if len(gen_dataset) == 0:
        raise ValueError("Generated dataset is empty — cannot compute metrics")

    logger.info(f"Real images: {len(real_dataset)}")
    logger.info(f"Generated images: {len(gen_dataset)}")

    dataset_sizes: Dict[str, int] = {
        "real": len(real_dataset),
        "generated": len(gen_dataset),
    }

    # Optionally load selected dataset
    selected_dataset = None
    if "selected" in data_config:
        sel_config = data_config["selected"]
        selected_dataset = SplitFileDataset(
            split_file=sel_config["split_file"],
            split=sel_config["split"],
            transform=transform,
            return_labels=False,
        )
        logger.info(f"Selected images: {len(selected_dataset)}")
        dataset_sizes["selected"] = len(selected_dataset)

    # Create feature extraction model
    model = create_feature_model(fe_config["model"], device)

    # Extract features
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

    logger.info("Extracting features from real images...")
    real_features, _ = extract_features_from_loader(
        model, real_loader, device, real_dataset
    )
    logger.info(f"Real features shape: {real_features.shape}")

    logger.info("Extracting features from generated images...")
    gen_features, _ = extract_features_from_loader(
        model, gen_loader, device, gen_dataset
    )
    logger.info(f"Generated features shape: {gen_features.shape}")

    sel_features: np.ndarray | None = None
    if selected_dataset is not None:
        sel_loader = DataLoader(
            selected_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        logger.info("Extracting features from selected images...")
        sel_features, _ = extract_features_from_loader(
            model, sel_loader, device, selected_dataset
        )
        logger.info(f"Selected features shape: {sel_features.shape}")

    # Compute metrics per pair
    comparisons: Dict[str, Dict[str, float]] = {}

    comparisons["real_vs_generated"] = _compute_pair_metrics(
        real_features, gen_features, k, "real_vs_generated"
    )

    if sel_features is not None:
        comparisons["real_vs_selected"] = _compute_pair_metrics(
            real_features, sel_features, k, "real_vs_selected"
        )
        comparisons["generated_vs_selected"] = _compute_pair_metrics(
            gen_features, sel_features, k, "generated_vs_selected"
        )

    # Write report
    reports_dir = resolve_output_path(config, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "evaluation.json"
    write_evaluation_report(
        output_path=report_path,
        comparisons=comparisons,
        dataset_sizes=dataset_sizes,
        config=config,
    )
    logger.info(f"Evaluation report saved to: {report_path}")

    # Log results
    for pair_name, metrics in comparisons.items():
        logger.info(f"--- {pair_name} ---")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

    return reports_dir


def _compute_pair_metrics(
    features_a: np.ndarray,
    features_b: np.ndarray,
    k: int,
    pair_name: str,
) -> Dict[str, float]:
    """Compute distribution metrics for a pair of feature sets.

    If either set has too few samples for ROC-AUC/PR-AUC, those metrics
    are skipped with a warning.

    Args:
        features_a: Feature vectors from the first set.
        features_b: Feature vectors from the second set.
        k: Number of nearest neighbors for precision/recall.
        pair_name: Name of the pair (for logging).

    Returns:
        Dictionary of metric name to value.
    """
    min_set_size = min(len(features_a), len(features_b))
    if min_set_size < _MIN_SAMPLES_FOR_AUC:
        logger.warning(
            f"Skipping ROC-AUC/PR-AUC for {pair_name}: "
            f"too few samples in smaller set ({min_set_size} < {_MIN_SAMPLES_FOR_AUC})"
        )
        fid = calculate_fid(features_a, features_b)
        precision, recall = calculate_precision_recall(features_a, features_b, k=k)
        return {
            "fid": fid,
            "precision": precision,
            "recall": recall,
        }

    logger.info(f"Computing metrics for {pair_name}...")
    return evaluate_generative_model(features_a, features_b, k=k)
