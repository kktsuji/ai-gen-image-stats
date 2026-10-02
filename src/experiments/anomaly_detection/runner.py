"""Anomaly Detection - Experiment Runner

Fits a one-class detector on normal images only and scores the held-out binary
test fold (abnormal vs suspicious) of an extended split file. Writes a
classifier-compatible ``reports/evaluation.json`` so the existing cross-split
report and paired statistical tests can consume it unchanged.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.experiments.anomaly_detection.features import build_feature_spec, get_features
from src.experiments.anomaly_detection.methods import build_detector
from src.experiments.anomaly_detection.report import (
    build_score_table,
    plot_subclass_scores,
    summarize_by_subclass,
)
from src.utils.config import resolve_output_path

logger = logging.getLogger(__name__)

REQUIRED_SPLIT_KEYS = ("train", "val", "test", "normal_extra_train", "normal_extra_val")


def load_extended_split(split_file: str) -> Dict[str, Any]:
    """Load an extended split JSON produced by ``splits.py``."""
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    missing = [k for k in (*REQUIRED_SPLIT_KEYS, "normal_extra_test") if k not in split]
    if missing:
        raise KeyError(
            f"Split file {split_file} lacks {missing}; generate it with "
            "python -m src.experiments.anomaly_detection.splits"
        )
    return split


def select_partitions(
    split: Dict[str, Any], normal_pool: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Choose the entries each stage uses.

    Training and threshold calibration use normal (label 0) images only; the
    abnormal training images are never touched. ``normal_pool='suspicious'``
    restricts both to the suspicious subclass. The held-out extra normals are
    always scored for the secondary abnormal-vs-all-normals metric.
    """

    def normals(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [e for e in entries if e["label"] == 0]

    train = normals(split["train"])
    val = normals(split["val"])
    if normal_pool == "all":
        train = train + split["normal_extra_train"]
        val = val + split["normal_extra_val"]
    elif normal_pool != "suspicious":
        raise ValueError(f"Unknown normal_pool: {normal_pool!r}")
    return {
        "train_normals": train,
        "val_normals": val,
        "test": split["test"],
        "extra_test": split["normal_extra_test"],
    }


def compute_binary_metrics(
    targets: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Classifier-compatible metrics with label 1 (abnormal) as positive.

    ``pr_auc`` uses average precision, exactly as ClassifierTrainer does.
    """
    preds = (scores > threshold).astype(int)
    metrics: Dict[str, float] = {
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds))
    }
    prec, rec, f1, _ = precision_recall_fscore_support(
        targets,
        preds,
        labels=[0, 1],
        zero_division=0.0,  # type: ignore[arg-type]
    )
    for cls in (0, 1):
        metrics[f"precision_{cls}"] = float(prec[cls])  # type: ignore[index]
        metrics[f"recall_{cls}"] = float(rec[cls])  # type: ignore[index]
        metrics[f"f1_{cls}"] = float(f1[cls])  # type: ignore[index]
    metrics["f1_macro"] = float(np.mean(f1))
    if len(np.unique(targets)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(targets, scores))
        metrics["pr_auc"] = float(average_precision_score(targets, scores))
    else:
        logger.warning("Skipping AUC metrics: only one class in targets")
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    for i in range(2):
        for j in range(2):
            metrics[f"cm_{i}_{j}"] = float(cm[i, j])
    return metrics


def run_anomaly_detection(config: Dict[str, Any], device: str) -> Path:
    """Fit on normals, score held-out images, write reports.

    Returns:
        Path to the reports directory.
    """
    data_config = config["data"]
    fe_config = config["feature_extraction"]
    method_config = config["method"]
    percentile = config["threshold"]["normal_percentile"]

    split = load_extended_split(data_config["split_file"])
    parts = select_partitions(split, data_config["normal_pool"])
    for name, entries in parts.items():
        logger.info(f"{name}: {len(entries)} images")
    if not parts["train_normals"] or not parts["val_normals"]:
        raise ValueError("No normal images available for training/threshold")

    spec = build_feature_spec(config)
    all_paths = [e["path"] for entries in parts.values() for e in entries]
    t0 = time.time()
    features = get_features(
        all_paths,
        spec,
        device=device,
        batch_size=fe_config["batch_size"],
        num_workers=fe_config["num_workers"],
        cache_dir=fe_config["cache_dir"],
    )
    logger.info(f"Features {features.shape} ready in {time.time() - t0:.1f}s")

    # Slice the row-aligned feature array back into partitions.
    feats: Dict[str, np.ndarray] = {}
    offset = 0
    for name, entries in parts.items():
        feats[name] = features[offset : offset + len(entries)]
        offset += len(entries)

    detector = build_detector(method_config, config["compute"].get("seed"), device)
    t0 = time.time()
    detector.fit(feats["train_normals"])
    logger.info(f"Detector '{method_config['type']}' fit in {time.time() - t0:.1f}s")

    t0 = time.time()
    val_scores = detector.score(feats["val_normals"])
    test_scores = detector.score(feats["test"])
    extra_scores = detector.score(feats["extra_test"])
    logger.info(f"Scoring done in {time.time() - t0:.1f}s")

    # Threshold from normal validation scores only: no CTC label is used.
    threshold = float(np.percentile(val_scores, percentile))
    targets = np.array([e["label"] for e in parts["test"]])
    metrics = compute_binary_metrics(targets, test_scores, threshold)

    # Secondary: abnormal vs every held-out normal subclass (~38:1 regime).
    all_targets = np.concatenate([targets, np.zeros(len(extra_scores), dtype=int)])
    all_scores = np.concatenate([test_scores, extra_scores])
    if len(np.unique(all_targets)) == 2:
        metrics["pr_auc_vs_all_normals"] = float(
            average_precision_score(all_targets, all_scores)
        )
        metrics["roc_auc_vs_all_normals"] = float(
            roc_auc_score(all_targets, all_scores)
        )
    else:
        logger.warning("Skipping vs-all-normals AUC metrics: no abnormal in test")

    classes = split["metadata"]["classes"]
    class_names = [name for name, _ in sorted(classes.items(), key=lambda x: x[1])]
    report_payload = {
        **metrics,
        "split": "test",
        "num_classes": 2,
        "class_names": class_names,
        "positive_class": 1,
        "threshold": threshold,
        "normal_percentile": percentile,
        "method": method_config["type"],
        "feature_model": fe_config["model"],
        "normal_pool": data_config["normal_pool"],
        "n_train_normals": len(parts["train_normals"]),
        "n_val_normals": len(parts["val_normals"]),
        "n_test": len(parts["test"]),
        "n_test_vs_all_normals": len(all_targets),
        "split_file": data_config["split_file"],
    }

    reports_dir = resolve_output_path(config, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    np.savez_compressed(
        reports_dir / "predictions_test.npz",
        targets=targets,
        predictions=(test_scores > threshold).astype(int),
        scores=test_scores,
        paths=np.array([e["path"] for e in parts["test"]]),
        subclasses=np.array([e["subclass"] for e in parts["test"]]),
    )

    table = pd.concat(
        [
            build_score_table(parts["test"], test_scores, "test"),
            build_score_table(parts["extra_test"], extra_scores, "normal_extra_test"),
        ],
        ignore_index=True,
    )
    table.to_csv(reports_dir / "subclass_scores.csv", index=False)
    summary = summarize_by_subclass(table, threshold)
    summary.to_csv(reports_dir / "subclass_summary.csv", index=False)
    plot_subclass_scores(
        table,
        threshold,
        reports_dir / "subclass_scores.png",
        title=(
            f"{method_config['type']} / {fe_config['model']} / "
            f"pool={data_config['normal_pool']}"
        ),
    )

    logger.info(
        f"PR-AUC {metrics.get('pr_auc', float('nan')):.4f}  "
        f"ROC-AUC {metrics.get('roc_auc', float('nan')):.4f}  "
        f"recall_1 {metrics['recall_1']:.4f}  "
        f"PR-AUC vs all normals "
        f"{metrics.get('pr_auc_vs_all_normals', float('nan')):.4f}"
    )
    logger.info("Per-subclass summary:\n" + summary.to_string(index=False))
    logger.info(f"Reports written to: {reports_dir}")
    return reports_dir
