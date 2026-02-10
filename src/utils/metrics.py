"""
Common metrics for evaluating generative models and classification tasks.

This module provides implementations of standard metrics including:
- FID (Fréchet Inception Distance)
- Inception Score (IS)
- Precision and Recall for generative models
- ROC-AUC and PR-AUC for domain classification
- Wasserstein distances for distribution comparison
"""

from typing import Tuple

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and fake features.

    FID measures the distance between two multivariate Gaussians fitted to
    real and generated feature representations. Lower FID indicates better
    quality and diversity of generated samples.

    Args:
        real_features: Feature vectors from real images, shape (N, D)
        fake_features: Feature vectors from generated images, shape (M, D)

    Returns:
        FID score (float). Lower is better.

    References:
        Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
        Converge to a Local Nash Equilibrium", NeurIPS 2017
    """
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: real={real_features.shape[1]}, "
            f"fake={fake_features.shape[1]}"
        )

    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # Calculate squared difference of means
    diff = mu1 - mu2

    # Calculate matrix square root of product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Handle numerical errors (complex values)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def calculate_inception_score(
    predictions: np.ndarray, splits: int = 10, eps: float = 1e-16
) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS) from model predictions.

    IS measures both quality (confidence) and diversity (uniform marginal) of
    generated samples based on a pre-trained classifier's predictions.

    Args:
        predictions: Softmax probabilities from classifier, shape (N, num_classes)
        splits: Number of splits for computing mean and std
        eps: Small constant for numerical stability

    Returns:
        Tuple of (mean_is, std_is). Higher IS indicates better quality/diversity.

    References:
        Salimans et al., "Improved Techniques for Training GANs", NeurIPS 2016

    Note:
        This is a placeholder implementation. For full IS computation, you need
        to pass pre-computed predictions from an Inception-V3 network.
    """
    if predictions.ndim != 2:
        raise ValueError(f"Expected 2D predictions, got shape {predictions.shape}")

    # Split predictions into groups
    split_size = predictions.shape[0] // splits
    if split_size < 1:
        raise ValueError(
            f"Not enough samples ({predictions.shape[0]}) for {splits} splits"
        )

    scores = []
    for i in range(splits):
        start = i * split_size
        end = (i + 1) * split_size if i < splits - 1 else predictions.shape[0]
        part = predictions[start:end]

        # KL divergence: KL(p(y|x) || p(y))
        py = np.mean(part, axis=0) + eps  # Marginal distribution
        kl_div = part * (np.log(part + eps) - np.log(py))
        kl_div = np.sum(kl_div, axis=1)

        # Exponential of mean KL divergence
        scores.append(np.exp(np.mean(kl_div)))

    return float(np.mean(scores)), float(np.std(scores))


def calculate_precision_recall(
    real_features: np.ndarray, fake_features: np.ndarray, k: int = 5
) -> Tuple[float, float]:
    """
    Calculate precision and recall metrics for generative models.

    Precision measures if fake samples fall within the real data manifold.
    Recall measures if real samples are covered by the fake distribution.

    Args:
        real_features: Feature vectors from real images, shape (N, D)
        fake_features: Feature vectors from generated images, shape (M, D)
        k: Number of nearest neighbors for manifold estimation

    Returns:
        Tuple of (precision, recall). Both in range [0, 1], higher is better.

    References:
        Kynkäänniemi et al., "Improved Precision and Recall Metric for
        Assessing Generative Models", NeurIPS 2019
    """
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: real={real_features.shape[1]}, "
            f"fake={fake_features.shape[1]}"
        )

    # Compute k-th nearest neighbor distances within each distribution
    real_nbrs = NearestNeighbors(n_neighbors=k + 1).fit(real_features)
    real_distances, _ = real_nbrs.kneighbors(real_features)
    # Take the k-th neighbor (index k, since index 0 is the point itself)
    real_kth_distances = real_distances[:, k]

    fake_nbrs = NearestNeighbors(n_neighbors=k + 1).fit(fake_features)
    fake_distances, _ = fake_nbrs.kneighbors(fake_features)
    fake_kth_distances = fake_distances[:, k]

    # Precision: For each fake sample, find nearest real sample
    fake_to_real_nbrs = NearestNeighbors(n_neighbors=1).fit(real_features)
    fake_to_real_distances, fake_to_real_indices = fake_to_real_nbrs.kneighbors(
        fake_features
    )
    fake_to_real_distances = fake_to_real_distances.flatten()
    fake_to_real_indices = fake_to_real_indices.flatten()

    # Precision: fake is within real manifold if distance to nearest real
    # is <= that real point's k-th nearest neighbor distance
    precision = np.mean(
        fake_to_real_distances <= real_kth_distances[fake_to_real_indices]
    )

    # Recall: For each real sample, find nearest fake sample
    real_to_fake_nbrs = NearestNeighbors(n_neighbors=1).fit(fake_features)
    real_to_fake_distances, real_to_fake_indices = real_to_fake_nbrs.kneighbors(
        real_features
    )
    real_to_fake_distances = real_to_fake_distances.flatten()
    real_to_fake_indices = real_to_fake_indices.flatten()

    # Recall: real is covered by fake manifold if distance to nearest fake
    # is <= that fake point's k-th nearest neighbor distance
    recall = np.mean(real_to_fake_distances <= fake_kth_distances[real_to_fake_indices])

    return float(precision), float(recall)


def calculate_roc_auc_pr_auc(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Calculate ROC-AUC and PR-AUC for domain classification task.

    Trains a logistic regression classifier to distinguish real from fake features.
    Higher AUC values indicate larger domain gap (easier to distinguish).

    Args:
        real_features: Feature vectors from real images, shape (N, D)
        fake_features: Feature vectors from fake/synthetic images, shape (M, D)
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (roc_auc, pr_auc). Both in range [0, 1].
        Values close to 0.5 indicate similar distributions.
        Values close to 1.0 indicate easily distinguishable distributions.

    Note:
        This metric can be used to measure quality of synthetic data.
        Lower values indicate synthetic data is closer to real data.
    """
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: real={real_features.shape[1]}, "
            f"fake={fake_features.shape[1]}"
        )

    # Prepare data: real=0, fake=1
    X = np.vstack([real_features, fake_features])
    y = np.concatenate([np.zeros(len(real_features)), np.ones(len(fake_features))])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train classifier
    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    return float(roc_auc), float(pr_auc)


def calculate_wasserstein_distances(
    features1: np.ndarray, features2: np.ndarray
) -> np.ndarray:
    """
    Calculate Wasserstein (Earth Mover's) distances between feature distributions.

    Computes the 1D Wasserstein distance for each feature dimension independently.
    This provides a per-dimension measure of distribution similarity.

    Args:
        features1: Feature vectors from first distribution, shape (N, D)
        features2: Feature vectors from second distribution, shape (M, D)

    Returns:
        Array of Wasserstein distances for each dimension, shape (D,)

    Note:
        Lower distances indicate more similar distributions.
    """
    if features1.shape[1] != features2.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: features1={features1.shape[1]}, "
            f"features2={features2.shape[1]}"
        )

    distances = []
    for dim in range(features1.shape[1]):
        dist = wasserstein_distance(features1[:, dim], features2[:, dim])
        distances.append(dist)

    return np.array(distances)


# Convenience functions for common use cases
def evaluate_generative_model(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 5,
) -> dict:
    """
    Evaluate a generative model using multiple metrics.

    Args:
        real_features: Feature vectors from real images
        fake_features: Feature vectors from generated images
        k: Number of nearest neighbors for precision/recall

    Returns:
        Dictionary containing all computed metrics:
        - fid: Fréchet Inception Distance
        - precision: Precision score
        - recall: Recall score
        - roc_auc: ROC-AUC for domain classification
        - pr_auc: PR-AUC for domain classification
    """
    fid = calculate_fid(real_features, fake_features)
    precision, recall = calculate_precision_recall(real_features, fake_features, k=k)
    roc_auc, pr_auc = calculate_roc_auc_pr_auc(real_features, fake_features)

    return {
        "fid": fid,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
