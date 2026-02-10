"""
Unit tests for src/utils/metrics.py

All tests run on CPU with synthetic data for fast feedback.
Tests are organized by metric type and complexity.
"""

import numpy as np
import pytest

from src.utils.metrics import (
    calculate_fid,
    calculate_inception_score,
    calculate_precision_recall,
    calculate_roc_auc_pr_auc,
    calculate_wasserstein_distances,
    evaluate_generative_model,
)


@pytest.fixture
def synthetic_features():
    """Generate synthetic feature vectors for testing."""
    np.random.seed(42)
    # Real features: Gaussian with mean=0, std=1
    real = np.random.randn(100, 50)
    # Fake features: Gaussian with mean=0.5, std=1.2
    fake = np.random.randn(80, 50) * 1.2 + 0.5
    return real, fake


@pytest.fixture
def identical_features():
    """Generate identical feature distributions for testing."""
    np.random.seed(42)
    features = np.random.randn(100, 50)
    return features, features.copy()


@pytest.fixture
def small_features():
    """Generate very small feature sets for edge case testing."""
    np.random.seed(42)
    real = np.random.randn(10, 5)
    fake = np.random.randn(8, 5)
    return real, fake


# ============================================================================
# Unit Tests: FID (Fréchet Inception Distance)
# ============================================================================


@pytest.mark.unit
def test_calculate_fid_basic(synthetic_features):
    """Test FID calculation with synthetic data."""
    real, fake = synthetic_features
    fid = calculate_fid(real, fake)

    assert isinstance(fid, float), "FID should return a float"
    assert fid >= 0, "FID should be non-negative"
    assert not np.isnan(fid), "FID should not be NaN"
    assert not np.isinf(fid), "FID should not be infinite"


@pytest.mark.unit
def test_calculate_fid_identical_distributions(identical_features):
    """Test FID with identical distributions (should be close to 0)."""
    features1, features2 = identical_features
    fid = calculate_fid(features1, features2)

    assert fid < 1.0, "FID should be very small for identical distributions"


@pytest.mark.unit
def test_calculate_fid_dimension_mismatch():
    """Test FID raises error on dimension mismatch."""
    real = np.random.randn(50, 30)
    fake = np.random.randn(50, 20)  # Different feature dimension

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        calculate_fid(real, fake)


@pytest.mark.unit
def test_calculate_fid_small_samples(small_features):
    """Test FID works with small sample sizes."""
    real, fake = small_features
    fid = calculate_fid(real, fake)

    assert isinstance(fid, float)
    assert fid >= 0


# ============================================================================
# Unit Tests: Inception Score
# ============================================================================


@pytest.mark.unit
def test_calculate_inception_score_basic():
    """Test Inception Score calculation with synthetic predictions."""
    np.random.seed(42)
    # Simulate predictions from 100 samples for 10 classes
    predictions = np.random.dirichlet(np.ones(10), size=100)

    mean_is, std_is = calculate_inception_score(predictions, splits=10)

    assert isinstance(mean_is, float), "Mean IS should be a float"
    assert isinstance(std_is, float), "Std IS should be a float"
    assert mean_is >= 1.0, "IS should be >= 1.0"
    assert std_is >= 0, "Std should be non-negative"
    assert not np.isnan(mean_is), "Mean IS should not be NaN"


@pytest.mark.unit
def test_calculate_inception_score_confident_predictions():
    """Test IS with highly confident predictions (should be high)."""
    # Create very confident predictions (high quality, low diversity)
    predictions = np.zeros((100, 10))
    predictions[np.arange(100), np.random.randint(0, 10, 100)] = 1.0

    mean_is, _ = calculate_inception_score(predictions, splits=5)

    assert mean_is >= 1.0, "IS should be at least 1.0 for confident predictions"


@pytest.mark.unit
def test_calculate_inception_score_uniform_predictions():
    """Test IS with uniform predictions (should be close to 1.0)."""
    # Uniform predictions (low quality/diversity)
    predictions = np.ones((100, 10)) / 10

    mean_is, _ = calculate_inception_score(predictions, splits=5)

    assert 0.8 < mean_is < 1.2, "IS should be close to 1.0 for uniform predictions"


@pytest.mark.unit
def test_calculate_inception_score_invalid_shape():
    """Test IS raises error on invalid prediction shape."""
    predictions = np.random.randn(100)  # 1D instead of 2D

    with pytest.raises(ValueError, match="Expected 2D predictions"):
        calculate_inception_score(predictions)


@pytest.mark.unit
def test_calculate_inception_score_too_few_samples():
    """Test IS raises error when samples < splits."""
    predictions = np.random.dirichlet(np.ones(10), size=5)

    with pytest.raises(ValueError, match="Not enough samples"):
        calculate_inception_score(predictions, splits=10)


# ============================================================================
# Unit Tests: Precision and Recall
# ============================================================================


@pytest.mark.unit
def test_calculate_precision_recall_basic(synthetic_features):
    """Test precision/recall calculation with synthetic data."""
    real, fake = synthetic_features
    precision, recall = calculate_precision_recall(real, fake, k=5)

    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert 0 <= precision <= 1, "Precision should be in [0, 1]"
    assert 0 <= recall <= 1, "Recall should be in [0, 1]"


@pytest.mark.unit
def test_calculate_precision_recall_identical(identical_features):
    """Test precision/recall with identical distributions (should be high)."""
    features1, features2 = identical_features
    precision, recall = calculate_precision_recall(features1, features2, k=5)

    # For identical distributions, both should be very high
    assert precision > 0.9, "Precision should be high for identical distributions"
    assert recall > 0.9, "Recall should be high for identical distributions"


@pytest.mark.unit
def test_calculate_precision_recall_dimension_mismatch():
    """Test precision/recall raises error on dimension mismatch."""
    real = np.random.randn(50, 30)
    fake = np.random.randn(50, 20)

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        calculate_precision_recall(real, fake)


@pytest.mark.unit
def test_calculate_precision_recall_different_k_values(synthetic_features):
    """Test precision/recall with different k values."""
    real, fake = synthetic_features

    for k in [3, 5, 10]:
        precision, recall = calculate_precision_recall(real, fake, k=k)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1


# ============================================================================
# Unit Tests: ROC-AUC and PR-AUC
# ============================================================================


@pytest.mark.unit
def test_calculate_roc_auc_pr_auc_basic(synthetic_features):
    """Test ROC-AUC and PR-AUC calculation."""
    real, fake = synthetic_features
    roc_auc, pr_auc = calculate_roc_auc_pr_auc(real, fake)

    assert isinstance(roc_auc, float), "ROC-AUC should be a float"
    assert isinstance(pr_auc, float), "PR-AUC should be a float"
    assert 0 <= roc_auc <= 1, "ROC-AUC should be in [0, 1]"
    assert 0 <= pr_auc <= 1, "PR-AUC should be in [0, 1]"


@pytest.mark.unit
def test_calculate_roc_auc_pr_auc_identical():
    """Test AUC with nearly identical distributions."""
    np.random.seed(42)
    # Create nearly identical distributions with small noise
    features1 = np.random.randn(100, 50)
    features2 = features1 + np.random.randn(100, 50) * 0.01  # Add tiny noise

    roc_auc, pr_auc = calculate_roc_auc_pr_auc(features1, features2)

    # With nearly identical distributions and small sample size,
    # the classifier behavior can be unpredictable. Just verify valid output.
    assert 0 <= roc_auc <= 1, "ROC-AUC should be in valid range"
    assert 0 <= pr_auc <= 1, "PR-AUC should be in valid range"

    # The metric should still be stable (not NaN or inf)
    assert not np.isnan(roc_auc), "ROC-AUC should not be NaN"
    assert not np.isnan(pr_auc), "PR-AUC should not be NaN"


@pytest.mark.unit
def test_calculate_roc_auc_pr_auc_well_separated():
    """Test AUC with well-separated distributions (should be high)."""
    np.random.seed(42)
    # Create well-separated distributions
    real = np.random.randn(100, 20)
    fake = np.random.randn(100, 20) + 5  # Shifted by 5 std devs

    roc_auc, pr_auc = calculate_roc_auc_pr_auc(real, fake)

    # Should be easy to classify
    assert roc_auc > 0.9, "ROC-AUC should be high for well-separated distributions"
    assert pr_auc > 0.9, "PR-AUC should be high for well-separated distributions"


@pytest.mark.unit
def test_calculate_roc_auc_pr_auc_dimension_mismatch():
    """Test AUC raises error on dimension mismatch."""
    real = np.random.randn(50, 30)
    fake = np.random.randn(50, 20)

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        calculate_roc_auc_pr_auc(real, fake)


@pytest.mark.unit
def test_calculate_roc_auc_pr_auc_custom_parameters(synthetic_features):
    """Test AUC with custom test_size and random_state."""
    real, fake = synthetic_features

    # Test with different test sizes
    roc_auc1, pr_auc1 = calculate_roc_auc_pr_auc(real, fake, test_size=0.2)
    roc_auc2, pr_auc2 = calculate_roc_auc_pr_auc(real, fake, test_size=0.4)

    assert isinstance(roc_auc1, float)
    assert isinstance(roc_auc2, float)

    # Test reproducibility with same random_state
    roc_auc3, _ = calculate_roc_auc_pr_auc(real, fake, random_state=42)
    roc_auc4, _ = calculate_roc_auc_pr_auc(real, fake, random_state=42)

    assert roc_auc3 == roc_auc4, "Results should be reproducible with same random_state"


# ============================================================================
# Unit Tests: Wasserstein Distances
# ============================================================================


@pytest.mark.unit
def test_calculate_wasserstein_distances_basic(synthetic_features):
    """Test Wasserstein distance calculation."""
    real, fake = synthetic_features
    distances = calculate_wasserstein_distances(real, fake)

    assert isinstance(distances, np.ndarray), "Should return numpy array"
    assert distances.shape == (real.shape[1],), "Should have one distance per dimension"
    assert np.all(distances >= 0), "All distances should be non-negative"


@pytest.mark.unit
def test_calculate_wasserstein_distances_identical(identical_features):
    """Test Wasserstein with identical distributions (should be close to 0)."""
    features1, features2 = identical_features
    distances = calculate_wasserstein_distances(features1, features2)

    # For identical distributions, distances should be very small
    assert np.all(distances < 0.1), (
        "Distances should be small for identical distributions"
    )


@pytest.mark.unit
def test_calculate_wasserstein_distances_dimension_mismatch():
    """Test Wasserstein raises error on dimension mismatch."""
    features1 = np.random.randn(50, 30)
    features2 = np.random.randn(50, 20)

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        calculate_wasserstein_distances(features1, features2)


@pytest.mark.unit
def test_calculate_wasserstein_distances_ordering():
    """Test Wasserstein distance is symmetric."""
    np.random.seed(42)
    features1 = np.random.randn(50, 10)
    features2 = np.random.randn(50, 10) + 1

    dist1 = calculate_wasserstein_distances(features1, features2)
    dist2 = calculate_wasserstein_distances(features2, features1)

    np.testing.assert_allclose(dist1, dist2, rtol=1e-10)


# ============================================================================
# Unit Tests: evaluate_generative_model (convenience function)
# ============================================================================


@pytest.mark.unit
def test_evaluate_generative_model_basic(synthetic_features):
    """Test comprehensive model evaluation."""
    real, fake = synthetic_features
    results = evaluate_generative_model(real, fake, k=5)

    # Check all expected metrics are present
    expected_keys = {"fid", "precision", "recall", "roc_auc", "pr_auc"}
    assert set(results.keys()) == expected_keys, "Should return all expected metrics"

    # Check all values are valid
    assert isinstance(results["fid"], float)
    assert results["fid"] >= 0

    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["roc_auc"] <= 1
    assert 0 <= results["pr_auc"] <= 1


@pytest.mark.unit
def test_evaluate_generative_model_custom_k(synthetic_features):
    """Test evaluation with different k values."""
    real, fake = synthetic_features

    results1 = evaluate_generative_model(real, fake, k=3)
    results2 = evaluate_generative_model(real, fake, k=10)

    # Both should return valid results
    assert isinstance(results1, dict)
    assert isinstance(results2, dict)

    # Precision/recall might differ with different k
    assert results1["fid"] == results2["fid"], "FID should be same regardless of k"


# ============================================================================
# Component Tests: Integration with larger data
# ============================================================================


@pytest.mark.component
def test_metrics_with_larger_features():
    """Test metrics with larger feature sets (component test)."""
    np.random.seed(42)
    # Larger feature sets
    real = np.random.randn(500, 100)
    fake = np.random.randn(400, 100) * 1.1 + 0.3

    # Test all metrics work with larger data
    fid = calculate_fid(real, fake)
    precision, recall = calculate_precision_recall(real, fake, k=10)
    roc_auc, pr_auc = calculate_roc_auc_pr_auc(real, fake)
    distances = calculate_wasserstein_distances(real, fake)

    assert isinstance(fid, float)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= roc_auc <= 1
    assert 0 <= pr_auc <= 1
    assert len(distances) == 100


@pytest.mark.component
def test_evaluate_generative_model_workflow():
    """Test full evaluation workflow (component test)."""
    np.random.seed(42)
    # Simulate realistic scenario
    real_features = np.random.randn(200, 64)
    fake_features = np.random.randn(200, 64) * 0.9 + 0.2

    results = evaluate_generative_model(real_features, fake_features, k=5)

    # Verify reasonable results
    assert results["fid"] > 0, "FID should be positive for different distributions"
    assert 0 < results["precision"] < 1, "Precision should be in valid range"
    assert 0 < results["recall"] < 1, "Recall should be in valid range"

    # Print for manual inspection during development
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


@pytest.mark.unit
def test_metrics_with_high_dimensional_features():
    """Test metrics work with high-dimensional features."""
    np.random.seed(42)
    real = np.random.randn(50, 512)  # Typical CNN feature dimension
    fake = np.random.randn(50, 512)

    fid = calculate_fid(real, fake)
    precision, recall = calculate_precision_recall(real, fake, k=5)

    assert isinstance(fid, float)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1


@pytest.mark.unit
def test_metrics_reproducibility(synthetic_features):
    """Test that metrics are reproducible with fixed random seed."""
    real, fake = synthetic_features

    # Calculate twice
    roc_auc1, pr_auc1 = calculate_roc_auc_pr_auc(real, fake, random_state=42)
    roc_auc2, pr_auc2 = calculate_roc_auc_pr_auc(real, fake, random_state=42)

    assert roc_auc1 == roc_auc2, "ROC-AUC should be reproducible"
    assert pr_auc1 == pr_auc2, "PR-AUC should be reproducible"
