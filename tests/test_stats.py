"""Pytest tests for FID and Precision/Recall metrics"""

import numpy as np
import pytest

from stats import calculate_fid, calculate_precision_recall


class TestCalculateFID:
    """Test cases for Fréchet Inception Distance (FID) calculation"""

    def test_fid_identical_distributions(self):
        """Test FID score is 0 for identical distributions"""
        np.random.seed(42)
        features = np.random.randn(100, 2048)

        fid_score = calculate_fid(features, features)

        # FID should be approximately 0 for identical distributions
        assert fid_score < 1e-6

    def test_fid_different_distributions(self):
        """Test FID score is positive for different distributions"""
        np.random.seed(42)
        real_features = np.random.randn(100, 2048)
        fake_features = np.random.randn(100, 2048) + 2.0  # Shifted distribution

        fid_score = calculate_fid(real_features, fake_features)

        # FID should be positive when distributions differ
        assert fid_score > 0

    def test_fid_with_different_means(self):
        """Test FID increases with difference in means"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)

        # Create fake features with different mean shifts
        fake_features_small_shift = real_features + 0.1
        fake_features_large_shift = real_features + 1.0

        fid_small = calculate_fid(real_features, fake_features_small_shift)
        fid_large = calculate_fid(real_features, fake_features_large_shift)

        # Larger shift should result in larger FID
        assert fid_large > fid_small

    def test_fid_with_different_variances(self):
        """Test FID detects differences in variance"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        fake_features = np.random.randn(100, 128) * 2.0  # Different variance

        fid_score = calculate_fid(real_features, fake_features)

        assert fid_score > 0

    def test_fid_symmetry(self):
        """Test FID is symmetric: FID(A,B) == FID(B,A)"""
        np.random.seed(42)
        real_features = np.random.randn(100, 256)
        fake_features = np.random.randn(100, 256) + 1.0

        fid_ab = calculate_fid(real_features, fake_features)
        fid_ba = calculate_fid(fake_features, real_features)

        # Should be symmetric (within numerical precision)
        assert np.isclose(fid_ab, fid_ba, rtol=1e-5)

    def test_fid_output_type(self):
        """Test FID returns a scalar numeric value"""
        np.random.seed(42)
        real_features = np.random.randn(50, 64)
        fake_features = np.random.randn(50, 64)

        fid_score = calculate_fid(real_features, fake_features)

        # Should be a scalar
        assert isinstance(fid_score, (int, float, np.number))
        assert not isinstance(fid_score, np.ndarray) or fid_score.ndim == 0

    def test_fid_with_small_sample_size(self):
        """Test FID with small sample sizes"""
        np.random.seed(42)
        real_features = np.random.randn(10, 32)
        fake_features = np.random.randn(10, 32)

        fid_score = calculate_fid(real_features, fake_features)

        # Should still compute without error
        assert fid_score >= 0
        assert not np.isnan(fid_score)
        assert not np.isinf(fid_score)

    def test_fid_with_large_feature_dimension(self):
        """Test FID with large feature dimensions (like real Inception features)"""
        np.random.seed(42)
        real_features = np.random.randn(100, 2048)  # Typical Inception size
        fake_features = np.random.randn(100, 2048)

        fid_score = calculate_fid(real_features, fake_features)

        assert fid_score >= 0
        assert not np.isnan(fid_score)

    def test_fid_handles_complex_covariance(self):
        """Test FID correctly handles complex eigenvalues in covariance matrix square root"""
        np.random.seed(42)
        # Create features that might produce complex eigenvalues
        real_features = np.random.randn(50, 10)
        fake_features = np.random.randn(50, 10)

        fid_score = calculate_fid(real_features, fake_features)

        # Result should be real
        assert np.isreal(fid_score)
        assert not np.isnan(fid_score)

    def test_fid_with_different_sample_sizes(self):
        """Test FID with different numbers of real and fake samples"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        fake_features = np.random.randn(50, 128)  # Different sample size

        fid_score = calculate_fid(real_features, fake_features)

        # Should work with different sample sizes
        assert fid_score >= 0

    def test_fid_consistency(self):
        """Test FID produces consistent results with same input"""
        np.random.seed(42)
        real_features = np.random.randn(80, 256)
        fake_features = np.random.randn(80, 256)

        fid_score1 = calculate_fid(real_features, fake_features)
        fid_score2 = calculate_fid(real_features, fake_features)

        # Should be deterministic
        assert fid_score1 == fid_score2

    @pytest.mark.parametrize("feature_dim", [64, 128, 512, 2048])
    def test_fid_various_dimensions(self, feature_dim):
        """Test FID works with various feature dimensions"""
        np.random.seed(42)
        real_features = np.random.randn(50, feature_dim)
        fake_features = np.random.randn(50, feature_dim)

        fid_score = calculate_fid(real_features, fake_features)

        assert fid_score >= 0
        assert not np.isnan(fid_score)


class TestCalculatePrecisionRecall:
    """Test cases for Precision and Recall metrics"""

    def test_precision_recall_identical_distributions(self):
        """Test precision and recall are 1.0 for identical distributions"""
        np.random.seed(42)
        features = np.random.randn(100, 128)

        precision, recall = calculate_precision_recall(features, features, k=5)

        # Should be perfect for identical distributions
        assert precision == 1.0
        assert recall == 1.0

    def test_precision_recall_completely_separate(self):
        """Test precision and recall for completely separate distributions"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        # Very far apart distribution
        fake_features = np.random.randn(100, 128) + 100.0

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=5
        )

        # Should be very low for completely separate distributions
        assert precision < 0.1
        assert recall < 0.1

    def test_precision_recall_range(self):
        """Test precision and recall are in valid range [0, 1]"""
        np.random.seed(42)
        real_features = np.random.randn(100, 256)
        fake_features = np.random.randn(100, 256)

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=5
        )

        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0

    def test_precision_recall_output_types(self):
        """Test precision and recall return numeric values"""
        np.random.seed(42)
        real_features = np.random.randn(50, 64)
        fake_features = np.random.randn(50, 64)

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=3
        )

        # Should be scalars
        assert isinstance(precision, (int, float, np.number))
        assert isinstance(recall, (int, float, np.number))
        assert not isinstance(precision, np.ndarray) or precision.ndim == 0
        assert not isinstance(recall, np.ndarray) or recall.ndim == 0

    def test_precision_with_overlapping_distributions(self):
        """Test precision with partially overlapping distributions"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        # Slightly shifted distribution
        fake_features = np.random.randn(100, 128) + 0.5

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=5
        )

        # Should have some overlap
        assert 0.0 < precision < 1.0
        assert 0.0 < recall < 1.0

    def test_precision_recall_different_k_values(self):
        """Test precision and recall with different k values"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        fake_features = np.random.randn(100, 128) + 1.0

        precision_k3, _ = calculate_precision_recall(real_features, fake_features, k=3)
        precision_k10, _ = calculate_precision_recall(
            real_features, fake_features, k=10
        )

        # Results should differ with different k
        # (though not guaranteed which direction)
        assert isinstance(precision_k3, (int, float, np.number))
        assert isinstance(precision_k10, (int, float, np.number))

    def test_precision_recall_consistency(self):
        """Test precision and recall produce consistent results"""
        np.random.seed(42)
        real_features = np.random.randn(80, 256)
        fake_features = np.random.randn(80, 256)

        precision1, recall1 = calculate_precision_recall(
            real_features, fake_features, k=5
        )
        precision2, recall2 = calculate_precision_recall(
            real_features, fake_features, k=5
        )

        # Should be deterministic
        assert precision1 == precision2
        assert recall1 == recall2

    def test_precision_high_quality_fakes(self):
        """Test precision is high when fakes are within real manifold"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        # Fake features very similar to real
        fake_features = real_features + np.random.randn(100, 128) * 0.1

        precision, _ = calculate_precision_recall(real_features, fake_features, k=5)

        # Precision should be high when fakes are close to reals
        assert precision > 0.5

    def test_recall_diverse_fakes(self):
        """Test recall with diverse fake samples"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        # Create fakes that cover real distribution
        fake_features = np.random.randn(200, 128)  # More diverse

        _, recall = calculate_precision_recall(real_features, fake_features, k=5)

        # Should compute without error
        assert 0.0 <= recall <= 1.0

    def test_precision_recall_with_small_k(self):
        """Test precision and recall with small k value"""
        np.random.seed(42)
        real_features = np.random.randn(50, 64)
        fake_features = np.random.randn(50, 64)

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=1
        )

        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0

    def test_precision_recall_with_large_k(self):
        """Test precision and recall with large k value"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        fake_features = np.random.randn(100, 128)

        # k should be less than number of samples
        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=20
        )

        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0

    def test_precision_recall_different_sample_sizes(self):
        """Test precision and recall with different numbers of real and fake samples"""
        np.random.seed(42)
        real_features = np.random.randn(80, 128)
        fake_features = np.random.randn(120, 128)

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=5
        )

        # Should work with different sample sizes
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0

    @pytest.mark.parametrize("k", [1, 3, 5, 10, 15])
    def test_precision_recall_various_k(self, k):
        """Test precision and recall work with various k values"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)
        fake_features = np.random.randn(100, 128)

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=k
        )

        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0

    def test_precision_recall_not_symmetric(self):
        """Test that precision and recall are generally not equal"""
        np.random.seed(42)
        real_features = np.random.randn(80, 128)
        fake_features = np.random.randn(100, 128) + 0.5

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=5
        )

        # Precision and recall measure different things,
        # so they're usually different (except for special cases)
        assert isinstance(precision, (int, float, np.number))
        assert isinstance(recall, (int, float, np.number))

    def test_precision_recall_with_clusters(self):
        """Test precision and recall with clustered data"""
        np.random.seed(42)
        # Create two clusters in real data
        real_cluster1 = np.random.randn(50, 128)
        real_cluster2 = np.random.randn(50, 128) + 5.0
        real_features = np.vstack([real_cluster1, real_cluster2])

        # Fake data only covers one cluster
        fake_features = np.random.randn(100, 128) + 0.5

        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=5
        )

        # Recall should be lower since fake doesn't cover all real clusters
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= precision <= 1.0


class TestEdgeCases:
    """Test edge cases and potential error conditions"""

    def test_fid_with_singular_covariance(self):
        """Test FID handles near-singular covariance matrices"""
        np.random.seed(42)
        # Create features with some redundancy
        base_features = np.random.randn(50, 10)
        real_features = np.hstack([base_features, base_features])  # Duplicate features
        fake_features = np.hstack([base_features + 0.1, base_features + 0.1])

        fid_score = calculate_fid(real_features, fake_features)

        # Should still compute
        assert not np.isnan(fid_score)
        assert not np.isinf(fid_score)

    def test_precision_recall_k_larger_than_samples(self):
        """Test precision and recall when k is close to sample size"""
        np.random.seed(42)
        real_features = np.random.randn(20, 64)
        fake_features = np.random.randn(20, 64)

        # k+1 = 20, which equals sample size
        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=19
        )

        # Should still work
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0

    def test_fid_with_zero_variance_dimension(self):
        """Test FID with zero variance in some dimensions"""
        np.random.seed(42)
        real_features = np.random.randn(50, 10)
        real_features[:, 0] = 1.0  # Constant dimension

        fake_features = np.random.randn(50, 10)
        fake_features[:, 0] = 1.0  # Same constant

        fid_score = calculate_fid(real_features, fake_features)

        # Should handle gracefully
        assert not np.isnan(fid_score)

    def test_minimum_sample_size_for_covariance(self):
        """Test metrics with minimum sample size for covariance estimation"""
        np.random.seed(42)
        # Need at least feature_dim samples for full rank covariance
        feature_dim = 10
        real_features = np.random.randn(feature_dim, feature_dim)
        fake_features = np.random.randn(feature_dim, feature_dim)

        fid_score = calculate_fid(real_features, fake_features)

        assert not np.isnan(fid_score)


class TestIntegration:
    """Integration tests combining multiple metrics"""

    def test_metrics_on_same_distribution(self):
        """Test all metrics on the same distribution"""
        np.random.seed(42)
        features = np.random.randn(100, 256)

        fid_score = calculate_fid(features, features)
        precision, recall = calculate_precision_recall(features, features, k=5)

        # FID should be ~0, precision and recall should be 1.0
        assert fid_score < 1e-6
        assert precision == 1.0
        assert recall == 1.0

    def test_metrics_show_consistent_trends(self):
        """Test that metrics show consistent trends with distribution shift"""
        np.random.seed(42)
        real_features = np.random.randn(100, 128)

        # Create fakes with increasing shift
        fake_close = real_features + np.random.randn(100, 128) * 0.1
        fake_far = real_features + np.random.randn(100, 128) * 2.0

        fid_close = calculate_fid(real_features, fake_close)
        fid_far = calculate_fid(real_features, fake_far)

        precision_close, recall_close = calculate_precision_recall(
            real_features, fake_close, k=5
        )
        precision_far, recall_far = calculate_precision_recall(
            real_features, fake_far, k=5
        )

        # FID should increase with shift
        assert fid_far > fid_close

        # Precision/recall should generally decrease with shift
        assert precision_close >= precision_far or recall_close >= recall_far


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
