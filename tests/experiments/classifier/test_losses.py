"""Tests for configurable classification losses (focal, class-balanced)."""

import pytest
import torch
import torch.nn.functional as F

from src.experiments.classifier.losses import (
    build_loss,
    class_balanced_loss,
    focal_loss,
)


@pytest.fixture
def logits_targets():
    torch.manual_seed(0)
    logits = torch.randn(16, 2)
    targets = torch.randint(0, 2, (16,))
    return logits, targets


@pytest.mark.unit
class TestFocalLoss:
    def test_gamma_zero_equals_cross_entropy(self, logits_targets):
        logits, targets = logits_targets
        focal = focal_loss(logits, targets, gamma=0.0, alpha=None)
        ce = F.cross_entropy(logits, targets)
        assert torch.allclose(focal, ce, atol=1e-6)

    def test_returns_scalar_for_mean_and_sum(self, logits_targets):
        logits, targets = logits_targets
        assert focal_loss(logits, targets, gamma=2.0).ndim == 0
        assert focal_loss(logits, targets, gamma=2.0, reduction="sum").ndim == 0

    def test_reduction_none_returns_per_sample(self, logits_targets):
        logits, targets = logits_targets
        per_sample = focal_loss(logits, targets, gamma=2.0, reduction="none")
        assert per_sample.shape == (logits.shape[0],)

    def test_gamma_downweights_easy_examples(self):
        # A confidently-correct example contributes less under focal than CE.
        logits = torch.tensor([[5.0, -5.0]])  # very confident class 0
        targets = torch.tensor([0])
        ce = F.cross_entropy(logits, targets)
        focal = focal_loss(logits, targets, gamma=2.0)
        assert focal < ce

    def test_alpha_weights_applied(self, logits_targets):
        logits, targets = logits_targets
        unweighted = focal_loss(logits, targets, gamma=2.0, reduction="none")
        alpha = torch.tensor([1.0, 3.0])
        weighted = focal_loss(logits, targets, gamma=2.0, alpha=alpha, reduction="none")
        expected = alpha[targets] * unweighted
        assert torch.allclose(weighted, expected, atol=1e-6)


@pytest.mark.unit
class TestClassBalancedLoss:
    def test_cross_entropy_base_matches_weighted_ce(self, logits_targets):
        logits, targets = logits_targets
        weights = torch.tensor([0.5, 2.0])
        cb = class_balanced_loss(logits, targets, weights=weights, base="cross_entropy")
        expected = F.cross_entropy(logits, targets, weight=weights)
        assert torch.allclose(cb, expected, atol=1e-6)

    def test_focal_base_runs(self, logits_targets):
        logits, targets = logits_targets
        weights = torch.tensor([0.5, 2.0])
        cb = class_balanced_loss(
            logits, targets, weights=weights, base="focal", gamma=2.0
        )
        assert torch.isfinite(cb)

    def test_invalid_base_raises(self, logits_targets):
        logits, targets = logits_targets
        with pytest.raises(ValueError):
            class_balanced_loss(logits, targets, weights=torch.ones(2), base="hinge")


@pytest.mark.unit
class TestBuildLoss:
    def test_cross_entropy_default(self, logits_targets):
        logits, targets = logits_targets
        fn = build_loss({"type": "cross_entropy"}, None, 2)
        assert torch.allclose(fn(logits, targets), F.cross_entropy(logits, targets))

    def test_focal_callable_matches_focal_loss(self, logits_targets):
        logits, targets = logits_targets
        fn = build_loss({"type": "focal", "gamma": 2.0, "alpha": None}, None, 2)
        assert torch.allclose(
            fn(logits, targets), focal_loss(logits, targets, gamma=2.0)
        )

    def test_class_balanced_minority_gets_higher_weight(self):
        # Class 1 is the minority (10 vs 90); its effective-number weight must exceed
        # class 0's, so build_loss must source counts correctly.
        fn = build_loss(
            {"type": "class_balanced", "beta": 0.99, "base": "cross_entropy"},
            class_counts=[90, 10],
            num_classes=2,
        )
        # Confirm ordering via loss behavior: with the same logits, a class-1
        # (minority) sample incurs more loss than a class-0 sample. Use reduction
        # "sum" so the per-class weight is observable (mean reduction divides by the
        # batch weight-sum, which cancels the weight for a single-sample batch).
        logit = torch.tensor([[0.0, 0.0]])
        loss_majority = fn(logit, torch.tensor([0]), reduction="sum")
        loss_minority = fn(logit, torch.tensor([1]), reduction="sum")
        assert loss_minority > loss_majority

    def test_class_balanced_requires_counts(self):
        with pytest.raises(ValueError):
            build_loss(
                {"type": "class_balanced", "beta": 0.99, "base": "cross_entropy"},
                class_counts=None,
                num_classes=2,
            )

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            build_loss({"type": "hinge"}, None, 2)

    def test_reduction_passthrough(self, logits_targets):
        logits, targets = logits_targets
        fn = build_loss({"type": "focal", "gamma": 1.0, "alpha": None}, None, 2)
        assert fn(logits, targets, reduction="none").shape == (logits.shape[0],)
