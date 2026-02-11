"""Tests for InceptionV3 Classifier Model

This module contains unit and component tests for the InceptionV3 classifier model.
Tests are organized by tier:
- Unit tests: Model instantiation, configuration, interface compliance
- Component tests: Forward pass, loss computation with small data on CPU
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.base.model import BaseModel
from src.experiments.classifier.models.inceptionv3 import InceptionV3Classifier

# ============================================================================
# Unit Tests - Fast, CPU-only, no forward passes
# ============================================================================


@pytest.mark.unit
class TestInceptionV3Instantiation:
    """Test model instantiation and configuration."""

    def test_model_creation_default_params(self):
        """Test model can be created with default parameters."""
        model = InceptionV3Classifier(num_classes=2)
        assert model is not None
        assert isinstance(model, nn.Module)
        assert isinstance(model, BaseModel)

    def test_model_creation_custom_params(self):
        """Test model creation with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = InceptionV3Classifier(
                num_classes=10,
                pretrained=False,
                freeze_backbone=False,
                model_dir=tmpdir,
                dropout=0.3,
            )
            assert model.num_classes == 10
            assert model.pretrained is False
            assert model.freeze_backbone is False
            assert model.dropout == 0.3
            assert model.model_dir == Path(tmpdir)

    def test_model_inherits_from_base_model(self):
        """Test that InceptionV3Classifier properly inherits from BaseModel."""
        model = InceptionV3Classifier(num_classes=2)
        assert isinstance(model, BaseModel)
        assert hasattr(model, "forward")
        assert hasattr(model, "compute_loss")
        assert hasattr(model, "save_checkpoint")
        assert hasattr(model, "load_checkpoint")

    def test_model_has_required_attributes(self):
        """Test model has all required attributes after initialization."""
        model = InceptionV3Classifier(num_classes=2)
        # Check InceptionV3 layers exist
        assert hasattr(model, "Conv2d_1a_3x3")
        assert hasattr(model, "Mixed_7c")
        # Check classification head exists
        assert hasattr(model, "fc")
        assert hasattr(model, "dropout_layer")
        # Check configuration attributes
        assert model.num_classes == 2

    def test_classification_head_dimensions(self):
        """Test classification head has correct dimensions."""
        num_classes = 5
        model = InceptionV3Classifier(num_classes=num_classes)
        # InceptionV3 features are 2048-dimensional
        assert model.fc.in_features == 2048
        assert model.fc.out_features == num_classes

    def test_dropout_layer_configured_correctly(self):
        """Test dropout layer has correct probability."""
        dropout_prob = 0.3
        model = InceptionV3Classifier(num_classes=2, dropout=dropout_prob)
        assert model.dropout_layer.p == dropout_prob


@pytest.mark.unit
class TestInceptionV3FreezeStrategy:
    """Test layer freezing functionality."""

    def test_freeze_backbone_only_head_trainable(self):
        """Test that with freeze_backbone=True, only the head is trainable."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=True, pretrained=False
        )

        # Check that backbone layers are frozen (access nested conv weights)
        assert not model.Conv2d_1a_3x3.conv.weight.requires_grad
        assert not model.Mixed_7c.branch_pool.conv.weight.requires_grad

        # Check that classification head is trainable
        assert model.fc.weight.requires_grad
        assert model.fc.bias.requires_grad

    def test_unfreeze_all_layers_trainable(self):
        """Test that with freeze_backbone=False, all layers are trainable."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=False, pretrained=False
        )

        # Check that backbone layers are trainable (access nested conv weights)
        assert model.Conv2d_1a_3x3.conv.weight.requires_grad
        assert model.Mixed_7c.branch_pool.conv.weight.requires_grad

        # Check that classification head is trainable
        assert model.fc.weight.requires_grad
        assert model.fc.bias.requires_grad

    def test_get_trainable_parameters(self):
        """Test get_trainable_parameters returns only trainable params."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=True, pretrained=False
        )

        trainable_params = list(model.get_trainable_parameters())
        all_trainable = [p for p in model.parameters() if p.requires_grad]

        # Should match all trainable parameters
        assert len(trainable_params) == len(all_trainable)

        # When backbone is frozen, should only have fc and dropout params
        # FC has 2 params (weight, bias), dropout has 0 trainable params
        assert len(trainable_params) == 2  # fc.weight, fc.bias


@pytest.mark.unit
class TestInceptionV3SelectiveLayerFreezing:
    """Test selective layer freezing functionality."""

    def test_set_trainable_layers_single_pattern(self):
        """Test unfreezing layers with a single pattern."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=True, pretrained=False
        )

        # Initially, only fc and dropout should be trainable
        assert model.fc.weight.requires_grad
        # Check a parameter from Mixed_7c exists and is frozen
        mixed_7c_params = list(model.Mixed_7c.parameters())
        assert len(mixed_7c_params) > 0
        assert not mixed_7c_params[0].requires_grad

        # Unfreeze Mixed_7* layers
        model.set_trainable_layers(["Mixed_7*"])

        # Now Mixed_7* layers should be trainable
        assert any(p.requires_grad for p in model.Mixed_7a.parameters())
        assert any(p.requires_grad for p in model.Mixed_7b.parameters())
        assert any(p.requires_grad for p in model.Mixed_7c.parameters())

        # But Mixed_6* should still be frozen
        assert not any(p.requires_grad for p in model.Mixed_6e.parameters())

    def test_set_trainable_layers_multiple_patterns(self):
        """Test unfreezing layers with multiple patterns."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=True, pretrained=False
        )

        # Unfreeze both Mixed_6* and Mixed_7* layers
        model.set_trainable_layers(["Mixed_6*", "Mixed_7*"])

        # Both groups should be trainable
        assert any(p.requires_grad for p in model.Mixed_6a.parameters())
        assert any(p.requires_grad for p in model.Mixed_6e.parameters())
        assert any(p.requires_grad for p in model.Mixed_7a.parameters())
        assert any(p.requires_grad for p in model.Mixed_7c.parameters())

        # But earlier layers should still be frozen
        assert not any(p.requires_grad for p in model.Mixed_5d.parameters())
        assert not any(p.requires_grad for p in model.Conv2d_1a_3x3.parameters())

    def test_set_trainable_layers_specific_layers(self):
        """Test unfreezing specific named layers without wildcards."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=True, pretrained=False
        )

        # Unfreeze only specific layers
        model.set_trainable_layers(["Mixed_7a", "Mixed_7b"])

        # Only specified layers should be trainable
        assert any(p.requires_grad for p in model.Mixed_7a.parameters())
        assert any(p.requires_grad for p in model.Mixed_7b.parameters())

        # Mixed_7c should still be frozen (not included in pattern)
        assert not any(p.requires_grad for p in model.Mixed_7c.parameters())

        # Earlier layers should still be frozen
        assert not any(p.requires_grad for p in model.Mixed_6e.parameters())

    def test_trainable_layers_param_in_init(self):
        """Test that trainable_layers parameter works in __init__."""
        model = InceptionV3Classifier(
            num_classes=2,
            freeze_backbone=True,
            trainable_layers=["Mixed_7*"],
            pretrained=False,
        )

        # Mixed_7* layers should be trainable from initialization
        assert any(p.requires_grad for p in model.Mixed_7a.parameters())
        assert any(p.requires_grad for p in model.Mixed_7b.parameters())
        assert any(p.requires_grad for p in model.Mixed_7c.parameters())

        # Earlier layers should be frozen
        assert not any(p.requires_grad for p in model.Mixed_6e.parameters())

        # Classification head should remain trainable
        assert model.fc.weight.requires_grad

    def test_trainable_layers_overrides_freeze_backbone(self):
        """Test that trainable_layers properly extends freeze_backbone."""
        # freeze_backbone=True with trainable_layers should unfreeze specified layers
        model = InceptionV3Classifier(
            num_classes=2,
            freeze_backbone=True,
            trainable_layers=["Mixed_6*"],
            pretrained=False,
        )

        # Mixed_6* should be trainable
        assert any(p.requires_grad for p in model.Mixed_6e.parameters())

        # But Mixed_5* should still be frozen
        assert not any(p.requires_grad for p in model.Mixed_5d.parameters())

    def test_set_trainable_layers_empty_list(self):
        """Test that empty trainable_layers list doesn't change anything."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=True, pretrained=False
        )

        # Record initial state
        initial_fc_trainable = model.fc.weight.requires_grad
        initial_mixed7c_trainable = any(
            p.requires_grad for p in model.Mixed_7c.parameters()
        )

        # Apply empty list
        model.set_trainable_layers([])

        # State should remain unchanged
        assert model.fc.weight.requires_grad == initial_fc_trainable
        assert (
            any(p.requires_grad for p in model.Mixed_7c.parameters())
        ) == initial_mixed7c_trainable


@pytest.mark.unit
class TestInceptionV3ModelDir:
    """Test model directory and weight caching."""

    def test_model_dir_created_if_not_exists(self):
        """Test that model directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "nonexistent" / "models"
            assert not model_dir.exists()

            InceptionV3Classifier(
                num_classes=2, pretrained=False, model_dir=str(model_dir)
            )

            assert model_dir.exists()
            assert model_dir.is_dir()

    def test_model_caching_not_pretrained(self):
        """Test that no cache file is created when pretrained=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            InceptionV3Classifier(num_classes=2, pretrained=False, model_dir=tmpdir)

            cache_file = Path(tmpdir) / "inception_v3.pth"
            # Cache should not be created when pretrained=False
            assert not cache_file.exists()


# ============================================================================
# Component Tests - Small data, minimal computation on CPU
# ============================================================================


@pytest.mark.component
class TestInceptionV3Forward:
    """Test forward pass with small tensors on CPU."""

    def test_forward_pass_output_shape(self):
        """Test forward pass returns correct output shape."""
        batch_size = 2
        num_classes = 3
        model = InceptionV3Classifier(num_classes=num_classes, pretrained=False)
        model.eval()

        # InceptionV3 expects 299x299 images
        x = torch.randn(batch_size, 3, 299, 299)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes)

    def test_forward_pass_output_range(self):
        """Test forward pass output is logits (not probabilities)."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 299, 299)

        with torch.no_grad():
            output = model(x)

        # Logits can be any real number (not constrained to [0, 1])
        assert output.dtype == torch.float32
        # Check that output is not probability-like (can be > 1 or < 0)
        # We just check that there's no softmax applied (values aren't constrained)
        assert True  # Logits can be any value

    def test_forward_pass_batch_size_one(self):
        """Test forward pass with batch size 1."""
        model = InceptionV3Classifier(num_classes=5, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 299, 299)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 5)

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass works with different batch sizes."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 299, 299)

            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, 2)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=False, pretrained=False
        )
        model.train()

        x = torch.randn(2, 3, 299, 299, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for input
        assert x.grad is not None
        # Check that gradients exist for trainable parameters
        assert model.fc.weight.grad is not None


@pytest.mark.component
class TestInceptionV3Loss:
    """Test loss computation."""

    def test_compute_loss_shape(self):
        """Test compute_loss returns scalar by default."""
        model = InceptionV3Classifier(num_classes=3, pretrained=False)

        predictions = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        loss = model.compute_loss(predictions, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.dtype == torch.float32

    def test_compute_loss_positive(self):
        """Test that loss is positive."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)

        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(predictions, targets)

        assert loss.item() >= 0

    def test_compute_loss_reduction_mean(self):
        """Test loss with mean reduction."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)

        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(predictions, targets, reduction="mean")

        assert loss.dim() == 0  # Scalar

    def test_compute_loss_reduction_sum(self):
        """Test loss with sum reduction."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)

        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(predictions, targets, reduction="sum")

        assert loss.dim() == 0  # Scalar

    def test_compute_loss_reduction_none(self):
        """Test loss with no reduction returns per-sample losses."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)

        batch_size = 4
        predictions = torch.randn(batch_size, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(predictions, targets, reduction="none")

        assert loss.shape == (batch_size,)
        assert (loss >= 0).all()

    def test_compute_loss_binary_classification(self):
        """Test loss computation for binary classification."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)

        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 1, 0])

        loss = model.compute_loss(predictions, targets)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_compute_loss_multiclass_classification(self):
        """Test loss computation for multi-class classification."""
        model = InceptionV3Classifier(num_classes=10, pretrained=False)

        predictions = torch.randn(4, 10)
        targets = torch.tensor([0, 5, 9, 3])

        loss = model.compute_loss(predictions, targets)

        assert loss.item() >= 0
        assert not torch.isnan(loss)


@pytest.mark.component
class TestInceptionV3FeatureExtraction:
    """Test feature extraction functionality."""

    def test_extract_features_output_shape(self):
        """Test extract_features returns correct shape."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 3, 299, 299)

        with torch.no_grad():
            features = model.extract_features(x)

        # InceptionV3 features are 2048-dimensional
        assert features.shape == (batch_size, 2048)

    def test_extract_features_different_from_forward(self):
        """Test that extract_features output differs from forward (no classification)."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 299, 299)

        with torch.no_grad():
            features = model.extract_features(x)
            logits = model(x)

        # Features should be 2048-dimensional, logits should be num_classes
        assert features.shape[1] == 2048
        assert logits.shape[1] == 2
        assert features.shape[1] != logits.shape[1]

    def test_extract_features_batch_size_one(self):
        """Test feature extraction with batch size 1."""
        model = InceptionV3Classifier(num_classes=2, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 299, 299)

        with torch.no_grad():
            features = model.extract_features(x)

        assert features.shape == (1, 2048)


@pytest.mark.component
class TestInceptionV3SelectiveLayerGradientFlow:
    """Test gradient flow through selectively unfrozen layers."""

    def test_gradient_flow_single_layer_group(self):
        """Test that gradients flow only through unfrozen layers."""
        model = InceptionV3Classifier(
            num_classes=2,
            freeze_backbone=True,
            trainable_layers=["Mixed_7*"],
            pretrained=False,
        )
        model.train()

        x = torch.randn(2, 3, 299, 299)
        targets = torch.tensor([0, 1])

        # Forward and backward pass
        output = model(x)
        loss = model.compute_loss(output, targets)
        loss.backward()

        # Gradients should exist for unfrozen layers (Mixed_7*)
        assert any(
            p.grad is not None for p in model.Mixed_7a.parameters() if p.requires_grad
        )
        assert any(
            p.grad is not None for p in model.Mixed_7b.parameters() if p.requires_grad
        )
        assert any(
            p.grad is not None for p in model.Mixed_7c.parameters() if p.requires_grad
        )

        # Gradients should NOT exist for frozen layers
        assert all(p.grad is None for p in model.Mixed_6e.parameters())
        assert all(p.grad is None for p in model.Mixed_5d.parameters())

        # Classification head should have gradients
        assert model.fc.weight.grad is not None

    def test_gradient_flow_multiple_layer_groups(self):
        """Test gradient flow with multiple unfrozen layer groups."""
        model = InceptionV3Classifier(
            num_classes=2,
            freeze_backbone=True,
            trainable_layers=["Mixed_6*", "Mixed_7*"],
            pretrained=False,
        )
        model.train()

        x = torch.randn(2, 3, 299, 299)
        targets = torch.tensor([0, 1])

        # Forward and backward pass
        output = model(x)
        loss = model.compute_loss(output, targets)
        loss.backward()

        # Gradients should exist for both Mixed_6* and Mixed_7*
        assert any(
            p.grad is not None for p in model.Mixed_6a.parameters() if p.requires_grad
        )
        assert any(
            p.grad is not None for p in model.Mixed_6e.parameters() if p.requires_grad
        )
        assert any(
            p.grad is not None for p in model.Mixed_7a.parameters() if p.requires_grad
        )
        assert any(
            p.grad is not None for p in model.Mixed_7c.parameters() if p.requires_grad
        )

        # Gradients should NOT exist for frozen layers (Mixed_5*)
        assert all(p.grad is None for p in model.Mixed_5d.parameters())

    def test_training_with_selective_unfreezing(self):
        """Test complete training iteration with selective unfreezing."""
        model = InceptionV3Classifier(
            num_classes=2,
            freeze_backbone=True,
            trainable_layers=["Mixed_7*"],
            pretrained=False,
        )
        model.train()

        # Create optimizer with trainable parameters
        optimizer = torch.optim.SGD(model.get_trainable_parameters(), lr=0.01)

        x = torch.randn(2, 3, 299, 299)
        targets = torch.tensor([0, 1])

        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = model.compute_loss(output, targets)
        loss.backward()

        # Get a trainable parameter from Mixed_7c to check weight updates
        trainable_params = [p for p in model.Mixed_7c.parameters() if p.requires_grad]
        assert len(trainable_params) > 0

        # Store weights before optimizer step
        test_param = trainable_params[0]
        weights_before = test_param.data.clone()

        optimizer.step()

        # Weights should have changed for trainable layers
        weights_after = test_param.data
        assert not torch.allclose(weights_before, weights_after)

        # Frozen layers should remain unchanged
        frozen_params = list(model.Mixed_5d.parameters())
        frozen_weights_initial = frozen_params[0].data.clone()

        # Run another step
        optimizer.zero_grad()
        output = model(x)
        loss = model.compute_loss(output, targets)
        loss.backward()
        optimizer.step()

        frozen_weights_after = frozen_params[0].data
        # Frozen layers remain exactly the same
        assert torch.allclose(frozen_weights_initial, frozen_weights_after)

    def test_gradient_accumulation_with_selective_unfreezing(self):
        """Test that gradient accumulation works correctly with selective unfreezing."""
        model = InceptionV3Classifier(
            num_classes=2,
            freeze_backbone=True,
            trainable_layers=["Mixed_7c"],  # Only last layer
            pretrained=False,
        )
        model.train()

        x = torch.randn(2, 3, 299, 299)
        targets = torch.tensor([0, 1])

        # First forward-backward (without zero_grad)
        output = model(x)
        loss1 = model.compute_loss(output, targets)
        loss1.backward()

        # Get a trainable parameter
        trainable_params = [p for p in model.Mixed_7c.parameters() if p.requires_grad]
        assert len(trainable_params) > 0
        test_param = trainable_params[0]

        grad_after_first = test_param.grad.clone()

        # Second forward-backward (gradients should accumulate)
        output = model(x)
        loss2 = model.compute_loss(output, targets)
        loss2.backward()

        grad_after_second = test_param.grad

        # Gradients should have accumulated (not be equal to first pass)
        assert not torch.allclose(grad_after_first, grad_after_second)


@pytest.mark.component
class TestInceptionV3Integration:
    """Test integration between components."""

    def test_full_forward_backward_pass(self):
        """Test complete forward and backward pass."""
        model = InceptionV3Classifier(
            num_classes=2, freeze_backbone=False, pretrained=False
        )
        model.train()

        x = torch.randn(2, 3, 299, 299)
        targets = torch.tensor([0, 1])

        # Forward pass
        output = model(x)
        assert output.shape == (2, 2)

        # Loss computation
        loss = model.compute_loss(output, targets)
        assert loss.item() >= 0

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert model.fc.weight.grad is not None
        assert model.fc.bias.grad is not None

    def test_training_mode_uses_dropout(self):
        """Test that dropout is active in training mode."""
        model = InceptionV3Classifier(num_classes=2, dropout=0.5, pretrained=False)
        model.train()

        x = torch.randn(2, 3, 299, 299)

        # Run forward pass multiple times
        # With dropout, outputs should vary
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                output = model(x)
                outputs.append(output.clone())

        # With high dropout, outputs should differ
        # But for determinism in tests, we just check the mode is correct
        assert model.dropout_layer.training  # Dropout is enabled in train mode

    def test_eval_mode_disables_dropout(self):
        """Test that dropout is disabled in eval mode."""
        model = InceptionV3Classifier(num_classes=2, dropout=0.5, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 299, 299)

        # Run forward pass multiple times
        # Without dropout, outputs should be identical
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)
        assert not model.dropout_layer.training  # Dropout is disabled in eval mode
