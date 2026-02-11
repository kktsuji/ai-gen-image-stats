"""Tests for ResNet Classifier Model

This module contains unit and component tests for the ResNet classifier models.
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
from src.experiments.classifier.models.resnet import ResNetClassifier

# ============================================================================
# Unit Tests - Fast, CPU-only, no forward passes
# ============================================================================


@pytest.mark.unit
class TestResNetInstantiation:
    """Test model instantiation and configuration."""

    def test_model_creation_default_params(self):
        """Test model can be created with default parameters."""
        model = ResNetClassifier(num_classes=2)
        assert model is not None
        assert isinstance(model, nn.Module)
        assert isinstance(model, BaseModel)

    def test_model_creation_resnet50(self):
        """Test ResNet50 variant creation."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        assert model.variant == "resnet50"
        assert model._feature_dim == 2048

    def test_model_creation_resnet101(self):
        """Test ResNet101 variant creation."""
        model = ResNetClassifier(num_classes=2, variant="resnet101", pretrained=False)
        assert model.variant == "resnet101"
        assert model._feature_dim == 2048

    def test_model_creation_resnet152(self):
        """Test ResNet152 variant creation."""
        model = ResNetClassifier(num_classes=2, variant="resnet152", pretrained=False)
        assert model.variant == "resnet152"
        assert model._feature_dim == 2048

    def test_model_creation_invalid_variant(self):
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported ResNet variant"):
            ResNetClassifier(num_classes=2, variant="resnet34", pretrained=False)

    def test_model_creation_custom_params(self):
        """Test model creation with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = ResNetClassifier(
                num_classes=10,
                variant="resnet101",
                pretrained=False,
                freeze_backbone=False,
                model_dir=tmpdir,
                dropout=0.5,
            )
            assert model.num_classes == 10
            assert model.variant == "resnet101"
            assert model.pretrained is False
            assert model.freeze_backbone is False
            assert model.dropout == 0.5
            assert model.model_dir == Path(tmpdir)

    def test_model_inherits_from_base_model(self):
        """Test that ResNetClassifier properly inherits from BaseModel."""
        model = ResNetClassifier(num_classes=2, pretrained=False)
        assert isinstance(model, BaseModel)
        assert hasattr(model, "forward")
        assert hasattr(model, "compute_loss")
        assert hasattr(model, "save_checkpoint")
        assert hasattr(model, "load_checkpoint")

    def test_model_has_required_attributes(self):
        """Test model has all required attributes after initialization."""
        model = ResNetClassifier(num_classes=2, pretrained=False)
        # Check ResNet layers exist
        assert hasattr(model, "conv1")
        assert hasattr(model, "bn1")
        assert hasattr(model, "layer1")
        assert hasattr(model, "layer2")
        assert hasattr(model, "layer3")
        assert hasattr(model, "layer4")
        assert hasattr(model, "avgpool")
        # Check classification head exists
        assert hasattr(model, "fc")
        assert hasattr(model, "dropout_layer")
        # Check configuration attributes
        assert model.num_classes == 2

    def test_classification_head_dimensions(self):
        """Test classification head has correct dimensions."""
        num_classes = 5
        model = ResNetClassifier(num_classes=num_classes, pretrained=False)
        # ResNet features are 2048-dimensional for ResNet50+
        assert model.fc.in_features == 2048
        assert model.fc.out_features == num_classes

    def test_dropout_layer_configured_correctly_with_dropout(self):
        """Test dropout layer has correct probability when dropout > 0."""
        dropout_prob = 0.3
        model = ResNetClassifier(num_classes=2, dropout=dropout_prob, pretrained=False)
        assert isinstance(model.dropout_layer, nn.Dropout)
        assert model.dropout_layer.p == dropout_prob

    def test_dropout_layer_is_identity_when_no_dropout(self):
        """Test dropout layer is Identity when dropout = 0."""
        model = ResNetClassifier(num_classes=2, dropout=0.0, pretrained=False)
        assert isinstance(model.dropout_layer, nn.Identity)


@pytest.mark.unit
class TestResNetFreezeStrategy:
    """Test layer freezing functionality."""

    def test_freeze_backbone_only_head_trainable(self):
        """Test that with freeze_backbone=True, only the head is trainable."""
        model = ResNetClassifier(num_classes=2, freeze_backbone=True, pretrained=False)

        # Check that backbone layers are frozen
        assert not any(p.requires_grad for p in model.conv1.parameters())
        assert not any(p.requires_grad for p in model.layer1.parameters())
        assert not any(p.requires_grad for p in model.layer4.parameters())

        # Check that classification head is trainable
        assert model.fc.weight.requires_grad
        assert model.fc.bias.requires_grad

    def test_unfreeze_all_layers_trainable(self):
        """Test that with freeze_backbone=False, all layers are trainable."""
        model = ResNetClassifier(num_classes=2, freeze_backbone=False, pretrained=False)

        # Check that backbone layers are trainable
        assert any(p.requires_grad for p in model.conv1.parameters())
        assert any(p.requires_grad for p in model.layer1.parameters())
        assert any(p.requires_grad for p in model.layer4.parameters())

        # Check that classification head is trainable
        assert model.fc.weight.requires_grad
        assert model.fc.bias.requires_grad

    def test_get_trainable_parameters(self):
        """Test get_trainable_parameters returns only trainable params."""
        model = ResNetClassifier(num_classes=2, freeze_backbone=True, pretrained=False)

        trainable_params = list(model.get_trainable_parameters())
        all_trainable = [p for p in model.parameters() if p.requires_grad]

        # Should match all trainable parameters
        assert len(trainable_params) == len(all_trainable)

        # When backbone is frozen, should only have fc params
        # FC has 2 params (weight, bias)
        assert len(trainable_params) == 2  # fc.weight, fc.bias


@pytest.mark.unit
class TestResNetSelectiveLayerFreezing:
    """Test selective layer freezing functionality."""

    def test_set_trainable_layers_single_pattern(self):
        """Test unfreezing layers with a single pattern."""
        model = ResNetClassifier(num_classes=2, freeze_backbone=True, pretrained=False)

        # Initially, only fc should be trainable
        assert model.fc.weight.requires_grad
        assert not any(p.requires_grad for p in model.layer4.parameters())

        # Unfreeze layer4*
        model.set_trainable_layers(["layer4*"])

        # Now layer4 should be trainable
        assert any(p.requires_grad for p in model.layer4.parameters())

        # But layer3 should still be frozen
        assert not any(p.requires_grad for p in model.layer3.parameters())

    def test_set_trainable_layers_multiple_patterns(self):
        """Test unfreezing layers with multiple patterns."""
        model = ResNetClassifier(num_classes=2, freeze_backbone=True, pretrained=False)

        # Unfreeze both layer3 and layer4
        model.set_trainable_layers(["layer3*", "layer4*"])

        # Both groups should be trainable
        assert any(p.requires_grad for p in model.layer3.parameters())
        assert any(p.requires_grad for p in model.layer4.parameters())

        # But earlier layers should still be frozen
        assert not any(p.requires_grad for p in model.layer1.parameters())
        assert not any(p.requires_grad for p in model.conv1.parameters())

    def test_set_trainable_layers_specific_layers(self):
        """Test unfreezing specific named layers without wildcards."""
        model = ResNetClassifier(num_classes=2, freeze_backbone=True, pretrained=False)

        # Unfreeze only specific layers
        model.set_trainable_layers(["layer3", "layer4"])

        # Only specified layers should be trainable
        assert any(p.requires_grad for p in model.layer3.parameters())
        assert any(p.requires_grad for p in model.layer4.parameters())

        # Earlier layers should still be frozen
        assert not any(p.requires_grad for p in model.layer2.parameters())

    def test_trainable_layers_param_in_init(self):
        """Test that trainable_layers parameter works in __init__."""
        model = ResNetClassifier(
            num_classes=2,
            freeze_backbone=True,
            trainable_layers=["layer4*"],
            pretrained=False,
        )

        # layer4 should be trainable from initialization
        assert any(p.requires_grad for p in model.layer4.parameters())

        # Earlier layers should be frozen
        assert not any(p.requires_grad for p in model.layer3.parameters())

        # Classification head should remain trainable
        assert model.fc.weight.requires_grad

    def test_trainable_layers_overrides_freeze_backbone(self):
        """Test that trainable_layers properly extends freeze_backbone."""
        # freeze_backbone=True with trainable_layers should unfreeze specified layers
        model = ResNetClassifier(
            num_classes=2,
            freeze_backbone=True,
            trainable_layers=["layer3*"],
            pretrained=False,
        )

        # layer3 should be trainable
        assert any(p.requires_grad for p in model.layer3.parameters())

        # But layer2 should still be frozen
        assert not any(p.requires_grad for p in model.layer2.parameters())

    def test_set_trainable_layers_empty_list(self):
        """Test that empty trainable_layers list doesn't change anything."""
        model = ResNetClassifier(num_classes=2, freeze_backbone=True, pretrained=False)

        # Record initial state
        initial_fc_trainable = model.fc.weight.requires_grad
        initial_layer4_trainable = any(
            p.requires_grad for p in model.layer4.parameters()
        )

        # Apply empty list
        model.set_trainable_layers([])

        # State should remain unchanged
        assert model.fc.weight.requires_grad == initial_fc_trainable
        assert (
            any(p.requires_grad for p in model.layer4.parameters())
        ) == initial_layer4_trainable


@pytest.mark.unit
class TestResNetModelDir:
    """Test model directory and weight caching."""

    def test_model_dir_created_if_not_exists(self):
        """Test that model directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "nonexistent" / "models"
            assert not model_dir.exists()

            ResNetClassifier(num_classes=2, pretrained=False, model_dir=str(model_dir))

            assert model_dir.exists()
            assert model_dir.is_dir()

    def test_model_caching_not_pretrained(self):
        """Test that no cache file is created when pretrained=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ResNetClassifier(
                num_classes=2, variant="resnet50", pretrained=False, model_dir=tmpdir
            )

            cache_file = Path(tmpdir) / "resnet50.pth"
            # Cache should not be created when pretrained=False
            assert not cache_file.exists()


# ============================================================================
# Component Tests - Small data, minimal computation on CPU
# ============================================================================


@pytest.mark.component
class TestResNetForward:
    """Test forward pass with small tensors on CPU."""

    def test_forward_pass_output_shape_resnet50(self):
        """Test forward pass returns correct output shape for ResNet50."""
        batch_size = 2
        num_classes = 3
        model = ResNetClassifier(
            num_classes=num_classes, variant="resnet50", pretrained=False
        )
        model.eval()

        # ResNet accepts 224x224 images (but works with other sizes)
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes)

    def test_forward_pass_output_shape_resnet101(self):
        """Test forward pass returns correct output shape for ResNet101."""
        batch_size = 2
        num_classes = 5
        model = ResNetClassifier(
            num_classes=num_classes, variant="resnet101", pretrained=False
        )
        model.eval()

        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes)

    def test_forward_pass_different_input_sizes(self):
        """Test forward pass works with different input sizes."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model.eval()

        # ResNet is fully convolutional and can handle various sizes
        for size in [64, 128, 224]:
            x = torch.randn(2, 3, size, size)

            with torch.no_grad():
                output = model(x)

            assert output.shape == (2, 2)

    def test_forward_pass_output_range(self):
        """Test forward pass output is logits (not probabilities)."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        # Logits can be any real number (not constrained to [0, 1])
        assert output.dtype == torch.float32
        # Just verify output exists and is not NaN
        assert not torch.isnan(output).any()

    def test_forward_pass_batch_size_one(self):
        """Test forward pass with batch size 1."""
        model = ResNetClassifier(num_classes=5, variant="resnet50", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 5)

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass works with different batch sizes."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)

            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, 2)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        model = ResNetClassifier(
            num_classes=2, variant="resnet50", freeze_backbone=False, pretrained=False
        )
        model.train()

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for input
        assert x.grad is not None
        # Check that gradients exist for trainable parameters
        assert model.fc.weight.grad is not None

    def test_forward_pass_with_dropout(self):
        """Test forward pass with dropout enabled."""
        model = ResNetClassifier(
            num_classes=2, variant="resnet50", dropout=0.5, pretrained=False
        )
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2)
        assert not torch.isnan(output).any()


@pytest.mark.component
class TestResNetLoss:
    """Test loss computation."""

    def test_compute_loss_shape(self):
        """Test compute_loss returns scalar by default."""
        model = ResNetClassifier(num_classes=3, variant="resnet50", pretrained=False)

        predictions = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        loss = model.compute_loss(predictions, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.dtype == torch.float32

    def test_compute_loss_positive(self):
        """Test that loss is positive."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)

        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(predictions, targets)

        assert loss.item() >= 0

    def test_compute_loss_reduction_mean(self):
        """Test loss with mean reduction."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)

        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(predictions, targets, reduction="mean")

        assert loss.dim() == 0  # Scalar

    def test_compute_loss_reduction_sum(self):
        """Test loss with sum reduction."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)

        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(predictions, targets, reduction="sum")

        assert loss.dim() == 0  # Scalar

    def test_compute_loss_reduction_none(self):
        """Test loss with no reduction returns per-sample losses."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)

        batch_size = 4
        predictions = torch.randn(batch_size, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = model.compute_loss(predictions, targets, reduction="none")

        assert loss.shape == (batch_size,)
        assert (loss >= 0).all()

    def test_compute_loss_binary_classification(self):
        """Test loss computation for binary classification."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)

        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 1, 0])

        loss = model.compute_loss(predictions, targets)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_compute_loss_multiclass_classification(self):
        """Test loss computation for multi-class classification."""
        model = ResNetClassifier(num_classes=10, variant="resnet50", pretrained=False)

        predictions = torch.randn(4, 10)
        targets = torch.tensor([0, 5, 9, 3])

        loss = model.compute_loss(predictions, targets)

        assert loss.item() >= 0
        assert not torch.isnan(loss)


@pytest.mark.component
class TestResNetFeatureExtraction:
    """Test feature extraction functionality."""

    def test_extract_features_output_shape(self):
        """Test extract_features returns correct shape."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            features = model.extract_features(x)

        # ResNet features are 2048-dimensional
        assert features.shape == (batch_size, 2048)

    def test_extract_features_different_variants(self):
        """Test feature extraction for different ResNet variants."""
        for variant in ["resnet50", "resnet101", "resnet152"]:
            model = ResNetClassifier(num_classes=2, variant=variant, pretrained=False)
            model.eval()

            x = torch.randn(2, 3, 224, 224)

            with torch.no_grad():
                features = model.extract_features(x)

            # All ResNet variants use 2048-dimensional features
            assert features.shape == (2, 2048)

    def test_extract_features_different_from_forward(self):
        """Test that extract_features output differs from forward (no classification)."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            features = model.extract_features(x)
            logits = model(x)

        # Features should be 2048-dimensional, logits should be num_classes
        assert features.shape[1] == 2048
        assert logits.shape[1] == 2
        assert features.shape[1] != logits.shape[1]

    def test_extract_features_batch_size_one(self):
        """Test feature extraction with batch size 1."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            features = model.extract_features(x)

        assert features.shape == (1, 2048)

    def test_extract_features_different_input_sizes(self):
        """Test feature extraction with different input sizes."""
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model.eval()

        for size in [64, 128, 224]:
            x = torch.randn(2, 3, size, size)

            with torch.no_grad():
                features = model.extract_features(x)

            # Output should always be 2048-dimensional due to global avg pooling
            assert features.shape == (2, 2048)


@pytest.mark.component
class TestResNetCheckpointing:
    """Test model checkpoint save/load functionality."""

    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"

            # Create and save model
            model1 = ResNetClassifier(
                num_classes=5, variant="resnet50", pretrained=False
            )
            model1.save_checkpoint(str(checkpoint_path))

            assert checkpoint_path.exists()

            # Create new model and load checkpoint
            model2 = ResNetClassifier(
                num_classes=5, variant="resnet50", pretrained=False
            )
            model2.load_checkpoint(str(checkpoint_path))

            # Verify weights are the same
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)

    def test_save_and_load_checkpoint_preserves_trainability(self):
        """Test that checkpoint preserves parameter trainability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"

            # Create model with frozen backbone
            model1 = ResNetClassifier(
                num_classes=2,
                variant="resnet50",
                freeze_backbone=True,
                pretrained=False,
            )
            model1.save_checkpoint(str(checkpoint_path))

            # Load into new model
            model2 = ResNetClassifier(
                num_classes=2,
                variant="resnet50",
                freeze_backbone=True,
                pretrained=False,
            )
            model2.load_checkpoint(str(checkpoint_path))

            # Verify trainability is preserved
            assert model2.fc.weight.requires_grad
            assert not any(p.requires_grad for p in model2.layer1.parameters())
