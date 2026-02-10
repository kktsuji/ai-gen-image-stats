"""Tests for Base Model Interface

This module contains tests for the BaseModel abstract class and its interface.
Tests are organized into unit tests to ensure fast execution on CPU.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel

# Test fixtures and helper classes


class MinimalValidModel(BaseModel):
    """Minimal valid model implementation for testing.

    This is the simplest possible implementation that satisfies
    the BaseModel interface requirements.
    """

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(predictions, targets)


class MultiLossModel(BaseModel):
    """Model that returns multiple loss components."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        ce_loss = F.cross_entropy(predictions, targets)
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters())

        return {
            "total": ce_loss + 0.01 * l2_loss,
            "cross_entropy": ce_loss,
            "l2_regularization": l2_loss,
        }


class IncompleteModel(BaseModel):
    """Model that doesn't implement required abstract methods.

    This should fail to instantiate.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    # Missing forward() and compute_loss() implementations


# Unit Tests


@pytest.mark.unit
class TestBaseModelInterface:
    """Test that BaseModel enforces its interface requirements."""

    def test_cannot_instantiate_base_model_directly(self):
        """BaseModel is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseModel()

    def test_cannot_instantiate_incomplete_implementation(self):
        """Models that don't implement all abstract methods cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel()

    def test_can_instantiate_complete_implementation(self):
        """Models that implement all abstract methods can be instantiated."""
        model = MinimalValidModel()
        assert isinstance(model, BaseModel)
        assert isinstance(model, nn.Module)

    def test_forward_is_abstract(self):
        """forward() method must be implemented by subclasses."""
        # This is implicitly tested by test_cannot_instantiate_incomplete_implementation
        # But we verify the method exists in the base class
        assert hasattr(BaseModel, "forward")
        assert getattr(BaseModel.forward, "__isabstractmethod__", False)

    def test_compute_loss_is_abstract(self):
        """compute_loss() method must be implemented by subclasses."""
        assert hasattr(BaseModel, "compute_loss")
        assert getattr(BaseModel.compute_loss, "__isabstractmethod__", False)


@pytest.mark.unit
class TestModelForwardPass:
    """Test model forward pass functionality."""

    def test_forward_pass_simple(self):
        """Test basic forward pass with minimal model."""
        model = MinimalValidModel(input_dim=10, output_dim=2)
        x = torch.randn(4, 10)  # Batch of 4 samples

        output = model(x)

        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        model = MinimalValidModel(input_dim=5, output_dim=3)
        x = torch.randn(1, 5)

        output = model(x)

        assert output.shape == (1, 3)

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with various batch sizes."""
        model = MinimalValidModel(input_dim=8, output_dim=4)

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 8)
            output = model(x)
            assert output.shape == (batch_size, 4)


@pytest.mark.unit
class TestModelLossComputation:
    """Test loss computation functionality."""

    def test_compute_loss_single_value(self):
        """Test loss computation returning single tensor."""
        model = MinimalValidModel()
        predictions = torch.randn(4, 2, requires_grad=True)
        targets = torch.randint(0, 2, (4,))

        loss = model.compute_loss(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar tensor
        assert loss.requires_grad
        assert not torch.isnan(loss)

    def test_compute_loss_multiple_components(self):
        """Test loss computation returning multiple components."""
        model = MultiLossModel()
        predictions = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))

        loss_dict = model.compute_loss(predictions, targets)

        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "cross_entropy" in loss_dict
        assert "l2_regularization" in loss_dict

        # Verify all losses are valid tensors
        for key, value in loss_dict.items():
            assert isinstance(value, torch.Tensor)
            assert not torch.isnan(value)

        # Verify total is sum of components (approximately)
        expected_total = (
            loss_dict["cross_entropy"] + 0.01 * loss_dict["l2_regularization"]
        )
        torch.testing.assert_close(
            loss_dict["total"], expected_total, rtol=1e-5, atol=1e-5
        )


@pytest.mark.unit
class TestCheckpointSaveLoad:
    """Test checkpoint save and load functionality."""

    def test_save_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint saving."""
        model = MinimalValidModel()
        checkpoint_path = tmp_path / "model.pth"

        model.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()
        checkpoint = torch.load(checkpoint_path)
        assert "model_state_dict" in checkpoint
        assert "model_class" in checkpoint

    def test_save_checkpoint_with_metadata(self, tmp_path):
        """Test saving checkpoint with epoch and metrics."""
        model = MinimalValidModel()
        checkpoint_path = tmp_path / "model_epoch10.pth"

        metrics = {"accuracy": 0.95, "loss": 0.123}
        model.save_checkpoint(checkpoint_path, epoch=10, metrics=metrics)

        checkpoint = torch.load(checkpoint_path)
        assert checkpoint["epoch"] == 10
        assert checkpoint["metrics"] == metrics

    def test_save_checkpoint_with_optimizer(self, tmp_path):
        """Test saving checkpoint with optimizer state."""
        model = MinimalValidModel()
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint_path = tmp_path / "model_with_opt.pth"

        model.save_checkpoint(checkpoint_path, optimizer_state=optimizer.state_dict())

        checkpoint = torch.load(checkpoint_path)
        assert "optimizer_state_dict" in checkpoint

    def test_save_checkpoint_creates_directory(self, tmp_path):
        """Test that save_checkpoint creates parent directories."""
        model = MinimalValidModel()
        nested_path = tmp_path / "checkpoints" / "exp1" / "model.pth"

        model.save_checkpoint(nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_checkpoint_with_custom_metadata(self, tmp_path):
        """Test saving checkpoint with custom metadata."""
        model = MinimalValidModel()
        checkpoint_path = tmp_path / "model.pth"

        model.save_checkpoint(
            checkpoint_path, custom_field="custom_value", experiment_id=12345
        )

        checkpoint = torch.load(checkpoint_path)
        assert checkpoint["custom_field"] == "custom_value"
        assert checkpoint["experiment_id"] == 12345

    def test_load_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint loading."""
        # Create and save model
        model1 = MinimalValidModel()
        checkpoint_path = tmp_path / "model.pth"
        model1.save_checkpoint(checkpoint_path)

        # Load into new model
        model2 = MinimalValidModel()
        metadata = model2.load_checkpoint(checkpoint_path)

        # Verify weights are the same
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_load_checkpoint_with_metadata(self, tmp_path):
        """Test loading checkpoint and retrieving metadata."""
        model1 = MinimalValidModel()
        checkpoint_path = tmp_path / "model.pth"

        # Save with metadata
        model1.save_checkpoint(checkpoint_path, epoch=42, metrics={"val_loss": 0.5})

        # Load and check metadata
        model2 = MinimalValidModel()
        metadata = model2.load_checkpoint(checkpoint_path)

        assert metadata["epoch"] == 42
        assert metadata["metrics"]["val_loss"] == 0.5
        assert "model_state_dict" not in metadata  # Should be excluded

    def test_load_checkpoint_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        model = MinimalValidModel()

        with pytest.raises(FileNotFoundError):
            model.load_checkpoint("nonexistent.pth")

    def test_load_checkpoint_with_device(self, tmp_path):
        """Test loading checkpoint to specific device."""
        model1 = MinimalValidModel()
        checkpoint_path = tmp_path / "model.pth"
        model1.save_checkpoint(checkpoint_path)

        model2 = MinimalValidModel()
        model2.load_checkpoint(checkpoint_path, device="cpu")

        # Verify model is on CPU
        assert model2.get_device().type == "cpu"

    def test_save_load_roundtrip_preserves_weights(self, tmp_path):
        """Test that save/load roundtrip preserves exact weights."""
        model1 = MinimalValidModel()
        checkpoint_path = tmp_path / "model.pth"

        # Get initial weights
        initial_weights = {
            name: param.clone() for name, param in model1.named_parameters()
        }

        # Save and load
        model1.save_checkpoint(checkpoint_path)
        model1.load_checkpoint(checkpoint_path)

        # Verify weights unchanged
        for name, param in model1.named_parameters():
            torch.testing.assert_close(param, initial_weights[name])

    def test_load_checkpoint_strict_mode(self, tmp_path):
        """Test strict mode when loading checkpoints with missing/extra keys.

        Note: PyTorch's strict parameter only controls whether missing/unexpected
        keys are allowed, not shape mismatches. Shape mismatches always raise errors.
        """
        # Create a model and save it
        model1 = MinimalValidModel(input_dim=10, output_dim=2)
        checkpoint_path = tmp_path / "model.pth"
        model1.save_checkpoint(checkpoint_path)

        # Manually modify checkpoint to have extra key (simulating unexpected key)
        checkpoint = torch.load(checkpoint_path)
        checkpoint["model_state_dict"]["extra_param"] = torch.randn(5, 5)
        torch.save(checkpoint, checkpoint_path)

        # Try to load with extra key
        model2 = MinimalValidModel(input_dim=10, output_dim=2)

        # Should raise error in strict mode (unexpected key)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            model2.load_checkpoint(checkpoint_path, strict=True)

        # Should succeed in non-strict mode (extra keys ignored)
        model3 = MinimalValidModel(input_dim=10, output_dim=2)
        model3.load_checkpoint(checkpoint_path, strict=False)


@pytest.mark.unit
class TestParameterCounting:
    """Test parameter counting functionality."""

    def test_count_parameters_trainable_only(self):
        """Test counting only trainable parameters."""
        model = MinimalValidModel(input_dim=10, output_dim=5)

        # Linear layer has weight (10*5) + bias (5) = 55 parameters
        expected = 10 * 5 + 5
        assert model.count_parameters(trainable_only=True) == expected

    def test_count_parameters_all(self):
        """Test counting all parameters."""
        model = MinimalValidModel(input_dim=10, output_dim=5)

        trainable = model.count_parameters(trainable_only=True)
        total = model.count_parameters(trainable_only=False)

        # Initially, all parameters are trainable
        assert trainable == total

    def test_count_parameters_after_freeze(self):
        """Test parameter counting after freezing some parameters."""
        model = MinimalValidModel(input_dim=10, output_dim=5)

        initial_count = model.count_parameters(trainable_only=True)

        # Freeze the model
        model.freeze()

        frozen_count = model.count_parameters(trainable_only=True)
        total_count = model.count_parameters(trainable_only=False)

        assert frozen_count == 0
        assert total_count == initial_count


@pytest.mark.unit
class TestFreezeUnfreeze:
    """Test freeze and unfreeze functionality."""

    def test_freeze_disables_gradients(self):
        """Test that freeze() disables gradient computation."""
        model = MinimalValidModel()

        model.freeze()

        for param in model.parameters():
            assert not param.requires_grad

    def test_unfreeze_enables_gradients(self):
        """Test that unfreeze() enables gradient computation."""
        model = MinimalValidModel()

        model.freeze()
        model.unfreeze()

        for param in model.parameters():
            assert param.requires_grad

    def test_freeze_prevents_gradient_computation(self):
        """Test that frozen model doesn't compute gradients."""
        model = MinimalValidModel()
        model.freeze()

        x = torch.randn(4, 10, requires_grad=True)
        targets = torch.randint(0, 2, (4,))

        output = model(x)
        loss = model.compute_loss(output, targets)

        # Loss should still be computable but won't have grad_fn for model params
        # (it may have grad_fn for the input x if we traced through)
        assert isinstance(loss, torch.Tensor)

        # Try backward - should work but not update frozen parameters
        if loss.requires_grad:
            loss.backward()

        # Gradients should be None for all model parameters (they're frozen)
        for param in model.parameters():
            assert param.grad is None

    def test_partial_freeze_for_transfer_learning(self):
        """Test freezing part of model for transfer learning."""
        model = MinimalValidModel()

        # Freeze initial layers
        model.freeze()

        # Unfreeze specific layer for fine-tuning
        for param in model.fc.parameters():
            param.requires_grad = True

        # Only fc layer should have gradients
        assert any(p.requires_grad for p in model.fc.parameters())


@pytest.mark.unit
class TestDeviceDetection:
    """Test device detection functionality."""

    def test_get_device_cpu(self):
        """Test getting device for CPU model."""
        model = MinimalValidModel()

        device = model.get_device()

        assert isinstance(device, torch.device)
        assert device.type == "cpu"

    def test_get_device_after_to(self):
        """Test getting device after moving model."""
        model = MinimalValidModel()

        # Move to CPU explicitly
        model = model.to("cpu")
        device = model.get_device()

        assert device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_cuda(self):
        """Test getting device for CUDA model."""
        model = MinimalValidModel()
        model = model.to("cuda")

        device = model.get_device()

        assert device.type == "cuda"


@pytest.mark.unit
class TestModelInheritance:
    """Test that BaseModel properly inherits from nn.Module."""

    def test_inherits_from_nn_module(self):
        """Test that models inherit from nn.Module."""
        model = MinimalValidModel()

        assert isinstance(model, nn.Module)

    def test_has_nn_module_methods(self):
        """Test that models have standard nn.Module methods."""
        model = MinimalValidModel()

        # Should have standard nn.Module methods
        assert hasattr(model, "parameters")
        assert hasattr(model, "state_dict")
        assert hasattr(model, "load_state_dict")
        assert hasattr(model, "train")
        assert hasattr(model, "eval")

    def test_train_eval_modes(self):
        """Test switching between train and eval modes."""
        model = MinimalValidModel()

        # Default is train mode
        assert model.training

        model.eval()
        assert not model.training

        model.train()
        assert model.training

    def test_can_register_buffers(self):
        """Test that models can register buffers (nn.Module functionality)."""
        model = MinimalValidModel()

        # Register a buffer
        model.register_buffer("test_buffer", torch.ones(5))

        assert hasattr(model, "test_buffer")
        assert "test_buffer" in dict(model.named_buffers())

    def test_parameters_iterator_works(self):
        """Test that parameter iteration works."""
        model = MinimalValidModel(input_dim=10, output_dim=5)

        params = list(model.parameters())

        # Should have weight and bias
        assert len(params) == 2
        assert params[0].shape == (5, 10)  # weight
        assert params[1].shape == (5,)  # bias


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model_no_parameters(self):
        """Test model with no parameters."""

        class EmptyModel(BaseModel):
            def forward(self, x):
                return x

            def compute_loss(self, x, y):
                return torch.tensor(0.0)

        model = EmptyModel()
        assert model.count_parameters() == 0

    def test_save_checkpoint_with_pathlib(self, tmp_path):
        """Test that save_checkpoint accepts pathlib.Path."""
        model = MinimalValidModel()
        checkpoint_path = Path(tmp_path) / "model.pth"

        model.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

    def test_load_checkpoint_with_pathlib(self, tmp_path):
        """Test that load_checkpoint accepts pathlib.Path."""
        model1 = MinimalValidModel()
        checkpoint_path = Path(tmp_path) / "model.pth"
        model1.save_checkpoint(checkpoint_path)

        model2 = MinimalValidModel()
        model2.load_checkpoint(checkpoint_path)

        # Should succeed without errors

    def test_model_works_with_no_bias(self):
        """Test model with no bias parameters."""

        class NoBiasModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5, bias=False)

            def forward(self, x):
                return self.fc(x)

            def compute_loss(self, predictions, targets):
                return F.cross_entropy(predictions, targets)

        model = NoBiasModel()
        # Should have only weight parameters (10 * 5 = 50)
        assert model.count_parameters() == 50
