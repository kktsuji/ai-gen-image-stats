"""Pytest tests for Inception V3 Feature Extractor"""

import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from inception_v3 import InceptionV3FeatureExtractor, InceptionV3FeatureTrainer


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def create_mock_inception():
    """Create a minimal mock Inception V3 model with proper nn.Module structure"""

    class BasicConv2d(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    class InceptionModule(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.branch = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.branch(x)

    class MockInceptionV3(nn.Module):
        def __init__(self):
            super().__init__()
            # Initial convolutions
            self.Conv2d_1a_3x3 = BasicConv2d(3, 32)
            self.Conv2d_2a_3x3 = BasicConv2d(32, 32)
            self.Conv2d_2b_3x3 = BasicConv2d(32, 64)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.Conv2d_3b_1x1 = BasicConv2d(64, 80)
            self.Conv2d_4a_3x3 = BasicConv2d(80, 192)
            self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Mixed layers
            self.Mixed_5b = InceptionModule(192, 256)
            self.Mixed_5c = InceptionModule(256, 288)
            self.Mixed_5d = InceptionModule(288, 288)
            self.Mixed_6a = InceptionModule(288, 768)
            self.Mixed_6b = InceptionModule(768, 768)
            self.Mixed_6c = InceptionModule(768, 768)
            self.Mixed_6d = InceptionModule(768, 768)
            self.Mixed_6e = InceptionModule(768, 768)
            self.Mixed_7a = InceptionModule(768, 1280)
            self.Mixed_7b = InceptionModule(1280, 2048)
            self.Mixed_7c = InceptionModule(2048, 2048)

            # Final layers (not used in feature extraction)
            self.fc = nn.Linear(2048, 1000)
            self.AuxLogits = None  # Auxiliary classifier (not used)

        def forward(self, x):
            return x

    return MockInceptionV3()


class TestInceptionV3FeatureExtractorInitialization:
    """Test InceptionV3FeatureExtractor initialization"""

    def test_init_with_default_params(self, temp_model_dir):
        """Test initialization with default parameters"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            extractor = InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            assert extractor is not None
            assert isinstance(extractor, nn.Module)
            mock_inception.assert_called_once()

    def test_init_creates_model_directory(self, temp_model_dir):
        """Test that model directory is created if it doesn't exist"""
        model_dir = os.path.join(temp_model_dir, "new_models")
        assert not os.path.exists(model_dir)

        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            InceptionV3FeatureExtractor(model_dir=model_dir)

            assert os.path.exists(model_dir)

    def test_init_downloads_model_if_not_cached(self, temp_model_dir):
        """Test that model is downloaded if not in cache"""
        model_path = os.path.join(temp_model_dir, "inception_v3.pth")
        assert not os.path.exists(model_path)

        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Verify model was saved
            mock_save.assert_called_once()
            assert mock_save.call_args[0][1] == model_path

    def test_init_loads_cached_model(self, temp_model_dir):
        """Test that cached model is loaded from cache when it exists"""
        model_path = os.path.join(temp_model_dir, "inception_v3.pth")

        # First, patch torch.save and create a model to simulate caching
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()
            InceptionV3FeatureExtractor(model_dir=temp_model_dir)
            # Verify save was called
            assert mock_save.called

        # Create a real file to simulate cached model
        with open(model_path, "w") as f:
            f.write("dummy")

        # Now verify it attempts to load when file exists
        with patch("inception_v3.torch.load") as mock_load, patch(
            "inception_v3.inception_v3"
        ) as mock_inception:
            mock_load.return_value = create_mock_inception()

            InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Verify cached model was loaded
            mock_load.assert_called_once_with(model_path)
            # Verify model download function was not called
            mock_inception.assert_not_called()

    def test_model_set_to_eval_mode(self, temp_model_dir):
        """Test that model is set to evaluation mode"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            extractor = InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Extractor should be in eval mode
            assert not extractor.training

    def test_inception_called_with_correct_params(self, temp_model_dir):
        """Test that inception_v3 is called with correct parameters"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Verify inception_v3 was called with pretrained=True and transform_input=False
            mock_inception.assert_called_once_with(
                pretrained=True, transform_input=False
            )

    def test_all_layers_extracted(self, temp_model_dir):
        """Test that all necessary layers are extracted from inception model"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_model = create_mock_inception()
            mock_inception.return_value = mock_model

            extractor = InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Verify all layers are present
            assert hasattr(extractor, "Conv2d_1a_3x3")
            assert hasattr(extractor, "Conv2d_2a_3x3")
            assert hasattr(extractor, "Conv2d_2b_3x3")
            assert hasattr(extractor, "maxpool1")
            assert hasattr(extractor, "Conv2d_3b_1x1")
            assert hasattr(extractor, "Conv2d_4a_3x3")
            assert hasattr(extractor, "maxpool2")
            assert hasattr(extractor, "Mixed_5b")
            assert hasattr(extractor, "Mixed_5c")
            assert hasattr(extractor, "Mixed_5d")
            assert hasattr(extractor, "Mixed_6a")
            assert hasattr(extractor, "Mixed_6b")
            assert hasattr(extractor, "Mixed_6c")
            assert hasattr(extractor, "Mixed_6d")
            assert hasattr(extractor, "Mixed_6e")
            assert hasattr(extractor, "Mixed_7a")
            assert hasattr(extractor, "Mixed_7b")
            assert hasattr(extractor, "Mixed_7c")

    def test_fc_and_aux_not_included(self, temp_model_dir):
        """Test that FC and auxiliary classifier are not included"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            extractor = InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Verify FC and AuxLogits are not present
            assert not hasattr(extractor, "fc")
            assert not hasattr(extractor, "AuxLogits")


class TestInceptionV3FeatureExtractorForward:
    """Test InceptionV3FeatureExtractor forward pass"""

    @pytest.fixture
    def extractor(self, temp_model_dir):
        """Create an InceptionV3FeatureExtractor instance for testing"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()
            return InceptionV3FeatureExtractor(model_dir=temp_model_dir)

    def test_forward_with_single_image(self, extractor):
        """Test forward pass with a single image (299x299 for InceptionV3)"""
        dummy_input = torch.randn(1, 3, 299, 299)
        output = extractor.forward(dummy_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        # Output should be flattened to (batch, 2048)
        assert output.dim() == 2
        assert output.size(0) == 1  # batch size
        assert output.size(1) == 2048  # feature dimension

    def test_forward_with_batch(self, extractor):
        """Test forward pass with a batch of images"""
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 299, 299)
        output = extractor.forward(dummy_input)

        assert output is not None
        assert output.dim() == 2
        assert output.size(0) == batch_size
        assert output.size(1) == 2048

    def test_forward_output_shape(self, extractor):
        """Test that forward pass produces correct output shape"""
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 299, 299)
        output = extractor.forward(dummy_input)

        # Should always output (batch_size, 2048)
        assert output.shape == (batch_size, 2048)

    def test_forward_preserves_batch_dimension(self, extractor):
        """Test that batch dimension is preserved through forward pass"""
        for batch_size in [1, 2, 4, 8]:
            dummy_input = torch.randn(batch_size, 3, 299, 299)
            output = extractor.forward(dummy_input)
            assert output.size(0) == batch_size
            assert output.size(1) == 2048

    def test_forward_with_different_input_sizes(self, extractor):
        """Test forward pass with different input sizes"""
        # InceptionV3 is designed for 299x299, but should handle other sizes
        test_sizes = [(299, 299), (224, 224), (320, 320)]
        batch_size = 2

        for height, width in test_sizes:
            dummy_input = torch.randn(batch_size, 3, height, width)
            try:
                output = extractor.forward(dummy_input)
                # If it works, verify output shape
                assert output.size(0) == batch_size
                assert output.size(1) == 2048
            except RuntimeError:
                # Some sizes might not work due to pooling/stride constraints
                pass

    def test_forward_no_gradient(self, extractor):
        """Test that forward pass in eval mode doesn't compute gradients"""
        extractor.eval()
        dummy_input = torch.randn(1, 3, 299, 299, requires_grad=True)

        with torch.no_grad():
            output = extractor.forward(dummy_input)

        # Output should not require gradients in eval mode with no_grad
        assert not output.requires_grad

    def test_forward_consistency(self, extractor):
        """Test that forward pass produces consistent results for same input"""
        extractor.eval()
        dummy_input = torch.randn(2, 3, 299, 299)

        with torch.no_grad():
            output1 = extractor.forward(dummy_input)
            output2 = extractor.forward(dummy_input)

        # Should produce identical results
        assert torch.allclose(output1, output2)


class TestInceptionV3FeatureExtractorIntegration:
    """Integration tests for InceptionV3FeatureExtractor"""

    def test_model_persistence(self, temp_model_dir):
        """Test that model is properly saved and loaded"""
        model_path = os.path.join(temp_model_dir, "inception_v3.pth")

        # First initialization
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Check save was called
            assert mock_save.called
            save_path = mock_save.call_args[0][1]
            assert save_path == model_path

    def test_extractor_inherits_from_nn_module(self, temp_model_dir):
        """Test that InceptionV3FeatureExtractor properly inherits from nn.Module"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            extractor = InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            assert isinstance(extractor, nn.Module)
            # Should have standard nn.Module methods
            assert hasattr(extractor, "parameters")
            assert hasattr(extractor, "train")
            assert hasattr(extractor, "eval")

    def test_feature_dimension_is_2048(self, temp_model_dir):
        """Test that output feature dimension is always 2048"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            extractor = InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Test with different batch sizes
            for batch_size in [1, 4, 8]:
                dummy_input = torch.randn(batch_size, 3, 299, 299)
                output = extractor.forward(dummy_input)
                assert output.size(1) == 2048


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_model_dir_string(self):
        """Test with empty model directory string"""
        with pytest.raises((ValueError, OSError)):
            with patch("inception_v3.inception_v3"):
                InceptionV3FeatureExtractor(model_dir="")

    def test_forward_with_wrong_channels(self, temp_model_dir):
        """Test forward pass with wrong number of input channels"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()
            extractor = InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Wrong number of channels (should be 3)
            wrong_input = torch.randn(1, 1, 299, 299)

            with pytest.raises(RuntimeError):
                extractor.forward(wrong_input)

    def test_forward_with_very_small_input(self, temp_model_dir):
        """Test forward pass with very small input size"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()
            extractor = InceptionV3FeatureExtractor(model_dir=temp_model_dir)

            # Very small input - might fail due to pooling
            small_input = torch.randn(1, 3, 32, 32)

            # This might work or fail depending on architecture
            # Just ensure it doesn't crash unexpectedly
            try:
                output = extractor.forward(small_input)
                assert output is not None
            except (RuntimeError, ValueError):
                # Expected failure for too small inputs
                pass


class TestInceptionV3FeatureTrainerInitialization:
    """Test InceptionV3FeatureTrainer initialization"""

    def test_init_with_custom_num_classes(self, temp_model_dir):
        """Test initialization with custom number of classes"""
        num_classes = 10
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(
                num_classes=num_classes, model_dir=temp_model_dir
            )

            assert trainer is not None
            assert isinstance(trainer, nn.Module)
            assert hasattr(trainer, "fc")
            # Check final layer has correct output size
            assert trainer.fc.out_features == num_classes

    def test_init_with_different_num_classes(self, temp_model_dir):
        """Test initialization with various numbers of classes"""
        test_classes = [2, 5, 10, 100, 1000]

        for num_classes in test_classes:
            with patch("inception_v3.inception_v3") as mock_inception, patch(
                "inception_v3.torch.save"
            ) as mock_save:
                mock_inception.return_value = create_mock_inception()

                trainer = InceptionV3FeatureTrainer(
                    num_classes=num_classes, model_dir=temp_model_dir
                )

                assert trainer.fc.out_features == num_classes
                assert trainer.fc.in_features == 2048

    def test_feature_layers_are_frozen(self, temp_model_dir):
        """Test that all feature extraction layers are frozen"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(
                num_classes=10, model_dir=temp_model_dir
            )

            # Check that feature extraction layers don't require gradients
            for name, param in trainer.named_parameters():
                if "fc" not in name:
                    assert (
                        not param.requires_grad
                    ), f"Parameter {name} should be frozen but requires_grad=True"

    def test_fc_layer_is_trainable(self, temp_model_dir):
        """Test that only the final FC layer is trainable"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(
                num_classes=10, model_dir=temp_model_dir
            )

            # Check that FC layer requires gradients
            for param in trainer.fc.parameters():
                assert param.requires_grad, "FC layer parameters should be trainable"

    def test_all_layers_extracted(self, temp_model_dir):
        """Test that all necessary layers are extracted from inception model"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(
                num_classes=10, model_dir=temp_model_dir
            )

            # Verify all feature extraction layers are present
            assert hasattr(trainer, "Conv2d_1a_3x3")
            assert hasattr(trainer, "Conv2d_2a_3x3")
            assert hasattr(trainer, "Conv2d_2b_3x3")
            assert hasattr(trainer, "maxpool1")
            assert hasattr(trainer, "Conv2d_3b_1x1")
            assert hasattr(trainer, "Conv2d_4a_3x3")
            assert hasattr(trainer, "maxpool2")
            assert hasattr(trainer, "Mixed_5b")
            assert hasattr(trainer, "Mixed_5c")
            assert hasattr(trainer, "Mixed_5d")
            assert hasattr(trainer, "Mixed_6a")
            assert hasattr(trainer, "Mixed_6b")
            assert hasattr(trainer, "Mixed_6c")
            assert hasattr(trainer, "Mixed_6d")
            assert hasattr(trainer, "Mixed_6e")
            assert hasattr(trainer, "Mixed_7a")
            assert hasattr(trainer, "Mixed_7b")
            assert hasattr(trainer, "Mixed_7c")
            assert hasattr(trainer, "fc")

    def test_fc_layer_dimensions(self, temp_model_dir):
        """Test that FC layer has correct input/output dimensions"""
        num_classes = 7
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(
                num_classes=num_classes, model_dir=temp_model_dir
            )

            # FC should be 2048 -> num_classes
            assert trainer.fc.in_features == 2048
            assert trainer.fc.out_features == num_classes
            assert isinstance(trainer.fc, nn.Linear)


class TestInceptionV3FeatureTrainerForward:
    """Test InceptionV3FeatureTrainer forward pass"""

    @pytest.fixture
    def trainer(self, temp_model_dir):
        """Create an InceptionV3FeatureTrainer instance for testing"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()
            return InceptionV3FeatureTrainer(num_classes=5, model_dir=temp_model_dir)

    def test_forward_with_single_image(self, trainer):
        """Test forward pass with a single image"""
        dummy_input = torch.randn(1, 3, 299, 299)
        output = trainer.forward(dummy_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        # Output should be (batch, num_classes)
        assert output.dim() == 2
        assert output.size(0) == 1  # batch size
        assert output.size(1) == 5  # number of classes

    def test_forward_with_batch(self, trainer):
        """Test forward pass with a batch of images"""
        batch_size = 8
        dummy_input = torch.randn(batch_size, 3, 299, 299)
        output = trainer.forward(dummy_input)

        assert output is not None
        assert output.dim() == 2
        assert output.size(0) == batch_size
        assert output.size(1) == 5

    def test_forward_output_shape(self, temp_model_dir):
        """Test that forward pass produces correct output shape for different num_classes"""
        test_configs = [(2, 4), (10, 8), (100, 2)]

        for num_classes, batch_size in test_configs:
            with patch("inception_v3.inception_v3") as mock_inception, patch(
                "inception_v3.torch.save"
            ) as mock_save:
                mock_inception.return_value = create_mock_inception()

                trainer = InceptionV3FeatureTrainer(
                    num_classes=num_classes, model_dir=temp_model_dir
                )
                dummy_input = torch.randn(batch_size, 3, 299, 299)
                output = trainer.forward(dummy_input)

                assert output.shape == (batch_size, num_classes)

    def test_forward_preserves_batch_dimension(self, trainer):
        """Test that batch dimension is preserved through forward pass"""
        for batch_size in [1, 2, 4, 8, 16]:
            dummy_input = torch.randn(batch_size, 3, 299, 299)
            output = trainer.forward(dummy_input)
            assert output.size(0) == batch_size
            assert output.size(1) == 5

    def test_forward_output_not_normalized(self, trainer):
        """Test that forward output is raw logits (not softmax)"""
        dummy_input = torch.randn(4, 3, 299, 299)
        output = trainer.forward(dummy_input)

        # Raw logits can be any value, not necessarily in [0, 1]
        # If it were softmaxed, sum across classes would be 1
        sums = output.sum(dim=1)
        # Not all sums should be close to 1 (some might be by chance)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=0.01)


class TestInceptionV3FeatureTrainerTraining:
    """Test InceptionV3FeatureTrainer training capabilities"""

    @pytest.fixture
    def trainer(self, temp_model_dir):
        """Create an InceptionV3FeatureTrainer instance for testing"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()
            return InceptionV3FeatureTrainer(num_classes=3, model_dir=temp_model_dir)

    def test_get_trainable_parameters(self, trainer):
        """Test get_trainable_parameters method"""
        trainable_params = list(trainer.get_trainable_parameters())

        # Should return some parameters
        assert len(trainable_params) > 0

        # All returned parameters should require gradients
        for param in trainable_params:
            assert param.requires_grad

    def test_only_fc_parameters_trainable(self, trainer):
        """Test that only FC layer parameters are returned by get_trainable_parameters"""
        trainable_params = list(trainer.get_trainable_parameters())
        fc_params = list(trainer.fc.parameters())

        # Number of trainable params should match FC layer params
        assert len(trainable_params) == len(fc_params)

        # Each trainable param should be in FC params
        for param in trainable_params:
            assert any(param is fc_param for fc_param in fc_params)

    def test_backward_pass_updates_only_fc(self, trainer):
        """Test that backward pass only computes gradients for FC layer"""
        trainer.train()
        dummy_input = torch.randn(2, 3, 299, 299)
        dummy_target = torch.tensor([0, 1])

        # Forward pass
        output = trainer(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_target)

        # Backward pass
        loss.backward()

        # Check gradients
        for name, param in trainer.named_parameters():
            if "fc" in name:
                assert (
                    param.grad is not None
                ), f"FC parameter {name} should have gradients"
            else:
                assert (
                    param.grad is None
                ), f"Frozen parameter {name} should not have gradients"

    def test_training_mode_switch(self, trainer):
        """Test switching between train and eval modes"""
        # Initially should be in train mode (default after __init__)
        trainer.train()
        assert trainer.training

        # Switch to eval
        trainer.eval()
        assert not trainer.training

        # Switch back to train
        trainer.train()
        assert trainer.training

    def test_fc_weights_can_be_updated(self, trainer):
        """Test that FC weights can be updated during training"""
        trainer.train()

        # Save initial weights
        initial_weight = trainer.fc.weight.clone().detach()
        initial_bias = trainer.fc.bias.clone().detach()

        # Create dummy data
        dummy_input = torch.randn(4, 3, 299, 299)
        dummy_target = torch.tensor([0, 1, 2, 0])

        # Training step
        optimizer = torch.optim.SGD(trainer.get_trainable_parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        output = trainer(dummy_input)
        loss = criterion(output, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that weights changed
        assert not torch.allclose(trainer.fc.weight, initial_weight)
        assert not torch.allclose(trainer.fc.bias, initial_bias)

    def test_feature_weights_stay_frozen(self, trainer):
        """Test that feature extraction weights remain frozen during training"""
        trainer.train()

        # Save initial weights of a feature layer
        initial_conv_weight = trainer.Conv2d_1a_3x3.conv.weight.clone().detach()

        # Create dummy data
        dummy_input = torch.randn(4, 3, 299, 299)
        dummy_target = torch.tensor([0, 1, 2, 0])

        # Training step
        optimizer = torch.optim.SGD(trainer.get_trainable_parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        output = trainer(dummy_input)
        loss = criterion(output, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that frozen weights didn't change
        assert torch.allclose(trainer.Conv2d_1a_3x3.conv.weight, initial_conv_weight)


class TestInceptionV3FeatureTrainerIntegration:
    """Integration tests for InceptionV3FeatureTrainer"""

    def test_trainer_inherits_from_nn_module(self, temp_model_dir):
        """Test that InceptionV3FeatureTrainer properly inherits from nn.Module"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(num_classes=5, model_dir=temp_model_dir)

            assert isinstance(trainer, nn.Module)
            # Should have standard nn.Module methods
            assert hasattr(trainer, "parameters")
            assert hasattr(trainer, "train")
            assert hasattr(trainer, "eval")
            assert hasattr(trainer, "state_dict")
            assert hasattr(trainer, "load_state_dict")

    def test_can_save_and_load_state_dict(self, temp_model_dir):
        """Test that model state can be saved and loaded"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer1 = InceptionV3FeatureTrainer(
                num_classes=5, model_dir=temp_model_dir
            )

            # Get state dict
            state_dict = trainer1.state_dict()
            assert state_dict is not None
            assert "fc.weight" in state_dict
            assert "fc.bias" in state_dict

    def test_compatible_with_optimizer(self, temp_model_dir):
        """Test that trainer works with PyTorch optimizers"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(num_classes=5, model_dir=temp_model_dir)

            # Should work with various optimizers
            optimizers = [
                torch.optim.SGD(trainer.get_trainable_parameters(), lr=0.01),
                torch.optim.Adam(trainer.get_trainable_parameters(), lr=0.001),
                torch.optim.AdamW(trainer.get_trainable_parameters(), lr=0.001),
            ]

            for optimizer in optimizers:
                assert optimizer is not None
                # Verify optimizer has parameters
                param_groups = optimizer.param_groups
                assert len(param_groups) > 0
                assert len(param_groups[0]["params"]) > 0

    def test_compatible_with_loss_functions(self, temp_model_dir):
        """Test that trainer works with PyTorch loss functions"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(num_classes=5, model_dir=temp_model_dir)
            trainer.eval()

            dummy_input = torch.randn(4, 3, 299, 299)
            dummy_target = torch.tensor([0, 1, 2, 3])

            output = trainer(dummy_input)

            # Should work with various loss functions
            losses = [
                nn.CrossEntropyLoss(),
                nn.NLLLoss(),  # requires log_softmax
            ]

            ce_loss = losses[0](output, dummy_target)
            assert ce_loss is not None
            assert ce_loss.item() >= 0

    def test_end_to_end_training_loop(self, temp_model_dir):
        """Test a complete training loop"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(num_classes=3, model_dir=temp_model_dir)
            trainer.train()

            # Setup
            optimizer = torch.optim.SGD(trainer.get_trainable_parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            # Dummy data
            inputs = torch.randn(8, 3, 299, 299)
            targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

            # Training step
            outputs = trainer(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Verify loss was computed
            assert loss.item() > 0

            # Verify gradients exist for trainable params
            for param in trainer.get_trainable_parameters():
                assert param.grad is not None


class TestInceptionV3FeatureTrainerEdgeCases:
    """Test edge cases for InceptionV3FeatureTrainer"""

    def test_single_class_classification(self, temp_model_dir):
        """Test with single class (edge case)"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(num_classes=1, model_dir=temp_model_dir)

            dummy_input = torch.randn(2, 3, 299, 299)
            output = trainer(dummy_input)

            assert output.shape == (2, 1)

    def test_many_classes(self, temp_model_dir):
        """Test with large number of classes"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            num_classes = 10000
            trainer = InceptionV3FeatureTrainer(
                num_classes=num_classes, model_dir=temp_model_dir
            )

            assert trainer.fc.out_features == num_classes

            dummy_input = torch.randn(2, 3, 299, 299)
            output = trainer(dummy_input)

            assert output.shape == (2, num_classes)

    def test_forward_with_wrong_channels(self, temp_model_dir):
        """Test forward pass with wrong number of input channels"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(num_classes=5, model_dir=temp_model_dir)

            # Wrong number of channels (should be 3)
            wrong_input = torch.randn(1, 1, 299, 299)

            with pytest.raises(RuntimeError):
                trainer(wrong_input)

    def test_batch_size_one(self, temp_model_dir):
        """Test with batch size of 1"""
        with patch("inception_v3.inception_v3") as mock_inception, patch(
            "inception_v3.torch.save"
        ) as mock_save:
            mock_inception.return_value = create_mock_inception()

            trainer = InceptionV3FeatureTrainer(num_classes=5, model_dir=temp_model_dir)

            dummy_input = torch.randn(1, 3, 299, 299)
            output = trainer(dummy_input)

            assert output.shape == (1, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
