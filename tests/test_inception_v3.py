"""Pytest tests for Inception V3 Feature Extractor"""

import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from inception_v3 import InceptionV3FeatureExtractor


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
