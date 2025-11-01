"""Pytest tests for ResNet Feature Extractor"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from resnet import ResNetFeatureExtractor


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def create_mock_resnet():
    """Create a minimal mock ResNet model with proper nn.Module structure"""

    class MockResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3)
            self.layer1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1))
            self.layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)

        def forward(self, x):
            return x

    return MockResNet()


class TestResNetFeatureExtractorInitialization:
    """Test ResNetFeatureExtractor initialization"""

    def test_init_with_default_params(self, temp_model_dir):
        """Test initialization with default parameters"""
        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()

            extractor = ResNetFeatureExtractor(model_dir=temp_model_dir)

            assert extractor is not None
            assert isinstance(extractor, nn.Module)
            mock_resnet50.assert_called_once()

    def test_init_creates_model_directory(self, temp_model_dir):
        """Test that model directory is created if it doesn't exist"""
        model_dir = os.path.join(temp_model_dir, "new_models")
        assert not os.path.exists(model_dir)

        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()

            ResNetFeatureExtractor(model_dir=model_dir)

            assert os.path.exists(model_dir)

    @pytest.mark.parametrize("variant", ["resnet50", "resnet101", "resnet152"])
    def test_init_with_different_variants(self, temp_model_dir, variant):
        """Test initialization with different ResNet variants"""
        with patch(f"resnet.{variant}") as mock_resnet, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet.return_value = create_mock_resnet()

            extractor = ResNetFeatureExtractor(
                model_dir=temp_model_dir, resnet_variant=variant
            )

            assert extractor is not None
            mock_resnet.assert_called_once()

    def test_init_with_invalid_variant(self, temp_model_dir):
        """Test that invalid variant raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported ResNet variant"):
            ResNetFeatureExtractor(model_dir=temp_model_dir, resnet_variant="resnet34")

    def test_init_downloads_model_if_not_cached(self, temp_model_dir):
        """Test that model is downloaded if not in cache"""
        model_path = os.path.join(temp_model_dir, "resnet50.pth")
        assert not os.path.exists(model_path)

        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()

            ResNetFeatureExtractor(model_dir=temp_model_dir, resnet_variant="resnet50")

            # Verify model was saved
            mock_save.assert_called_once()
            assert mock_save.call_args[0][1] == model_path

    def test_init_loads_cached_model(self, temp_model_dir):
        """Test that cached model is loaded from cache when it exists"""
        model_path = os.path.join(temp_model_dir, "resnet50.pth")

        # First, patch torch.save and create a model to simulate caching
        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()
            ResNetFeatureExtractor(model_dir=temp_model_dir, resnet_variant="resnet50")
            # Verify save was called
            assert mock_save.called

        # Create a real file to simulate cached model
        # Touch the file so os.path.exists returns True
        with open(model_path, "w") as f:
            f.write("dummy")

        # Now verify it attempts to load when file exists
        with patch("resnet.torch.load") as mock_load, patch(
            "resnet.resnet50"
        ) as mock_resnet50:
            # Return a valid mock when load is called
            mock_load.return_value = create_mock_resnet()

            ResNetFeatureExtractor(model_dir=temp_model_dir, resnet_variant="resnet50")

            # Verify cached model was loaded
            mock_load.assert_called_once_with(model_path)
            # Verify model download function was not called
            mock_resnet50.assert_not_called()

    def test_model_set_to_eval_mode(self, temp_model_dir):
        """Test that model is set to evaluation mode"""
        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()

            extractor = ResNetFeatureExtractor(model_dir=temp_model_dir)

            # Extractor should be in eval mode
            assert not extractor.training

    def test_final_fc_layer_removed(self, temp_model_dir):
        """Test that the final fully connected layer is removed"""
        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()

            extractor = ResNetFeatureExtractor(model_dir=temp_model_dir)

            # Verify features Sequential was created without last layer
            assert hasattr(extractor, "features")
            assert isinstance(extractor.features, nn.Sequential)
            # The mock has 10 children (conv1, bn1, relu, maxpool, layer1-4, avgpool, fc)
            # After removing fc, should have 9
            assert len(list(extractor.features.children())) == 9


class TestResNetFeatureExtractorForward:
    """Test ResNetFeatureExtractor forward pass"""

    @pytest.fixture
    def extractor(self, temp_model_dir):
        """Create a ResNetFeatureExtractor instance for testing"""
        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()

            return ResNetFeatureExtractor(model_dir=temp_model_dir)

    def test_forward_with_single_image(self, extractor):
        """Test forward pass with a single image"""
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)

        with patch.object(
            extractor.features, "__call__", return_value=torch.randn(1, 2048, 1, 1)
        ):
            output = extractor.forward(dummy_input)

            assert output is not None
            assert isinstance(output, torch.Tensor)
            # Output should be flattened
            assert output.dim() == 2
            assert output.size(0) == 1  # batch size

    def test_forward_with_batch(self, extractor):
        """Test forward pass with a batch of images"""
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        with patch.object(
            extractor.features,
            "__call__",
            return_value=torch.randn(batch_size, 2048, 1, 1),
        ):
            output = extractor.forward(dummy_input)

            assert output is not None
            assert output.dim() == 2
            assert output.size(0) == batch_size

    def test_forward_output_shape(self, extractor):
        """Test that forward pass produces correct output shape"""
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        # Mock features output - the mock runs through features which outputs (batch, 512, 1, 1)
        # after avgpool in our mock ResNet
        mock_features_output = torch.randn(batch_size, 512, 1, 1)

        with patch.object(
            extractor.features, "__call__", return_value=mock_features_output
        ):
            output = extractor.forward(dummy_input)

            # After view, should be (batch_size, 512)
            assert output.shape == (batch_size, 512)

    def test_forward_type_annotation(self, extractor):
        """Test that forward method has correct type annotations"""
        import inspect

        sig = inspect.signature(extractor.forward)

        # Check parameter type annotation
        assert "x" in sig.parameters
        assert sig.parameters["x"].annotation == torch.Tensor

        # Check return type annotation
        assert sig.return_annotation == torch.Tensor

    def test_forward_preserves_batch_dimension(self, extractor):
        """Test that batch dimension is preserved through forward pass"""
        for batch_size in [1, 2, 4, 8]:
            dummy_input = torch.randn(batch_size, 3, 224, 224)

            with patch.object(
                extractor.features,
                "__call__",
                return_value=torch.randn(batch_size, 2048, 1, 1),
            ):
                output = extractor.forward(dummy_input)

                assert output.size(0) == batch_size


class TestResNetFeatureExtractorIntegration:
    """Integration tests for ResNetFeatureExtractor"""

    def test_different_variants_use_correct_models(self, temp_model_dir):
        """Test that different variants use correct model loaders"""
        variants = ["resnet50", "resnet101", "resnet152"]

        for variant in variants:
            with patch(f"resnet.{variant}") as mock_resnet, patch(
                "resnet.torch.save"
            ) as mock_save:
                mock_resnet.return_value = create_mock_resnet()

                ResNetFeatureExtractor(model_dir=temp_model_dir, resnet_variant=variant)

                mock_resnet.assert_called_once()

    def test_model_persistence(self, temp_model_dir):
        """Test that model is properly saved and loaded"""
        model_path = os.path.join(temp_model_dir, "resnet50.pth")

        # First initialization
        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()

            ResNetFeatureExtractor(model_dir=temp_model_dir)

            # Check save was called
            assert mock_save.called
            save_path = mock_save.call_args[0][1]
            assert save_path == model_path

    def test_extractor_inherits_from_nn_module(self, temp_model_dir):
        """Test that ResNetFeatureExtractor properly inherits from nn.Module"""
        with patch("resnet.resnet50") as mock_resnet50, patch(
            "resnet.torch.save"
        ) as mock_save:
            mock_resnet50.return_value = create_mock_resnet()

            extractor = ResNetFeatureExtractor(model_dir=temp_model_dir)

            assert isinstance(extractor, nn.Module)
            # Should have standard nn.Module methods
            assert hasattr(extractor, "parameters")
            assert hasattr(extractor, "train")
            assert hasattr(extractor, "eval")


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_model_dir_string(self):
        """Test with empty model directory string"""
        with pytest.raises((ValueError, OSError)):
            with patch("resnet.resnet50"):
                ResNetFeatureExtractor(model_dir="")

    def test_none_variant_raises_error(self, temp_model_dir):
        """Test that None variant raises appropriate error"""
        with pytest.raises((ValueError, TypeError)):
            ResNetFeatureExtractor(model_dir=temp_model_dir, resnet_variant=None)

    def test_case_sensitive_variant_name(self, temp_model_dir):
        """Test that variant name is case-sensitive"""
        with pytest.raises(ValueError, match="Unsupported ResNet variant"):
            ResNetFeatureExtractor(
                model_dir=temp_model_dir, resnet_variant="ResNet50"  # Wrong case
            )

    def test_numeric_variant_raises_error(self, temp_model_dir):
        """Test that numeric variant raises error"""
        with pytest.raises(ValueError):
            ResNetFeatureExtractor(model_dir=temp_model_dir, resnet_variant="50")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
