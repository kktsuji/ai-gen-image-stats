"""Tests for WideResNet-28-10 CIFAR-10 Trainer and Feature Extractor Models."""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from wrn28_cifar10 import WRN28Cifar10FeatureExtractor, WRN28Cifar10Trainer


class TestWRN28Cifar10Trainer(unittest.TestCase):
    """Test cases for WRN28Cifar10Trainer"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.batch_size = 4
        self.input_size_32 = (self.batch_size, 3, 32, 32)
        self.input_size_40 = (self.batch_size, 3, 40, 40)

    def test_initialization(self):
        """Test model initialization"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        self.assertIsInstance(model, nn.Module)
        self.assertTrue(hasattr(model, "features"))
        self.assertTrue(hasattr(model, "fc1"))
        self.assertTrue(hasattr(model, "fc2"))
        self.assertTrue(hasattr(model, "relu"))
        self.assertTrue(hasattr(model, "dropout"))

    def test_initialization_with_custom_dropout(self):
        """Test model initialization with custom dropout rate"""
        dropout_rate = 0.5
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir, dropout_rate=dropout_rate)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.dropout.p, dropout_rate)

    def test_forward_pass_32x32(self):
        """Test forward pass with 32x32 input"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        model.eval()

        x = torch.randn(*self.input_size_32)
        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (self.batch_size, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_pass_40x40(self):
        """Test forward pass with 40x40 input (variable input size)"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        model.eval()

        x = torch.randn(*self.input_size_40)
        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (self.batch_size, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_output_shape(self):
        """Test that output has correct shape for binary classification"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        model.eval()

        x = torch.randn(1, 3, 40, 40)
        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (1, 2))

    def test_frozen_backbone(self):
        """Test that backbone features are frozen"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)

        for param in model.features.parameters():
            self.assertFalse(param.requires_grad)

    def test_trainable_head(self):
        """Test that classification head is trainable"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)

        for param in model.fc1.parameters():
            self.assertTrue(param.requires_grad)
        for param in model.fc2.parameters():
            self.assertTrue(param.requires_grad)

    def test_get_trainable_parameters(self):
        """Test get_trainable_parameters method"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)

        trainable_params = list(model.get_trainable_parameters())
        self.assertGreater(len(trainable_params), 0)

        for param in trainable_params:
            self.assertTrue(param.requires_grad)

    def test_trainable_parameters_count(self):
        """Test that only head parameters are trainable"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Trainable params should be much less than total params
        # fc1: feature_dim * 128 + 128 (bias)
        # fc2: 128 * 2 + 2 (bias)
        # Approximately: 640*128 + 128 + 128*2 + 2 = 82,306
        self.assertLess(trainable_params, total_params)
        self.assertGreater(trainable_params, 82000)  # At least ~82K params
        self.assertLess(trainable_params, 83000)  # Less than 83K params

    def test_training_mode(self):
        """Test model can be set to training mode"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        model.train()

        x = torch.randn(*self.input_size_40)
        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, 2))
        self.assertTrue(output.requires_grad)

    def test_backward_pass(self):
        """Test backward pass works correctly"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        model.train()

        x = torch.randn(2, 3, 40, 40)
        labels = torch.tensor([1, 0])

        output = model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        loss.backward()

        # Check that head gradients exist
        self.assertIsNotNone(model.fc1.weight.grad)
        self.assertIsNotNone(model.fc2.weight.grad)

        # Check that backbone gradients don't exist (frozen)
        for param in model.features.parameters():
            self.assertIsNone(param.grad)

    def test_model_caching(self):
        """Test that model is cached and reused"""
        model1 = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        model_path = os.path.join(self.temp_dir, "wrn28_10_cifar10.pth")
        self.assertTrue(os.path.exists(model_path))

        # Second initialization should load from cache
        model2 = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        self.assertIsInstance(model2, nn.Module)

    def test_dropout_during_training(self):
        """Test dropout is active during training"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir, dropout_rate=0.5)
        model.train()

        x = torch.randn(10, 3, 40, 40)

        # Run multiple forward passes - outputs should differ due to dropout
        with torch.no_grad():
            outputs = [model(x) for _ in range(3)]

        # At least some outputs should differ
        differences = sum(
            not torch.allclose(outputs[i], outputs[i + 1])
            for i in range(len(outputs) - 1)
        )
        self.assertGreater(differences, 0)

    def test_dropout_during_eval(self):
        """Test dropout is inactive during evaluation"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir, dropout_rate=0.5)
        model.eval()

        x = torch.randn(10, 3, 40, 40)

        # Run multiple forward passes - outputs should be identical
        with torch.no_grad():
            outputs = [model(x) for _ in range(3)]

        for i in range(len(outputs) - 1):
            self.assertTrue(torch.allclose(outputs[i], outputs[i + 1]))


class TestWRN28Cifar10FeatureExtractor(unittest.TestCase):
    """Test cases for WRN28Cifar10FeatureExtractor"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.batch_size = 4
        self.input_size_32 = (self.batch_size, 3, 32, 32)
        self.input_size_40 = (self.batch_size, 3, 40, 40)

    def test_initialization(self):
        """Test model initialization"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        self.assertIsInstance(model, nn.Module)
        self.assertTrue(hasattr(model, "features"))

    def test_forward_pass_32x32(self):
        """Test forward pass with 32x32 input"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        model.eval()

        x = torch.randn(*self.input_size_32)
        with torch.no_grad():
            output = model(x)

        # Output should be [batch_size, feature_dim]
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertGreater(output.shape[1], 0)  # Feature dimension > 0
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_pass_40x40(self):
        """Test forward pass with 40x40 input"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        model.eval()

        x = torch.randn(*self.input_size_40)
        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape[0], self.batch_size)
        self.assertGreater(output.shape[1], 0)
        self.assertTrue(torch.isfinite(output).all())

    def test_feature_dimension(self):
        """Test that feature dimension is consistent (typically 640 for WRN-28-10)"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        model.eval()

        x1 = torch.randn(1, 3, 32, 32)
        x2 = torch.randn(1, 3, 40, 40)

        with torch.no_grad():
            features1 = model(x1)
            features2 = model(x2)

        # Same feature dimension regardless of input size
        self.assertEqual(features1.shape[1], features2.shape[1])
        # Typically 640 for WRN-28-10
        self.assertGreater(features1.shape[1], 600)
        self.assertLess(features1.shape[1], 700)

    def test_eval_mode(self):
        """Test that model is in eval mode after initialization"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        self.assertFalse(model.training)

    def test_no_gradients(self):
        """Test that no gradients are computed (feature extraction only)"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)

        x = torch.randn(2, 3, 40, 40, requires_grad=True)

        # Use no_grad context for inference
        with torch.no_grad():
            output = model(x)

        # Output should not require gradients
        self.assertFalse(output.requires_grad)

    def test_deterministic_output(self):
        """Test that output is deterministic (no dropout/randomness)"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        model.eval()

        x = torch.randn(5, 3, 40, 40)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        self.assertTrue(torch.allclose(output1, output2))

    def test_batch_processing(self):
        """Test processing different batch sizes"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        model.eval()

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 40, 40)
            with torch.no_grad():
                output = model(x)
            self.assertEqual(output.shape[0], batch_size)

    def test_model_caching(self):
        """Test that model is cached and reused"""
        model1 = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        model_path = os.path.join(self.temp_dir, "wrn28_10_cifar10.pth")
        self.assertTrue(os.path.exists(model_path))

        # Second initialization should load from cache
        model2 = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        self.assertIsInstance(model2, nn.Module)

    def test_variable_input_sizes(self):
        """Test that GAP allows variable input sizes"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        model.eval()

        input_sizes = [(32, 32), (40, 40), (48, 48), (64, 64)]

        with torch.no_grad():
            outputs = [model(torch.randn(2, 3, h, w)) for h, w in input_sizes]

        # All outputs should have the same feature dimension
        feature_dims = [out.shape[1] for out in outputs]
        self.assertEqual(len(set(feature_dims)), 1)  # All dimensions are equal

    def test_feature_consistency_across_sizes(self):
        """Test that features from same input at different sizes are reasonable"""
        model = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        model.eval()

        x_small = torch.randn(1, 3, 32, 32)
        x_large = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            features_small = model(x_small)
            features_large = model(x_large)

        # Same feature dimension
        self.assertEqual(features_small.shape, features_large.shape)
        # Features should be finite
        self.assertTrue(torch.isfinite(features_small).all())
        self.assertTrue(torch.isfinite(features_large).all())


class TestModelIntegration(unittest.TestCase):
    """Integration tests for both models"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def test_shared_backbone_weights(self):
        """Test that Trainer and FeatureExtractor share the same cached model"""
        trainer = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        extractor = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)

        model_path = os.path.join(self.temp_dir, "wrn28_10_cifar10.pth")
        self.assertTrue(os.path.exists(model_path))

    def test_feature_extraction_vs_training_features(self):
        """Test that feature extractor and trainer produce consistent feature dimensions"""
        trainer = WRN28Cifar10Trainer(model_dir=self.temp_dir)
        extractor = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)

        trainer.eval()
        extractor.eval()

        x = torch.randn(2, 3, 40, 40)

        with torch.no_grad():
            # Get features from extractor
            features_extractor = extractor(x)

            # Get intermediate features from trainer (before fc layers)
            x_trainer = trainer.features(x)
            x_trainer = torch.nn.functional.adaptive_avg_pool2d(x_trainer, (1, 1))
            features_trainer = x_trainer.view(x_trainer.size(0), -1)

        # Should produce same shape (feature dimension consistency)
        self.assertEqual(features_extractor.shape, features_trainer.shape)
        # Both should be finite
        self.assertTrue(torch.isfinite(features_extractor).all())
        self.assertTrue(torch.isfinite(features_trainer).all())

    def test_end_to_end_training_workflow(self):
        """Test complete training workflow"""
        model = WRN28Cifar10Trainer(model_dir=self.temp_dir, dropout_rate=0.3)
        model.train()

        # Create dummy dataset
        x = torch.randn(8, 3, 40, 40)
        y = torch.randint(0, 2, (8,))

        # Forward pass
        logits = model(x)
        self.assertEqual(logits.shape, (8, 2))

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()

        # Check gradients exist for trainable parameters only
        self.assertIsNotNone(model.fc1.weight.grad)
        self.assertIsNotNone(model.fc2.weight.grad)

        for param in model.features.parameters():
            self.assertIsNone(param.grad)

    def test_end_to_end_inference_workflow(self):
        """Test complete inference workflow"""
        extractor = WRN28Cifar10FeatureExtractor(model_dir=self.temp_dir)
        extractor.eval()

        # Create dummy dataset
        x = torch.randn(16, 3, 40, 40)

        with torch.no_grad():
            features = extractor(x)

        self.assertEqual(features.shape[0], 16)
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
