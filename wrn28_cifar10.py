"""WideResNet-28-10 CIFAR-10 Trainer and Feature Extractor Models."""

import os

import torch
import torch.nn as nn


class WRN28Cifar10Trainer(nn.Module):
    """
    WideResNet-28-10 model pre-trained on CIFAR-10 for binary classification fine-tuning.

    Architecture:
    1. Frozen backbone (feature extractor from pre-trained WRN-28-10, always in eval mode)
    2. Global Average Pooling (parameter-free spatial pooling)
    3. Dense(256) with ReLU (bottleneck layer for task-specific feature mapping)
    4. Dropout (regularization to prevent overfitting with small positive samples)
    5. Dense(2) output (two logits for CrossEntropyLoss with softmax)

    Benefits:
    - Frozen features in eval mode prevent BatchNorm instability
    - GAP prevents parameter explosion and allows variable input sizes
    - Larger bottleneck (256) preserves more information through dropout
    - Proper weight initialization for faster convergence
    - Lower dropout rate (0.2) for better minority class learning
    """

    def __init__(self, model_dir: str = "./models/", dropout_rate: float = 0.2):
        """
        Args:
            model_dir: Directory to save/load pre-trained model
            dropout_rate: Dropout probability (0.1-0.3 recommended for small datasets, default 0.2)
        """
        super(WRN28Cifar10Trainer, self).__init__()

        os.makedirs(model_dir, exist_ok=True)
        _model_path = os.path.join(model_dir, "wrn28_10_cifar10.pth")

        # Load pretrained model
        if not os.path.exists(_model_path):
            from pytorchcv.model_provider import get_model as ptcv_get_model

            wrn_model = ptcv_get_model("wrn28_10_cifar10", pretrained=True)
            torch.save(wrn_model, _model_path)
        else:
            wrn_model = torch.load(_model_path)

        # Extract feature extractor (all layers except final classifier)
        self.features = wrn_model.features

        # Freeze all feature extraction layers (backbone remains pre-trained)
        for param in self.features.parameters():
            param.requires_grad = False

        # Set features to eval mode to freeze BatchNorm statistics
        self.features.eval()

        # Determine feature dimension by forward pass
        # WideResNet-28-10 typically outputs 640 channels
        # Using 40x40 input size instead of CIFAR-10's 32x32
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 40, 40)
            feature_output = self.features(dummy_input)
            # GAP: [B, C, H, W] -> [B, C, 1, 1]
            feature_output = torch.nn.functional.adaptive_avg_pool2d(
                feature_output, (1, 1)
            )
            feature_dim = feature_output.view(1, -1).shape[1]

        # Build custom classification head (trainable)
        # GAP is applied in forward(), no parameters needed here
        # Increased bottleneck size for better feature preservation
        self.fc1 = nn.Linear(feature_dim, 256)  # Bottleneck layer (increased from 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, 2)  # Two outputs for binary classification

        # Initialize new layers with proper initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        # Only the new head layers have requires_grad=True
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            logits: [B, 2] raw logits (use with CrossEntropyLoss, softmax applied automatically)
        """
        # Step 1: Feature extraction (frozen backbone, always in eval mode)
        with torch.no_grad():
            x = self.features(x)  # [B, C, H', W']

        # Step 2: Global Average Pooling (parameter-free)
        # Spatial dimensions (H', W') -> (1, 1)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]

        # Step 3: Bottleneck layer with ReLU
        x = self.fc1(x)  # [B, 256]
        x = self.relu(x)

        # Step 4: Dropout for regularization
        x = self.dropout(x)

        # Step 5: Output layer (two logits)
        x = self.fc2(x)  # [B, 2]

        return x

    def get_trainable_parameters(self):
        """Returns only the parameters that require gradients (custom head only)"""
        return filter(lambda p: p.requires_grad, self.parameters())


class WRN28Cifar10FeatureExtractor(nn.Module):
    """
    WideResNet-28-10 model pre-trained on CIFAR-10 for feature extraction.
    """

    def __init__(self, model_dir: str = "./models/"):
        super(WRN28Cifar10FeatureExtractor, self).__init__()

        os.makedirs(model_dir, exist_ok=True)
        _model_path = os.path.join(model_dir, "wrn28_10_cifar10.pth")

        if not os.path.exists(_model_path):
            from pytorchcv.model_provider import get_model as ptcv_get_model

            wrn_model = ptcv_get_model("wrn28_10_cifar10", pretrained=True)
            torch.save(wrn_model, _model_path)
        else:
            wrn_model = torch.load(_model_path)

        wrn_model.eval()

        # Extract feature extractor (all layers except final classifier)
        self.features = wrn_model.features

        # Set to evaluation mode
        self.eval()

    def forward(self, x):
        # Pass through feature extraction layers
        x = self.features(x)

        # Apply global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # Flatten to [batch_size, feature_dim]
        x = x.view(x.size(0), -1)

        return x
