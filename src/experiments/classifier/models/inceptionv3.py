"""InceptionV3 Classifier Model

This module provides an InceptionV3-based classifier for transfer learning.
The model supports:
- Loading pretrained ImageNet weights
- Custom classification head for target task
- Feature extraction mode
- Frozen feature extractor with trainable head
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3

from src.base.model import BaseModel


class InceptionV3Classifier(BaseModel):
    """InceptionV3-based classifier with transfer learning support.

    This model wraps torchvision's InceptionV3 architecture and adapts it for
    binary/multi-class classification tasks. It supports:
    - Pretrained ImageNet weights
    - Custom classification head
    - Optional feature extraction (frozen backbone)
    - Flexible fine-tuning strategies

    The model can operate in two modes:
    1. Feature extraction: All layers frozen except final classification head
    2. Fine-tuning: All layers trainable (or selective unfreezing)

    Args:
        num_classes: Number of output classes for classification
        pretrained: Whether to load pretrained ImageNet weights (default: True)
        freeze_backbone: Whether to freeze all layers except classification head (default: True)
        model_dir: Directory to cache pretrained weights (default: "./models/")
        dropout: Dropout probability for classification head (default: 0.5)

    Example:
        >>> # Binary classification with frozen backbone
        >>> model = InceptionV3Classifier(num_classes=2, freeze_backbone=True)
        >>> output = model(images)  # [batch_size, 2]
        >>>
        >>> # Multi-class with fine-tuning
        >>> model = InceptionV3Classifier(num_classes=10, freeze_backbone=False)
        >>> loss = model.compute_loss(output, targets)
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        model_dir: str = "./models/",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model_dir = Path(model_dir)
        self.dropout = dropout

        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load InceptionV3 model
        self._load_inception_backbone()

        # Replace classification head
        self._create_classification_head()

        # Apply freezing strategy
        if self.freeze_backbone:
            self._freeze_backbone_layers()

    def _load_inception_backbone(self):
        """Load InceptionV3 backbone with optional pretrained weights.

        This method handles loading pretrained weights from either:
        1. Local cache (model_dir/inception_v3.pth)
        2. TorchVision's pretrained models (downloads if needed)
        """
        model_path = self.model_dir / "inception_v3.pth"

        if model_path.exists():
            # Load from local cache
            inception = torch.load(model_path, weights_only=False)
        else:
            # Download pretrained weights if requested
            inception = inception_v3(pretrained=self.pretrained, transform_input=False)
            # Cache the pretrained model
            if self.pretrained:
                torch.save(inception, model_path)

        # Extract feature extraction layers (excluding aux classifier and fc)
        # We manually copy layers to avoid issues with aux classifier during training
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

    def _create_classification_head(self):
        """Create custom classification head.

        Replaces the original InceptionV3 classification head with a new one
        for the target task. The head consists of:
        - Global Average Pooling (handled in forward)
        - Dropout layer
        - Linear layer for classification
        """
        # InceptionV3 features are 2048-dimensional after global pooling
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(2048, self.num_classes)

    def _freeze_backbone_layers(self):
        """Freeze all backbone layers except the classification head.

        This enables feature extraction mode where only the final layer is trained.
        Useful for transfer learning with limited data.
        """
        # Freeze all feature extraction layers
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze only the classification head
        for param in self.fc.parameters():
            param.requires_grad = True
        for param in self.dropout_layer.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through InceptionV3.

        Args:
            x: Input tensor of shape [batch_size, 3, 299, 299]
               InceptionV3 expects 299x299 images

        Returns:
            Logits tensor of shape [batch_size, num_classes]

        Note:
            Auxiliary outputs are not used during inference. During training,
            if auxiliary outputs are needed, they should be handled in the trainer.
        """
        # Feature extraction through InceptionV3 layers
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        # Global average pooling: [batch_size, 2048, 8, 8] -> [batch_size, 2048, 1, 1]
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten: [batch_size, 2048, 1, 1] -> [batch_size, 2048]
        x = x.view(x.size(0), -1)

        # Classification head
        x = self.dropout_layer(x)
        x = self.fc(x)

        return x

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute classification loss using cross-entropy.

        Args:
            predictions: Model output logits of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]
                     For multi-class: integer class indices
                     For binary: 0 or 1
            reduction: Loss reduction method ('mean', 'sum', or 'none')

        Returns:
            Scalar loss tensor (if reduction is 'mean' or 'sum')
            or per-sample losses (if reduction is 'none')

        Example:
            >>> model = InceptionV3Classifier(num_classes=2)
            >>> logits = model(images)
            >>> loss = model.compute_loss(logits, labels)
        """
        return F.cross_entropy(predictions, targets, reduction=reduction)

    def get_trainable_parameters(self):
        """Get only the trainable parameters.

        Useful for optimizers when using frozen backbone.

        Returns:
            Iterator over trainable parameters

        Example:
            >>> model = InceptionV3Classifier(num_classes=2, freeze_backbone=True)
            >>> optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=0.001)
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dimensional features without classification.

        Useful for feature analysis, visualization, or computing metrics like FID.

        Args:
            x: Input tensor of shape [batch_size, 3, 299, 299]

        Returns:
            Feature tensor of shape [batch_size, 2048]

        Example:
            >>> model = InceptionV3Classifier(num_classes=2)
            >>> features = model.extract_features(images)
            >>> print(features.shape)  # [batch_size, 2048]
        """
        # Feature extraction (same as forward but without classification head)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x
