"""ResNet Classifier Model

This module provides ResNet-based classifiers for transfer learning.
The model supports:
- Multiple ResNet variants (ResNet50, ResNet101, ResNet152)
- Loading pretrained ImageNet weights
- Custom classification head for target task
- Feature extraction mode
- Frozen feature extractor with trainable head
- Selective layer unfreezing for fine-tuning
"""

import fnmatch
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, resnet152

from src.base.model import BaseModel


class ResNetClassifier(BaseModel):
    """ResNet-based classifier with transfer learning support.

    This model wraps torchvision's ResNet architectures and adapts them for
    binary/multi-class classification tasks. It supports:
    - Multiple ResNet variants (ResNet50, ResNet101, ResNet152)
    - Pretrained ImageNet weights
    - Custom classification head
    - Optional feature extraction (frozen backbone)
    - Flexible fine-tuning strategies

    The model can operate in three modes:
    1. Feature extraction: All layers frozen except final classification head
    2. Fine-tuning: All layers trainable
    3. Selective unfreezing: Specific layers unfrozen based on patterns

    Args:
        num_classes: Number of output classes for classification
        variant: ResNet variant to use ('resnet50', 'resnet101', 'resnet152')
        pretrained: Whether to load pretrained ImageNet weights (default: True)
        freeze_backbone: Whether to freeze all layers except classification head (default: True)
        trainable_layers: List of layer name patterns to unfreeze (default: None)
                         Patterns support wildcards (e.g., "layer4*", "layer3*")
                         When provided, overrides freeze_backbone for matched layers
        model_dir: Directory to cache pretrained weights (default: "./models/")
        dropout: Dropout probability for classification head (default: 0.0)

    Example:
        >>> # Binary classification with frozen backbone
        >>> model = ResNetClassifier(num_classes=2, variant='resnet50', freeze_backbone=True)
        >>> output = model(images)  # [batch_size, 2]
        >>>
        >>> # Multi-class with fine-tuning
        >>> model = ResNetClassifier(num_classes=10, variant='resnet101', freeze_backbone=False)
        >>> loss = model.compute_loss(output, targets)
        >>>
        >>> # Selective unfreezing: only train layer4 and head
        >>> model = ResNetClassifier(num_classes=2, variant='resnet50',
        ...                          freeze_backbone=True, trainable_layers=["layer4*"])
        >>> # Or unfreeze multiple layer groups
        >>> model = ResNetClassifier(num_classes=2, variant='resnet50',
        ...                          freeze_backbone=True, trainable_layers=["layer3*", "layer4*"])
    """

    VALID_VARIANTS = ["resnet50", "resnet101", "resnet152"]

    def __init__(
        self,
        num_classes: int,
        variant: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        trainable_layers: Optional[List[str]] = None,
        model_dir: str = "./models/",
        dropout: float = 0.0,
    ):
        super().__init__()

        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unsupported ResNet variant: {variant}. "
                f"Supported variants: {self.VALID_VARIANTS}"
            )

        self.num_classes = num_classes
        self.variant = variant
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.trainable_layers = trainable_layers
        self.model_dir = Path(model_dir)
        self.dropout = dropout

        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load ResNet backbone
        self._load_resnet_backbone()

        # Replace classification head
        self._create_classification_head()

        # Apply freezing strategy
        if self.freeze_backbone:
            self._freeze_backbone_layers()

        # Apply selective unfreezing if specified
        if self.trainable_layers is not None:
            self.set_trainable_layers(self.trainable_layers)

    def _load_resnet_backbone(self):
        """Load ResNet backbone with optional pretrained weights.

        This method handles loading pretrained weights from either:
        1. Local cache (model_dir/{variant}.pth)
        2. TorchVision's pretrained models (downloads if needed)
        """
        model_path = self.model_dir / f"{self.variant}.pth"

        if model_path.exists():
            # Load from local cache
            resnet = torch.load(model_path, weights_only=False)
        else:
            # Download pretrained weights if requested
            if self.variant == "resnet50":
                resnet = resnet50(pretrained=self.pretrained)
            elif self.variant == "resnet101":
                resnet = resnet101(pretrained=self.pretrained)
            else:  # resnet152
                resnet = resnet152(pretrained=self.pretrained)

            # Cache the pretrained model
            if self.pretrained:
                torch.save(resnet, model_path)

        # Extract feature extraction layers (excluding fc)
        # ResNet architecture: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Store original fc for feature dimension reference
        self._feature_dim = resnet.fc.in_features

    def _create_classification_head(self):
        """Create custom classification head.

        Replaces the original ResNet classification head with a new one
        for the target task. The head consists of:
        - Optional Dropout layer
        - Linear layer for classification
        """
        # ResNet features dimension depends on variant (all use 2048 for ResNet50+)
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        else:
            self.dropout_layer = nn.Identity()

        self.fc = nn.Linear(self._feature_dim, self.num_classes)

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
        if self.dropout > 0:
            for param in self.dropout_layer.parameters():
                param.requires_grad = True

    def set_trainable_layers(self, layer_patterns: List[str]):
        """Selectively unfreeze layers matching the given patterns.

        This method allows fine-grained control over which layers are trainable,
        enabling strategies like:
        - Unfreezing only the top layers (e.g., "layer4*")
        - Unfreezing multiple layer groups (e.g., ["layer3*", "layer4*"])
        - Unfreezing specific layers (e.g., ["layer4", "layer3"])

        Layer patterns support Unix-style wildcards:
        - '*' matches any sequence of characters
        - '?' matches any single character
        - '[seq]' matches any character in seq

        Available layer names in ResNet:
        - conv1, bn1, relu, maxpool
        - layer1, layer2, layer3, layer4
        - avgpool
        - fc (classification head - always trainable)

        Args:
            layer_patterns: List of layer name patterns to unfreeze
                          Examples: ["layer4*"], ["layer3*", "layer4*"]

        Example:
            >>> model = ResNetClassifier(num_classes=2, variant='resnet50', freeze_backbone=True)
            >>> # Unfreeze only the top residual block
            >>> model.set_trainable_layers(["layer4*"])
            >>>
            >>> # Unfreeze multiple blocks
            >>> model.set_trainable_layers(["layer3*", "layer4*"])
            >>>
            >>> # Unfreeze specific layers
            >>> model.set_trainable_layers(["layer3", "layer4"])
        """
        # Get all layer names that correspond to named modules
        layer_names = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
            "dropout_layer",
        ]

        # Find layers matching the patterns
        layers_to_unfreeze = []
        for pattern in layer_patterns:
            for layer_name in layer_names:
                if fnmatch.fnmatch(layer_name, pattern):
                    layers_to_unfreeze.append(layer_name)

        # Remove duplicates while preserving order
        layers_to_unfreeze = list(dict.fromkeys(layers_to_unfreeze))

        # Unfreeze matched layers
        for layer_name in layers_to_unfreeze:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet.

        Args:
            x: Input tensor of shape [batch_size, 3, H, W]
               ResNet accepts various input sizes, but 224x224 is standard

        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        # Feature extraction through ResNet layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten: [batch_size, feature_dim, 1, 1] -> [batch_size, feature_dim]
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
            >>> model = ResNetClassifier(num_classes=2, variant='resnet50')
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
            >>> model = ResNetClassifier(num_classes=2, variant='resnet50', freeze_backbone=True)
            >>> optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=0.001)
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification.

        The feature dimension depends on the ResNet variant:
        - ResNet50: 2048-dimensional
        - ResNet101: 2048-dimensional
        - ResNet152: 2048-dimensional

        Useful for feature analysis, visualization, or computing metrics like FID.

        Args:
            x: Input tensor of shape [batch_size, 3, H, W]

        Returns:
            Feature tensor of shape [batch_size, feature_dim]
            where feature_dim is typically 2048

        Example:
            >>> model = ResNetClassifier(num_classes=2, variant='resnet50')
            >>> features = model.extract_features(images)
            >>> print(features.shape)  # [batch_size, 2048]
        """
        # Feature extraction (same as forward but without classification head)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
