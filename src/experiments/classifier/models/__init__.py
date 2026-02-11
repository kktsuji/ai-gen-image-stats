"""Classifier Models

This module contains classifier model implementations for the classifier experiment.
Models are designed for binary/multi-class classification tasks and inherit from
the BaseModel interface.

Available models:
- InceptionV3: Transfer learning with InceptionV3 backbone
- ResNet: ResNet50 and ResNet101 variants (to be implemented)

Each model provides:
- Pretrained weight loading
- Custom classification head
- Feature extraction capabilities
"""

from src.experiments.classifier.models.inceptionv3 import InceptionV3Classifier

__all__ = [
    "InceptionV3Classifier",
]
