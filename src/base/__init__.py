"""Base classes and interfaces for experiments.

This module provides abstract base classes that define common interfaces
and shared functionality across all experiment types.
"""

from src.base.dataloader import BaseDataLoader
from src.base.model import BaseModel

__all__ = [
    "BaseDataLoader",
    "BaseModel",
]

# Import other base classes as they are implemented:
# from src.base.logger import BaseLogger          # Step 13
# from src.base.trainer import BaseTrainer        # Step 14
