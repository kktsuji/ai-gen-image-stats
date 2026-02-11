"""
Diffusion Experiment Module

This module contains the implementation for diffusion model experiments, including:
- Diffusion model implementations (DDPM, etc.)
- Diffusion-specific training logic
- Data loading and preprocessing for diffusion tasks
- Generated image logging and evaluation

The diffusion experiment is used to:
1. Train diffusion models on real data
2. Generate synthetic data samples
3. Evaluate generation quality (FID, IS, etc.)
4. Provide synthetic data augmentation for downstream tasks

Usage:
    from src.experiments.diffusion import DiffusionTrainer, DiffusionDataLoader

    # Initialize components
    trainer = DiffusionTrainer(config)
    trainer.train()

    # Generate synthetic samples
    trainer.generate(num_samples=1000)

For more information, see the Architecture Specification at:
docs/standards/architecture.md
"""

from src.experiments.diffusion.dataloader import DiffusionDataLoader

__all__ = [
    # Will be populated as components are implemented
    # 'DiffusionTrainer',
    "DiffusionDataLoader",
    # 'DiffusionLogger',
    # 'DDPMModel',
]
