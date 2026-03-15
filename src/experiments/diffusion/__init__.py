"""
Diffusion Experiment Module

This module contains the implementation for diffusion model experiments, including:
- Diffusion model implementations (DDPM, etc.)
- Diffusion-specific training logic
- Generated image logging and evaluation

The diffusion experiment is used to:
1. Train diffusion models on real data
2. Generate synthetic data samples
3. Evaluate generation quality (FID, IS, etc.)
4. Provide synthetic data augmentation for downstream tasks

Usage:
    from src.experiments.diffusion import DiffusionTrainer

    # Initialize components
    trainer = DiffusionTrainer(config)
    trainer.train()

    # Generate synthetic samples
    trainer.generate(num_samples=1000)
"""

__all__: list[str] = [
    # Will be populated as components are implemented
    # 'DiffusionTrainer',
    # 'DiffusionLogger',
    # 'DDPMModel',
]
