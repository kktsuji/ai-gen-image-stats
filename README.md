# AI Gen Image Stats

A modular framework for training and evaluating generative AI models (GANs, Diffusion Models) and classifiers for research purposes. This project enables rapid experimentation with synthetic data augmentation and model comparisons.

## Overview

This repository implements a **Vertical Slice + Base Class** architecture pattern that prioritizes:

- **Code Reusability**: Shared training logic through base classes
- **Testability**: Four-tier testing strategy (unit, component, integration, smoke)
- **Rapid Experimentation**: Self-contained experiment types with configuration-driven workflows
- **CPU-First Development**: All code testable on CPU without GPU requirements

## Features

- **Multiple Experiment Types**:
  - **Diffusion Models**: DDPM for synthetic image generation
  - **Classifiers**: InceptionV3, ResNet (50/101), WRN for classification tasks
  - **GANs**: (Planned) Adversarial training for image generation

- **Comprehensive Testing**: Four-tier strategy from fast unit tests to GPU smoke tests

- **Configuration Management**: JSON-based configs with CLI overrides

- **Metrics & Evaluation**: FID, IS, PR-AUC, ROC-AUC for model evaluation

## Architecture

```
src/
├── base/              # Base classes for trainers, models, data loaders
├── experiments/       # Self-contained experiment implementations
│   ├── classifier/    # Classification models and training
│   ├── diffusion/     # Diffusion models (DDPM)
│   └── gan/          # GAN models (planned)
├── utils/            # CLI, config, device management, metrics
└── data/             # Dataset implementations and transforms

tests/                # Mirror of src/ structure with test tiers
configs/              # Experiment configurations (JSON)
outputs/              # Training outputs (gitignored)
```

See [docs/standards/architecture.md](docs/standards/architecture.md) for complete architecture specification.

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, but recommended for training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-gen-image-stats

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (for testing)
pip install -r requirements-dev.txt
```

## Quick Start

### Training a Classifier

```bash
# Train with config file
python -m src.main --experiment classifier --config configs/classifier/baseline.json

# Train with CLI arguments
python -m src.main --experiment classifier --model inceptionv3 --epochs 10 --batch-size 32
```

### Training a Diffusion Model

```bash
# Train diffusion model
python -m src.main --experiment diffusion --config configs/diffusion/default.json

# Generate synthetic images
python -m src.main --experiment diffusion --mode generate \
  --checkpoint outputs/checkpoints/diffusion_best.pth --num-samples 1000
```

## Testing

Our testing strategy uses four tiers:

- **Unit Tests**: Fast (< 100ms), CPU-only, pure logic
- **Component Tests**: Medium (1-5s), CPU with small data
- **Integration Tests**: Slow (10-60s), mini workflows
- **Smoke Tests**: Very slow (5-15min), GPU validation

```bash
# Fast feedback during development
pytest -m unit

# Pre-commit validation
pytest -m "unit or component"

# CI pipeline (all except smoke tests)
pytest -m "not smoke"

# Full validation with GPU
pytest -m smoke --gpu
```

## Project Status

**Current Phase**: Phase 1 - Project Foundation  
**Completed Steps**: 1/46  
**Last Updated**: 2026-02-10

This project is undergoing active refactoring according to the plan in [docs/research/refactor.md](docs/research/refactor.md).

## Documentation

- [Architecture Specification](docs/standards/architecture.md) - Complete architecture design
- [Refactoring Plan](docs/research/refactor.md) - Step-by-step implementation guide
- [User Requirements](docs/research/user-requirements.md) - Project goals and requirements
- [Technical Requirements](docs/research/technical-requirements.md) - Technical specifications

## Research Workflow

1. **Generate Synthetic Data** (optional): Train diffusion model to generate synthetic images
2. **Train Baseline Classifier**: Train on real data only
3. **Train with Synthetic Augmentation**: Train on real + synthetic data
4. **Compare Results**: Analyze performance differences using built-in tools

## Docker Usage (Legacy)

The project also supports Docker for consistent environments:

```bash
# Build the docker image
docker build -t kktsuji/pytorch-2.9.0-cuda12.8-cudnn9 .

# Run training
docker run --rm -it --gpus all --network=host \
  -v $PWD:/work -w /work \
  --user $(id -u):$(id -g) \
  kktsuji/pytorch-2.9.0-cuda12.8-cudnn9 \
  python -m src.main --experiment classifier
```

## Contributing

This is a personal research project, but contributions are welcome. Please:

1. Follow the architecture patterns in [architecture.md](docs/standards/architecture.md)
2. Add tests for all new code (minimum: unit + component)
3. Ensure all tests pass: `pytest -m "unit or component"`
4. Update documentation as needed

## License

[Add your license here]

## Acknowledgments

- PyTorch for deep learning framework
- Original implementations that inspired this refactored architecture
