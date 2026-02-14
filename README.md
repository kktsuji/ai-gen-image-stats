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

- **Configuration Management**: YAML-based configs with CLI overrides

- **Metrics & Evaluation**: FID, IS, PR-AUC, ROC-AUC for model evaluation

## Architecture

```
src/
├── base/              # Base classes for trainers, models, data loaders
├── experiments/       # Self-contained experiment implementations
│   ├── classifier/    # Classification models and training
│   │   ├── config.py       # Config loading and validation
│   │   ├── default.yaml    # Default configuration values
│   │   └── train.py        # Training logic
│   ├── diffusion/     # Diffusion models (DDPM)
│   │   ├── config.py       # Config loading and validation
│   │   ├── default.yaml    # Default configuration values
│   │   ├── train.py        # Training logic
│   │   └── generate.py     # Generation logic
│   └── gan/          # GAN models (planned)
├── utils/            # CLI, config, device management, metrics
└── data/             # Dataset implementations and transforms

tests/                # Mirror of src/ structure with test tiers
configs/              # User experiment configurations (YAML)
outputs/              # Training outputs (gitignored)
```

**Note:** Default configuration files (`default.yaml`) are colocated with their experiment code in `src/experiments/`. The `configs/` directory is for user-provided configurations and experiment-specific overrides.

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
python -m src.main configs/classifier/baseline.yaml

# Train with InceptionV3
python -m src.main configs/classifier/inceptionv3.yaml
```

### Training a Diffusion Model

```bash
# Train diffusion model
python -m src.main configs/diffusion/default.yaml
```

**Note:** All parameters must be specified in the configuration file. The CLI accepts only the config file path as a positional argument. CLI parameter overrides are not supported.

## Configuration

### Configuration Structure (V2)

The configuration is organized into logical sections:

- **`compute`**: Device and seed settings
- **`model`**: Model architecture, diffusion parameters, and conditioning
  - `architecture`: U-Net architecture parameters
  - `diffusion`: Diffusion process parameters
  - `conditioning`: Conditional generation settings
- **`data`**: Dataset paths, loading, and augmentation
  - `paths`: Train and validation data paths
  - `loading`: Batch size, workers, memory settings
  - `augmentation`: Data augmentation settings
- **`output`**: Output directory structure
  - `base_dir`: Base output directory
  - `subdirs`: Subdirectories for logs, checkpoints, samples, generated images
- **`training`**: Training-specific parameters
  - `optimizer`: Optimizer configuration
  - `scheduler`: Learning rate scheduler
  - `ema`: Exponential moving average
  - `checkpointing`: Checkpoint saving
  - `validation`: Validation settings
  - `visualization`: Training visualization
  - `performance`: Performance optimizations
  - `resume`: Resume training settings
- **`generation`**: Generation-specific parameters
  - `sampling`: Sampling parameters
  - `output`: Generation output settings

See [src/experiments/diffusion/default.yaml](src/experiments/diffusion/default.yaml) for a complete example.

### Diffusion Model Configuration (V2)

#### Common Parameters

These parameters apply to both training and generation modes:

```yaml
experiment: diffusion
mode: train # Options: train, generate

# Compute configuration
compute:
  device: cuda # Options: cuda, cpu, auto
  seed: 42 # Random seed for reproducibility (null for random)

# Model configuration
model:
  architecture:
    image_size: 40
    in_channels: 3
    model_channels: 64
    channel_multipliers: [1, 2, 4]
    use_attention: [false, false, true]

  diffusion:
    num_timesteps: 1000
    beta_schedule: cosine
    beta_start: 0.0001
    beta_end: 0.02

  conditioning:
    type: null # Options: null (unconditional), "class" (class-conditional)
    num_classes: null # Required if type="class"
    class_dropout_prob: 0.1

# Data configuration (V2 - image_size derived from model)
data:
  paths:
    train: data/train
    val: null

  loading:
    batch_size: 32
    num_workers: 4
    pin_memory: true
    shuffle_train: true
    drop_last: false

  augmentation:
    horizontal_flip: true
    rotation_degrees: 0
    color_jitter:
      enabled: false
      strength: 0.1

# Output configuration
output:
  base_dir: outputs
  subdirs:
    logs: logs
    checkpoints: checkpoints
    samples: samples
    generated: generated
```

#### Training Mode Configuration

Training mode includes comprehensive training parameters:

```yaml
mode: train

training:
  epochs: 200

  # Optimizer configuration
  optimizer:
    type: adam
    learning_rate: 0.0001
    weight_decay: 0.0
    betas: [0.9, 0.999]
    gradient_clip_norm: null

  # Scheduler configuration
  scheduler:
    type: null # Options: null, cosine, step, plateau
    T_max: auto # Auto sets to number of epochs
    eta_min: 1.0e-6

  # EMA configuration
  ema:
    enabled: true
    decay: 0.9999

  # Checkpointing configuration
  checkpointing:
    save_frequency: 10
    save_best_only: false
    save_optimizer: true

  # Validation configuration
  validation:
    enabled: true
    frequency: 1
    metric: loss

  # Visualization configuration (training-time sampling)
  visualization:
    enabled: true
    interval: 10
    num_samples: 8
    guidance_scale: 3.0

  # Performance optimizations
  performance:
    use_amp: false # Automatic mixed precision
    use_tf32: true # TF32 on Ampere+ GPUs
    cudnn_benchmark: true
    compile_model: false # PyTorch 2.0+ torch.compile

  # Resume training configuration
  resume:
    enabled: false
    checkpoint: null
    reset_optimizer: false
    reset_scheduler: false
```

#### Generation Mode Configuration

Generation mode is used to generate images from a trained checkpoint:

```yaml
mode: generate

generation:
  checkpoint: path/to/model.pth # Required for generate mode

  # Sampling configuration
  sampling:
    num_samples: 100
    guidance_scale: 3.0
    use_ema: true

  # Output configuration
  output:
    save_individual: true
    save_grid: true
    grid_nrow: 10
```

#### Complete Example

See [src/experiments/diffusion/default.yaml](src/experiments/diffusion/default.yaml) for a complete, documented example configuration.

#### Migration from V1

If you have configurations from before February 2026, see the [migration guide](docs/research/diffusion-config-migration-guide.md) for updating to the V2 structure. Key changes:

- `device` and `seed` moved to `compute` section
- Model parameters reorganized into `architecture`, `diffusion`, `conditioning`
- Data parameters reorganized into `paths`, `loading`, `augmentation`
- `image_size` only in `model.architecture` (derived for data)
- `return_labels` derived from `model.conditioning.type`
- Optimizer/scheduler parameters nested under `training`
- Visualization moved from `generation` to `training`

## Testing

This project uses a comprehensive **four-tier testing strategy** that balances speed, coverage, and practical validation needs. All code is testable on CPU without GPU requirements.

### Test Tiers

Our testing hierarchy from fastest to slowest:

#### Tier 1: Unit Tests (< 100ms per test)

- **Purpose**: Immediate feedback during development
- **Scope**: Pure logic, configuration parsing, utility functions, model instantiation
- **Requirements**: CPU only, no heavy dependencies
- **When to Run**: On every file save, every commit

#### Tier 2: Component Tests (1-5 seconds per test)

- **Purpose**: Validate component behavior with minimal computation
- **Scope**: Model forward passes, loss calculations, single training steps
- **Requirements**: CPU with tiny batches (2-3 samples)
- **When to Run**: Before push, on pull requests

#### Tier 3: Integration Tests (10-60 seconds per test)

- **Purpose**: Verify components work together in realistic workflows
- **Scope**: Mini training loops (2-3 epochs), end-to-end pipelines
- **Requirements**: CPU or GPU optional, mini datasets (10-20 images)
- **When to Run**: CI pipeline, before merging

#### Tier 4: Smoke Tests (5-15 minutes per test)

- **Purpose**: Catch GPU-specific issues and performance regressions
- **Scope**: Real GPU training, memory usage validation, quality checks
- **Requirements**: GPU required
- **When to Run**: Manually, weekly, before releases

### Running Tests

```bash
# Fast feedback during development (< 10 seconds total)
pytest -m unit

# Pre-commit validation (< 1 minute)
pytest -m "unit or component"

# CI pipeline - all except smoke tests (< 5 minutes)
pytest -m "not smoke"

# Full validation with GPU (weekly/manual)
pytest -m smoke

# Run all tests
pytest

# Run tests for specific module
pytest tests/utils/
pytest tests/experiments/classifier/

# Run with coverage report
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v -s

# Run specific test file
pytest tests/utils/test_config.py

# Run tests matching pattern
pytest -k "test_model"
```

### Test Organization

Tests mirror the `src/` directory structure for easy navigation:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── fixtures/                # Test data and configurations
│   ├── configs/            # Sample config files
│   └── mock_data/          # Mock datasets (auto-generated)
├── base/                    # Tests for src/base/
├── experiments/             # Tests for src/experiments/
│   ├── classifier/
│   ├── diffusion/
│   └── gan/
├── utils/                   # Tests for src/utils/
└── data/                    # Tests for src/data/
```

### Writing New Tests

When adding new code, follow these guidelines:

1. **Mirror the structure**: Place tests in `tests/` mirroring the `src/` path
2. **Use appropriate markers**: Tag tests with `@pytest.mark.unit`, `@pytest.mark.component`, etc.
3. **Start with unit tests**: Write fast unit tests first, then component tests
4. **Use fixtures**: Leverage shared fixtures from `conftest.py`
5. **Keep tests fast**: Unit tests < 100ms, component tests < 5s
6. **Test on CPU**: Ensure tests work without GPU (use `device_cpu` fixture)

Example test structure:

```python
import pytest
import torch

@pytest.mark.unit
def test_model_instantiation():
    """Unit test: Fast, no GPU required."""
    from src.experiments.classifier.models import ResNet50
    model = ResNet50(num_classes=2)
    assert model is not None

@pytest.mark.component
def test_model_forward_pass(device_cpu, mock_batch_tensor):
    """Component test: Small data, single forward pass."""
    from src.experiments.classifier.models import ResNet50
    model = ResNet50(num_classes=2).to(device_cpu)
    output = model(mock_batch_tensor.to(device_cpu))
    assert output.shape == (2, 2)  # batch_size=2, num_classes=2

@pytest.mark.integration
def test_training_loop(tmp_output_dir, mock_dataset_small):
    """Integration test: Mini training loop."""
    # Test 2-3 epochs with small dataset
    pass
```

### Available Fixtures

See [tests/conftest.py](tests/conftest.py) for all available fixtures:

- **Device fixtures**: `device_cpu`, `device_gpu`, `device_auto`
- **Data fixtures**: `mock_image_tensor`, `mock_batch_tensor`, `mock_dataset_small`, `mock_dataset_medium`
- **Directory fixtures**: `tmp_output_dir`, `tmp_data_dir`
- **Config fixtures**: `mock_config_classifier`, `mock_config_diffusion`, `mock_config_gan`

### Continuous Integration

Our CI pipeline runs tests in stages:

1. **Fast Check** (< 30s): Unit tests only
2. **Standard Check** (< 5min): Unit + Component + Integration tests
3. **Full Check** (manual): All tests including smoke tests on GPU

### Test Coverage

We prioritize coverage for:

- ✅ All base classes and interfaces (100% coverage goal)
- ✅ Configuration loading and CLI parsing
- ✅ Data transformations and loading
- ✅ Model instantiation and forward passes
- 🎯 Training loops and optimization logic
- 🎯 Metrics and evaluation code

## Project Status

**Current Phase**: Phase 1 - Project Foundation  
**Completed Steps**: 3/46  
**Last Updated**: 2026-02-10

This project is undergoing active refactoring according to the plan in [docs/research/refactor.md](docs/research/refactor.md).

### Recently Completed

- ✅ Step 1: Initial Project Setup
- ✅ Step 2: Base Directory Structure
- ✅ Step 3: Test Infrastructure Setup

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
