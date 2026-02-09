# Architecture Specification

## Overview

This repository follows a **Vertical Slice + Base Class** architecture pattern for AI model training and experimentation. The design prioritizes code reusability, testability, and rapid experimentation for personal research.

## Design Principles

### 1. Vertical Slice Architecture

- Each experiment type (GAN, Diffusion, Classifier) is self-contained
- Changes to one experiment don't affect others
- Clear separation of concerns per experiment
- Easy to add new experiment types

### 2. Base Class Pattern

- Shared training logic in base classes to reduce duplication
- Consistent interface across all experiments
- Common utilities (logging, checkpointing, metrics) centralized
- Composition over inheritance where possible to avoid tight coupling
- See [Component Responsibilities](#component-responsibilities) for details

### 3. Testability First

- All code must be testable on CPU without GPU
- Four-tier testing strategy (unit, component, integration, smoke)
- Mock data and fixtures for fast testing
- See [Testing Strategy](#testing-strategy) for complete approach

### 4. Configuration Driven

- Experiments configured via JSON files or CLI arguments (see [Configuration Management](#configuration-management))
- CLI arguments override config file values, which override code defaults
- Reproducible experiments through configuration version control

## Directory Structure

```
ai-gen-image-stats/
├── src/
│   ├── __init__.py
│   ├── main.py                          # CLI entrypoint
│   │
│   ├── base/
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Base trainer class
│   │   ├── model.py                     # Base model interface/ABC
│   │   ├── dataloader.py                # Base dataloader interface
│   │   └── logger.py                    # Base logger interface
│   │
│   ├── experiments/                     # Vertical slices
│   │   ├── __init__.py
│   │   │
│   │   ├── gan/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py               # GAN-specific trainer (inherits base)
│   │   │   ├── model.py                 # GAN models (Generator, Discriminator)
│   │   │   ├── dataloader.py            # GAN-specific dataloader
│   │   │   ├── logger.py                # GAN-specific logging (e.g., generated images)
│   │   │   └── config.py                # GAN default configs
│   │   │
│   │   ├── diffusion/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py               # Diffusion-specific trainer
│   │   │   ├── model.py                 # Diffusion model (DDPM, etc.)
│   │   │   ├── dataloader.py            # Diffusion-specific dataloader
│   │   │   ├── logger.py                # Diffusion-specific logging
│   │   │   └── config.py                # Diffusion default configs
│   │   │
│   │   └── classifier/                  # Non-generative models
│   │       ├── __init__.py
│   │       ├── trainer.py               # Classification trainer
│   │       ├── models/
│   │       │   ├── __init__.py
│   │       │   ├── inceptionv3.py       # InceptionV3 wrapper
│   │       │   ├── resnet.py            # ResNet variants
│   │       │   └── custom.py            # Custom architectures
│   │       ├── dataloader.py            # Classification dataloader
│   │       ├── logger.py                # Classification metrics logging
│   │       ├── analyze_comparison.py    # Analysis tools
│   │       └── config.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── cli.py                       # CLI argument parsing
│   │   ├── config.py                    # Config loading (JSON/YAML)
│   │   ├── device.py                    # Device management (CPU/GPU)
│   │   └── metrics.py                   # Common metrics (FID, IS, PR-AUC, ROC-AUC)
│   │
│   └── data/
│       ├── __init__.py
│       ├── datasets.py                  # Dataset implementations
│       ├── transforms.py                # Common transformations
│       └── samplers.py                  # Custom samplers if needed
│
├── tests/                               # See Testing Strategy section for details
│   ├── base/                            # Mirror src/base/
│   ├── experiments/                     # Mirror src/experiments/
│   ├── utils/                           # Mirror src/utils/
│   ├── data/                            # Mirror src/data/
│   ├── integration/                     # Optional: Complex end-to-end tests
│   ├── smoke/                           # Optional: GPU-intensive tests
│   ├── fixtures/                        # Test fixtures, mock data
│   └── conftest.py                      # Shared pytest fixtures
│
├── configs/                             # Production configs
│   ├── gan/
│   │   ├── default.json
│   │   └── experiment_001.json
│   ├── diffusion/
│   │   ├── default.json
│   │   └── experiment_001.json
│   └── classifier/
│       ├── baseline.json
│       ├── with_synth.json
│       └── inceptionv3.json
│
├── outputs/                             # Training outputs (gitignored)
│   ├── checkpoints/
│   ├── logs/
│   └── generated/
│
├── models/                              # Pre-trained weights
├── data/                                # Dataset storage
├── requirements.txt
├── requirements-dev.txt                 # Test dependencies (pytest, etc.)
└── README.md
```

## Component Responsibilities

### Base Classes (`src/base/`)

**Purpose:** Provide common interfaces and shared functionality across all experiments.

**Key Components:**

- `trainer.py`: Abstract base trainer with common training loop, validation, checkpointing
- `model.py`: Model interface defining required methods (forward, loss, etc.)
- `dataloader.py`: Data loading interface with common preprocessing
- `logger.py`: Logging interface for metrics, images, and artifacts

### Experiment Slices (`src/experiments/`)

**Purpose:** Self-contained implementations for each research direction.

Each experiment module (GAN, Diffusion, Classifier) contains:

- **Trainer**: Experiment-specific training logic (inherits from base)
- **Model**: Architecture implementations
- **DataLoader**: Data loading for specific experiment needs
- **Logger**: Experiment-specific metrics and visualizations
- **Config**: Default configurations

**Experiment Isolation:**

- Each experiment can be developed independently
- No cross-dependencies between experiments
- Shared code only through base classes and utils

### Utilities and Data

**Utilities (`src/utils/`)**: Cross-cutting concerns

- CLI, config loading, device management, common metrics

**Data Pipeline (`src/data/`)**: Dataset management

- Dataset implementations, transforms, samplers

## CLI Interface

### Command Structure

```bash
python -m src.main --experiment <EXPERIMENT> [OPTIONS]
```

**Parameter Priority:** CLI arguments > Config file > Code defaults

### Usage Examples

```bash
# Train with config file
python -m src.main --experiment diffusion --config configs/diffusion/default.json

# Train with CLI arguments only
python -m src.main --experiment gan --epochs 100 --batch-size 64 --lr 0.0002

# Train with config + CLI overrides
python -m src.main --experiment classifier --model inceptionv3 \
  --config configs/classifier/baseline.json --batch-size 32

# Generate synthetic data
python -m src.main --experiment diffusion --mode generate \
  --checkpoint outputs/checkpoints/diffusion_best.pth --num-samples 1000
```

## Testing Strategy

### Overview

This project uses a **tiered testing strategy** that balances speed, coverage, and practical validation needs. Tests are organized into four tiers, each serving a specific purpose in the development workflow.

### Tier 1: Unit Tests (Fast - CPU Only)

**Target: < 100ms per test, run on every commit**

**Purpose:** Immediate feedback during development with zero GPU requirements.

**Scope:**

- Pure logic testing: Configuration parsing, CLI argument handling, metric calculations
- Component interfaces: Base class method signatures, abstract method enforcement
- Data transformations: Augmentations, normalization, tensor operations
- Utility functions: Device management, file I/O, logging helpers
- Model instantiation: Model creation without forward passes

**Key Principle:** No GPU required, minimal dependencies, fast feedback loop

### Tier 2: Component Tests (Medium - CPU with small data)

**Target: 1-5 seconds per test, run before push**

**Purpose:** Validate component behavior with minimal computation.

**Scope:**

- Model forward passes: Single batch through model on CPU with tiny input (e.g., 2x3x32x32)
- Loss calculations: Verify loss functions return expected shapes and ranges
- DataLoader functionality: Load 2-3 samples, verify batching and transforms
- Single training step: One batch through train loop (forward + backward + optimizer step)
- Checkpoint save/load: Verify state dict roundtrip consistency

**Key Principle:** Test component integration points without expensive computation

### Tier 3: Integration Tests (Slow - GPU Optional)

**Target: 10-60 seconds per test, run on CI/nightly**

**Purpose:** Verify components work together correctly in realistic workflows.

**Scope:**

- Mini training loop: 2-3 epochs with tiny dataset (10-20 images)
- End-to-end pipelines: Config → DataLoader → Model → Training → Checkpoint
- Experiment workflows: Full experiment slice execution with minimal settings
- Evaluation pipeline: Generate samples → Compute metrics (FID, IS, etc.)
- CLI interface: Test command parsing and execution paths

**Key Principle:** Validate end-to-end workflows with representative data flows

### Tier 4: Smoke Tests (Very Slow - GPU Required)

**Target: 5-15 minutes, run manually/weekly**

**Purpose:** Catch GPU-specific issues and performance regressions.

**Scope:**

- Real hardware validation: Train on actual GPU for a few epochs
- Memory usage: Verify batch sizes fit in GPU memory
- Performance benchmarks: Ensure no regression in training speed
- Generated sample quality: Visual inspection of GAN/diffusion outputs
- Full evaluation metrics: FID/IS on larger sample sets (100-500 images)

**Key Principle:** Real-world validation on target hardware

### Test Organization

**Directory Structure:**

```
tests/
├── base/                                # Mirror src/base/
│   ├── test_trainer.py
│   ├── test_model.py
│   └── test_logger.py
│
├── experiments/                         # Mirror src/experiments/
│   ├── gan/
│   │   ├── test_trainer.py
│   │   ├── test_model.py
│   │   └── test_dataloader.py
│   ├── diffusion/
│   │   ├── test_trainer.py
│   │   ├── test_model.py
│   │   └── test_dataloader.py
│   └── classifier/
│       ├── test_trainer.py
│       ├── test_models.py
│       └── test_dataloader.py
│
├── utils/                               # Mirror src/utils/
│   ├── test_cli.py
│   ├── test_config.py
│   └── test_metrics.py
│
├── data/                                # Mirror src/data/
│   ├── test_datasets.py
│   └── test_transforms.py
│
├── integration/                         # Optional: Complex end-to-end tests
│   ├── test_full_gan_workflow.py
│   ├── test_full_diffusion_workflow.py
│   └── test_classifier_pipeline.py
│
├── smoke/                               # Optional: GPU-intensive tests
│   ├── test_gpu_training.py
│   ├── test_performance.py
│   └── test_memory_usage.py
│
├── fixtures/                            # Test fixtures, mock data
│   ├── configs/
│   └── mock_data/
│
└── conftest.py                          # Shared pytest fixtures
```

**Layout Strategy:**

- **Primary Organization**: Tests mirror the `src/` directory structure for easy navigation
- **Marker-Based Tiers**: Each test file contains multiple test tiers organized by pytest markers
- **Optional Directories**: `integration/` and `smoke/` directories for complex multi-file tests (add only when needed)

**Test File Organization:**

Each test file contains tests from multiple tiers, organized top to bottom:

- **Unit tests** at the top: Fast, no dependencies, pure logic testing
- **Component tests** in the middle: Small data, minimal computation
- **Integration tests** at the bottom: Mini workflows, end-to-end validation

This keeps related tests together while allowing selective execution via markers.

**When to Add Optional Directories:**

Add `tests/integration/` and `tests/smoke/` when:

- Integration tests become complex multi-file orchestrations
- Smoke tests need special setup/teardown or CI/CD configuration
- You want to run entire test categories with directory-based commands
- Tests don't naturally fit the mirrored structure

Skip these directories if:

- Tests remain simple and component-scoped
- Markers provide sufficient organization
- You prefer keeping all related tests in one place

**Benefits of This Layout:**

- **Intuitive Navigation**: Easy to find tests for any source file
- **Maintenance Friendly**: 1-to-1 mapping between source and test files
- **IDE Support**: Modern IDEs work well with mirrored structures
- **Flexible Execution**: Run by directory, file, or marker as needed
- **No Ambiguity**: Clear ownership and organization

**Test Markers:**

Tests are tagged with pytest markers for selective execution:

- `unit`: Fast unit tests (CPU only)
- `component`: Component tests with small data
- `integration`: Integration tests with mini datasets
- `smoke`: Full workflow smoke tests (GPU preferred)
- `slow`: Tests that take > 10 seconds
- `gpu`: Tests requiring GPU hardware

**Running Tests:**

```bash
# Fast feedback during development (< 10 seconds total)
pytest -m unit

# Pre-commit validation (< 1 minute)
pytest -m "unit or component"

# CI pipeline (< 5 minutes)
pytest -m "not smoke"

# Full validation with GPU (weekly/manual)
pytest -m smoke --gpu
```

### Test Coverage Priorities

**Priority 1 (Must Have):**

- All base classes and interfaces
- Configuration loading and merging
- CLI argument parsing
- Data transformations
- Model instantiation

**Priority 2 (Should Have):**

- Training loop logic
- Loss calculations
- Metric computations
- Checkpoint save/load
- DataLoader functionality

**Priority 3 (Nice to Have):**

- End-to-end workflows
- GPU-specific behavior
- Performance benchmarks
- Generated sample quality

### When to Run Each Tier

**Development Workflow Integration:**

- **Unit Tests**: Every file save (IDE integration), every commit
- **Component Tests**: Before push, on pull requests
- **Integration Tests**: CI pipeline, before merging to main
- **Smoke Tests**: Before releases, weekly scheduled runs, manual validation

### Benefits of Tiered Strategy

**For Development:**

- Fast local feedback loop (< 10s for unit tests)
- Confidence before pushing code
- No GPU required for routine development
- Selective test execution based on context

**For CI/CD:**

- Efficient resource usage (most tests run on CPU)
- Quick feedback on pull requests
- Comprehensive validation before production
- Cost-effective testing strategy

**For Research:**

- Rapid iteration on new ideas
- Validation without blocking on GPU availability
- Performance regression detection
- Quality assurance for experiments

### Test Fixtures and Utilities

**Common Fixtures:**

- Mock datasets: Small in-memory datasets for fast testing
- Tiny batches: Minimal tensor shapes for component tests
- Device fixtures: CPU for regular tests, GPU for smoke tests (with skip on unavailable)
- Temporary directories: Isolated output locations for checkpoint/log testing
- Mock configs: Minimal configurations for each experiment type

**Test Utilities:**

- Assertion helpers: Shape checks, range validation, convergence detection
- Mock data generators: Synthetic images, labels, and annotations
- Performance profilers: Memory usage tracking, execution time measurement
- Reproducibility helpers: Fixed random seeds, deterministic operations

## Configuration Management

### Config File Format

JSON format for configuration files with the following structure:

```json
{
  "experiment": "classifier",
  "model": {
    "name": "inceptionv3",
    "pretrained": true,
    "num_classes": 2
  },
  "data": {
    "train_path": "data/train",
    "val_path": "data/val",
    "batch_size": 32,
    "num_workers": 4
  },
  "training": {
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "scheduler": "cosine"
  },
  "output": {
    "checkpoint_dir": "outputs/checkpoints",
    "log_dir": "outputs/logs"
  }
}
```

### Config Organization

- `configs/`: Production experiment configs (version controlled)
- `tests/fixtures/configs/`: Test configs with minimal settings
- One config per experiment variant for reproducibility

## Research Workflow

### Typical Workflow

1. **Generate Synthetic Data** (optional): Use diffusion experiment in generate mode
2. **Train Baseline Classifier**: Train on real data only
3. **Train with Synthetic Augmentation**: Train on real + synthetic data
4. **Compare Results**: Use analysis tools to evaluate performance differences

See [CLI Interface](#cli-interface) for command examples.

### Experiment Tracking

- All results saved to `outputs/` with timestamp and config hash
- Logs include: metrics CSV, tensorboard logs, model checkpoints
- Config file copied to output directory for reproducibility
- Generated samples saved for qualitative evaluation

## Integration with Existing Code

### Migration Path

Current code in `src/old/` will be refactored into new structure:

- `ddpm_train.py` → `src/experiments/diffusion/trainer.py`
- `ddpm.py` → `src/experiments/diffusion/model.py`
- `inception_v3.py`, `resnet.py` → `src/experiments/classifier/models/`
- `train.py` → `src/base/trainer.py` (common logic)
- `analyze_comparison.py` → `src/experiments/classifier/analyze_comparison.py`
- `stats.py` → `src/utils/metrics.py`
- `util.py` → `src/utils/` (split by concern)

### Pre-trained Model Weights

Existing model weights in `models/`:

- `inception_v3.pth`
- `resnet50.pth`
- `resnet101.pth`
- `wrn28_10_cifar10.pth`

These will be loaded by respective model implementations in `src/experiments/classifier/models/`.

## Benefits of This Architecture

### For Development

- ✓ Clear separation of concerns
- ✓ Easy to add new experiments
- ✓ Minimal code duplication
- ✓ Consistent interfaces

### For Testing

- ✓ Components testable in isolation
- ✓ Fast tests on CPU
- ✓ Easy to mock dependencies
- ✓ Clear test organization

### For Research

- ✓ Rapid experimentation
- ✓ Reproducible results
- ✓ Easy comparison of approaches
- ✓ Config-driven workflows

### For Maintenance

- ✓ Changes isolated to relevant slices
- ✓ Shared code centralized
- ✓ Self-documenting structure
- ✓ Easy to understand and navigate

## Anti-Patterns to Avoid

### Over-Abstraction

- Don't create base classes prematurely
- Refactor to base only when pattern emerges 2-3 times
- Keep base classes focused and minimal

### Cross-Slice Dependencies

- Experiments should not import from each other
- Share code through base classes or utils only
- Copy small amounts of code rather than creating complex dependencies

### Configuration Explosion

- Don't create too many config variants
- Use meaningful names for configs (not config1, config2)
- Document what each config variant tests

### Test Coupling

- Tests should not depend on each other
- Each test should be runnable independently
- Avoid shared mutable state in tests

## Future Extensions

This architecture supports adding:

- New generative models (VAE, Flow-based, etc.)
- New classifier architectures
- Multi-modal experiments
- Ensemble methods
- AutoML / hyperparameter optimization
- Distributed training

Each addition follows the same pattern: create a new vertical slice in `src/experiments/` that implements the base interfaces.
