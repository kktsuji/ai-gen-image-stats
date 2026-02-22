# Architecture Specification

## Overview

This repository follows a **Vertical Slice + Base Class** architecture pattern for AI model training and experimentation. The design prioritizes code reusability, testability, and rapid experimentation for personal research.

## Design Principles

### 1. Vertical Slice Architecture

- Each experiment type (Diffusion, Classifier) is self-contained
- Changes to one experiment don't affect others
- Clear separation of concerns per experiment
- Easy to add new experiment types (GAN planned for future)

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

- Experiments configured via YAML files (see [Configuration Management](#configuration-management))
- All configuration parameters must be specified in config files
- No CLI parameter overrides or code defaults
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
│   │   ├── data_preparation/
│   │   │   ├── __init__.py
│   │   │   ├── prepare.py               # Dataset scanning and splitting
│   │   │   ├── config.py                # Data preparation config validation
│   │   │   └── default.yaml             # Default data preparation configuration
│   │   │
│   │   ├── diffusion/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py               # Diffusion-specific trainer
│   │   │   ├── model.py                 # Diffusion model (DDPM, etc.)
│   │   │   ├── sampler.py               # Lightweight sampler for generation mode
│   │   │   ├── dataloader.py            # Diffusion-specific dataloader
│   │   │   ├── logger.py                # Diffusion-specific logging
│   │   │   ├── config.py                # Diffusion config validation
│   │   │   └── default.yaml             # Default diffusion configuration
│   │   │
│   │   └── classifier/                  # Classification models
│   │       ├── __init__.py
│   │       ├── trainer.py               # Classification trainer
│   │       ├── models/
│   │       │   ├── __init__.py
│   │       │   ├── inceptionv3.py       # InceptionV3 wrapper
│   │       │   └── resnet.py            # ResNet variants
│   │       ├── dataloader.py            # Classification dataloader
│   │       ├── logger.py                # Classification metrics logging
│   │       ├── analyze_comparison.py    # Analysis tools
│   │       ├── config.py                # Classifier config validation
│   │       └── default.yaml             # Default classifier configuration
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── cli.py                       # CLI argument parsing
│   │   ├── config.py                    # Config loading (YAML)
│   │   ├── device.py                    # Device management (CPU/GPU)
│   │   ├── logging.py                   # Logging configuration and setup
│   │   ├── metrics.py                   # Common metrics (FID, IS, PR-AUC, ROC-AUC)
│   │   ├── notification.py              # Slack webhook notifications
│   │   └── tensorboard.py               # TensorBoard utility functions (optional)
│   │
│   ├── deprecated/                      # Old code being phased out
│   │   ├── ddpm_train.py
│   │   ├── ddpm.py
│   │   ├── inception_v3.py
│   │   ├── resnet.py
│   │   ├── train.py
│   │   ├── analyze_comparison.py
│   │   ├── stats.py
│   │   ├── util.py
│   │   └── ...                          # Other deprecated files
│   │
│   └── data/
│       ├── __init__.py
│       ├── datasets.py                  # Dataset implementations
│       ├── transforms.py                # Common transformations
│       └── samplers.py                  # Custom data samplers
│
├── tests/                               # See Testing Strategy section for details
│   ├── base/                            # Mirror src/base/
│   ├── experiments/                     # Mirror src/experiments/
│   │   ├── classifier/
│   │   ├── data_preparation/
│   │   ├── diffusion/
│   │   └── gan/                         # Placeholder for future implementation
│   ├── utils/                           # Mirror src/utils/
│   ├── data/                            # Mirror src/data/
│   ├── integration/                     # Complex end-to-end tests
│   │   ├── test_classifier_pipeline.py
│   │   └── test_diffusion_pipeline.py
│   ├── fixtures/                        # Test fixtures, mock data
│   │   ├── configs/
│   │   ├── mock_data/
│   │   └── README.md
│   ├── conftest.py                      # Shared pytest fixtures
│   ├── test_main.py                     # Main CLI tests
│   └── test_infrastructure.py           # Infrastructure tests
│
├── configs/                             # User experiment configs (user folder)
│   └── (empty - users create their own configs here. gitignored)
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

**Currently Implemented:**

**Data Preparation Experiment** (`src/experiments/data_preparation/`):

- **Prepare**: Scans class directories, splits files into train/val with seeded shuffle, writes JSON split file
- **Config**: Configuration validation for classes, split ratio, seed, and output paths
- **default.yaml**: Default configuration template

**Diffusion Experiment** (`src/experiments/diffusion/`):

- **Trainer**: DDPM training logic with EMA support
- **Model**: U-Net architecture for diffusion models
- **Sampler**: Lightweight sampler for generation-only mode
- **DataLoader**: Dataset loading for diffusion training
- **Logger**: Training metrics and generated image logging
- **Config**: Configuration validation and default settings
- **default.yaml**: Default configuration template

**Classifier Experiment** (`src/experiments/classifier/`):

- **Trainer**: Classification training with validation
- **Models**: InceptionV3 and ResNet variants (50/101/152)
- **DataLoader**: Image classification dataset loading
- **Logger**: Training/validation metrics and confusion matrices
- **Config**: Configuration validation and default settings
- **default.yaml**: Default configuration template
- **analyze_comparison.py**: Tools for comparing classifier performance

**Planned:**

- **GAN Experiment**: Placeholder exists in CLI but not yet implemented

**Experiment Isolation:**

- Each experiment can be developed independently
- No cross-dependencies between experiments
- Shared code only through base classes and utils

### Utilities and Data

**Utilities (`src/utils/`)**: Cross-cutting concerns

- CLI, config loading, device management, common metrics, logging configuration, TensorBoard utilities

**Data Pipeline (`src/data/`)**: Dataset management

- `SplitFileDataset`: Loads train/val datasets from JSON split files produced by data_preparation

- Dataset implementations, transforms, samplers

## Logging Strategy

### Application Logging vs Metrics Logging

The project uses two complementary logging systems with distinct purposes:

**1. Application Logging** (Python `logging` library):

- **Purpose**: Runtime events, debugging, system messages, errors
- **Output**: Console + timestamped log files
- **Configuration**: `logging` section in YAML configs
- **Use Cases**:
  - Training start/end events
  - Model initialization
  - Checkpoint save/load operations
  - Device detection
  - Error and warning messages
  - Debug diagnostics (batch shapes, memory usage)

**2. Metrics Logging** (`BaseLogger` classes in `src/base/logger.py` and experiment-specific loggers):

- **Purpose**: Training metrics, evaluation results, generated artifacts
- **Output**: CSV files, PNG images, YAML hyperparams, and optionally TensorBoard event files
- **Configuration**: `logging.metrics` section in YAML configs
- **Use Cases**:
  - Loss curves (training/validation)
  - Accuracy metrics
  - Confusion matrices
  - Generated sample images
  - Denoising process visualizations
  - Evaluation metrics (FID, IS, etc.)

**Hybrid Metrics Strategy**: CSV logging is always active and provides lightweight, portable metrics records. TensorBoard is an optional layer on top, enabled via `logging.metrics.tensorboard.enabled: true`, providing interactive plots, image panels, and hyperparameter comparison. Both write the same data redundantly so either artifact alone is sufficient for reproducing results.

**Key Distinction**: Application logging tracks _what the system is doing_, while metrics logging tracks _how well the model is performing_.

### Logging Architecture

**Module-Level Loggers:**

Each module uses its own logger obtained via `logging.getLogger(__name__)`:

```python
# src/experiments/classifier/trainer.py
import logging

logger = logging.getLogger(__name__)

class ClassifierTrainer:
    def train(self):
        logger.info("Starting training for 100 epochs")
        logger.debug(f"Batch size: {self.batch_size}")
```

**Benefits:**

- Clear source identification in logs (`src.experiments.classifier.trainer`)
- Module-specific log level control
- Standard Python practice
- Easy to filter and search logs

**Centralized Configuration:**

Logging is configured once at application startup in `src/main.py`:

```python
from src.utils.logging import setup_logging, get_log_file_path

# Generate log file path with timestamp
log_file = get_log_file_path(
    output_base_dir=config["output"]["base_dir"],
    log_subdir=config["output"]["subdirs"]["logs"]
)

# Configure logging with settings from config
logging_config = config.get("logging", {})
logger = setup_logging(
    log_file=log_file,
    console_level=logging_config.get("console_level", "INFO"),
    file_level=logging_config.get("file_level", "DEBUG"),
    log_format=logging_config.get("format"),
    date_format=logging_config.get("date_format"),
    module_levels=logging_config.get("module_levels"),
)
```

### Configuration Format

Logging is configured in experiment YAML files:

```yaml
# Logging Configuration
logging:
  # Console output verbosity
  console_level: INFO # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

  # File output verbosity (can be more detailed than console)
  file_level: DEBUG

  # Log message format
  format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

  # Timestamp format
  date_format: "%Y-%m-%d %H:%M:%S"

  # Module-specific log levels (optional)
  module_levels:
    src.experiments.classifier.trainer: DEBUG
    src.base.trainer: INFO
    src.utils.device: WARNING
    torch: ERROR # Suppress verbose torch logging
```

**Configuration Notes:**

- If `logging` section is omitted, defaults to `INFO` for console, `DEBUG` for file
- `console_level` and `file_level` can differ to keep console clean while maintaining detailed file logs
- `module_levels` provides fine-grained control over specific components

### Log Levels and Usage Guidelines

**DEBUG (10)**: Detailed diagnostic information for development

```python
logger.debug(f"Batch shapes: input={x.shape}, target={y.shape}")
logger.debug(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
logger.debug(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
```

**INFO (20)**: Important progress and status messages (default)

```python
logger.info("Training started for 100 epochs")
logger.info(f"Epoch {epoch} completed - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
logger.info(f"Checkpoint saved: {checkpoint_path}")
logger.info(f"Using device: {device}")
```

**WARNING (30)**: Potential issues that don't prevent execution

```python
logger.warning(f"Learning rate very low: {lr:.2e}, training may be slow")
logger.warning("No validation dataset provided, skipping validation")
logger.warning(f"Checkpoint not found at {path}, starting from scratch")
```

**ERROR (40)**: Errors that affect results but don't crash

```python
logger.error(f"Failed to load checkpoint: {e}")
logger.error(f"Invalid configuration value: {key}={value}")
logger.exception("Exception during training step")  # Includes stack trace
```

**CRITICAL (50)**: Unrecoverable errors requiring termination

```python
logger.critical("Out of memory error, cannot continue training")
logger.critical(f"Required file not found: {path}")
```

### Logging Patterns

**Training Loop Logging:**

```python
# Epoch-level logging
logger.info(f"Epoch {epoch}/{num_epochs} started")

# Batch-level debug logging (periodic)
if batch_idx % log_interval == 0:
    logger.debug(
        f"Epoch [{epoch}/{num_epochs}] "
        f"Batch [{batch_idx}/{len(dataloader)}] "
        f"Loss: {loss.item():.4f}"
    )

# Epoch summary
logger.info(
    f"Epoch {epoch} completed - "
    f"Avg Loss: {avg_loss:.4f}, "
    f"Accuracy: {accuracy:.2f}%, "
    f"Time: {epoch_time:.1f}s"
)
```

**Checkpoint Operations:**

```python
logger.info(f"Saving checkpoint to: {checkpoint_path}")
logger.debug(f"  Epoch: {epoch}, Step: {global_step}")
logger.debug(f"  Metrics: {metrics}")
# ... save checkpoint ...
logger.info("Checkpoint saved successfully")
```

**Error Handling:**

```python
try:
    checkpoint = torch.load(checkpoint_path)
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    logger.debug(f"Checkpoint contains keys: {list(checkpoint.keys())}")
except FileNotFoundError:
    logger.warning(f"Checkpoint not found at {checkpoint_path}, starting fresh")
except Exception as e:
    logger.error(f"Failed to load checkpoint: {e}")
    logger.exception("Full exception details:")  # Includes traceback
    raise
```

### Log File Structure

**File Naming:** `log_YYYYMMDD_HHMMSS.log`

**Location:** `{output.base_dir}/{output.subdirs.logs}/log_<timestamp>.log`

**Example Directory Structure:**

```
outputs/
├── classifier-experiment/
│   ├── logs/
│   │   ├── log_20260217_143022.log    # Application logs
│   │   └── log_20260217_150130.log
│   ├── metrics/
│   │   ├── metrics.csv                 # Training metrics (BaseLogger)
│   │   └── hyperparams.yaml
│   ├── checkpoints/
│   │   ├── epoch_010.pt
│   │   ├── best_model.pth
│   │   ├── latest_checkpoint.pth
│   │   └── final_model.pth
│   ├── samples/                        # Generated images (metrics logging)
│   │   ├── epoch_010_samples.png
│   │   └── confusion_matrix_epoch_010.png
│   └── tensorboard/                    # TensorBoard event files (optional)
│       └── events.out.tfevents.*
└── diffusion-experiment/
    ├── logs/
    │   └── log_20260217_162045.log    # Application logs
    ├── metrics/
    │   └── metrics.csv                 # Training metrics
    ├── checkpoints/
    │   ├── epoch_050.pt
    │   ├── best_model.pth
    │   ├── latest_checkpoint.pth
    │   └── final_model.pth
    ├── generated/                      # Generated samples (metrics logging)
    │   └── samples_epoch_050.png
    └── tensorboard/                    # TensorBoard event files (optional)
        └── events.out.tfevents.*
```

### TensorBoard Integration

TensorBoard is an optional visualization layer over the CSV metrics system. It is disabled by default and requires no changes to existing code or workflows.

**Architecture:**

```
BaseLogger (Abstract)
    ├── log_metrics()          # Writes to CSV; optionally to TensorBoard
    ├── log_images()           # Saves PNG; optionally to TensorBoard
    ├── log_histogram()        # TensorBoard only (optional)
    └── log_hyperparams()      # TensorBoard HPARAMS tab

ClassifierLogger(BaseLogger)
    ├── _csv_writer            # Always active
    └── _tensorboard_writer    # Active when enabled in config

DiffusionLogger(BaseLogger)
    ├── _csv_writer            # Always active
    └── _tensorboard_writer    # Active when enabled in config
```

**Utility module** (`src/utils/tensorboard.py`): All TensorBoard calls are made through safe wrappers that accept `Optional[SummaryWriter]` and handle `None` gracefully. This means experiment loggers contain no conditional checks — TensorBoard is simply a no-op when disabled.

**Configuration** (`logging.metrics.tensorboard` in YAML):

```yaml
logging:
  metrics:
    csv:
      enabled: true # Always active; cannot be disabled
    tensorboard:
      enabled: false # Set to true to activate
      log_dir: null # null = auto (outputs/<name>/tensorboard)
      flush_secs: 30
      log_images: true
      log_histograms: false
      log_graph: false
```

**Graceful degradation**: If the `tensorboard` package is not installed and `enabled: true` is set, a warning is logged and training continues without TensorBoard. Zero breaking changes.

See [docs/features/20260218_tensorboard-user-guide.md](../features/20260218_tensorboard-user-guide.md) for usage details.

### Best Practices

**DO:**

- ✅ Use `logger.info()` for important events users should see
- ✅ Use `logger.debug()` for detailed diagnostics
- ✅ Use `logger.exception()` in exception handlers (includes stack trace)
- ✅ Include relevant context in log messages (epoch, batch, loss, etc.)
- ✅ Use f-strings for formatting: `logger.info(f"Epoch {epoch} completed")`
- ✅ Log at appropriate granularity (epoch summaries good, per-batch too verbose for INFO)
- ✅ Use module-level loggers: `logger = logging.getLogger(__name__)`

**DON'T:**

- ❌ Use `print()` statements for application logging
- ❌ Log sensitive information (API keys, passwords)
- ❌ Log excessive debug information at INFO level
- ❌ Create logger instances in functions (use module-level)
- ❌ Use string concatenation: `logger.info("Epoch " + str(epoch))` (use f-strings)
- ❌ Log every batch at INFO level (too verbose, use DEBUG)
- ❌ Confuse application logging with metrics logging

**When to Log:**

- **Always Log**: Training start/end, epoch completion, checkpoint save/load, errors
- **Usually Log**: Device selection, configuration summary, validation results
- **Debug Only**: Batch-level details, tensor shapes, memory usage, internal state
- **Never Log**: Every forward pass, every gradient update (too verbose)

### Integration with Testing

**Test Logging:**

Tests should use pytest's `caplog` fixture to verify logging behavior:

```python
import logging

def test_trainer_logs_epoch_start(trainer, caplog):
    """Trainer logs epoch start message at INFO level."""
    caplog.set_level(logging.INFO)
    trainer.train(num_epochs=1)

    assert "Epoch 1" in caplog.text
    assert "started" in caplog.text.lower()
```

**Test Fixtures:**

```python
# tests/conftest.py

@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages in tests."""
    caplog.set_level(logging.DEBUG)
    return caplog

@pytest.fixture
def temp_log_file(tmp_path):
    """Temporary log file for testing."""
    log_file = tmp_path / "test.log"
    return log_file
```

### Performance Considerations

**Minimal Overhead:**

- Logging adds ~1-5ms per message at INFO/DEBUG level
- File I/O is buffered, minimal impact on training
- DEBUG logs only computed if level is enabled
- No measurable impact on training speed (< 0.1%)

**Optimization Tips:**

- Use appropriate log levels (DEBUG only when needed)
- Avoid logging in tight inner loops
- Use conditional logging for expensive computations:
  ```python
  if logger.isEnabledFor(logging.DEBUG):
      logger.debug(f"Expensive computation: {expensive_function()}")
  ```

## CLI Interface

### Command Structure

```bash
python -m src.main <CONFIG_FILE>
```

**Configuration:** All parameters must be specified in the YAML configuration file. The experiment type is read from the config file's `experiment` field.

**Config File Locations:**

- `src/experiments/<experiment>/default.yaml`: Default configurations shipped with the project
- `configs/`: User folder for custom experiment configurations (user-managed)

### Strict Validation

The CLI enforces strict validation:

- All required fields must be present in the config file
- No default values are provided
- No CLI parameter overrides are allowed
- Clear error messages indicate missing or invalid fields

### Usage Examples

```bash
# Train classifier (using default config from src/experiments/classifier/)
python -m src.main src/experiments/classifier/default.yaml

# Train diffusion model (using default config from src/experiments/diffusion/)
python -m src.main src/experiments/diffusion/default.yaml

# Generate synthetic data with diffusion model (mode must be set in config)
python -m src.main configs/diffusion/generate.yaml

# Train with custom user config
python -m src.main configs/my_experiment.yaml
```

### Error Handling

```bash
# Missing config file
$ python -m src.main
Error: config_path is required

# File not found
$ python -m src.main nonexistent.yaml
Error: Config file not found: nonexistent.yaml

# Invalid config (missing required field)
$ python -m src.main incomplete_config.yaml
Error: Missing required field: model.name
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
│   ├── test_dataloader.py
│   └── test_logger.py
│
├── experiments/                         # Mirror src/experiments/
│   ├── diffusion/
│   │   ├── test_trainer.py
│   │   ├── test_model.py
│   │   ├── test_sampler.py
│   │   ├── test_dataloader.py
│   │   ├── test_logger.py
│   │   ├── test_config.py
│   │   └── test_trainer_sampler_integration.py
│   ├── classifier/
│   │   ├── test_trainer.py
│   │   ├── test_dataloader.py
│   │   ├── test_logger.py
│   │   ├── test_config.py
│   │   ├── test_analyze_comparison.py
│   │   └── models/
│   │       ├── test_inceptionv3.py
│   │       └── test_resnet.py
│   └── gan/                             # Placeholder for future tests
│       └── __init__.py
│
├── utils/                               # Mirror src/utils/
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_device.py
│   ├── test_logging.py
│   └── test_metrics.py
│
├── data/                                # Mirror src/data/
│   ├── test_datasets.py
│   ├── test_transforms.py
│   └── test_samplers.py
│
├── integration/                         # Complex end-to-end tests
│   ├── test_classifier_pipeline.py
│   └── test_diffusion_pipeline.py
│
├── fixtures/                            # Test fixtures, mock data
│   ├── configs/
│   │   ├── classifier/
│   │   ├── diffusion/
│   │   ├── classifier_minimal.yaml
│   │   ├── diffusion_minimal.yaml
│   │   └── gan_minimal.yaml
│   ├── mock_data/
│   └── README.md
│
├── conftest.py                          # Shared pytest fixtures
├── test_main.py                         # Main CLI tests
└── test_infrastructure.py               # Infrastructure tests
```

**Layout Strategy:**

- **Primary Organization**: Tests mirror the `src/` directory structure for easy navigation
- **Marker-Based Tiers**: Each test file contains multiple test tiers organized by pytest markers
- **Integration Directory**: Used for complex end-to-end workflow tests

**Test File Organization:**

Each test file contains tests from multiple tiers, organized top to bottom:

- **Unit tests** at the top: Fast, no dependencies, pure logic testing
- **Component tests** in the middle: Small data, minimal computation
- **Integration tests** at the bottom: Mini workflows, end-to-end validation

This keeps related tests together while allowing selective execution via markers.

**Integration Tests:**

The `tests/integration/` directory contains end-to-end workflow tests:

- Complex multi-component orchestration
- Full training/generation pipelines with minimal settings
- Tests that verify multiple components work together correctly

These tests use the `@pytest.mark.integration` marker and typically run longer than component tests.

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

YAML format for configuration files. Each experiment type has its own structure.

**Classifier Example:**

```yaml
experiment: classifier
mode: train
compute:
  device: cuda
  seed: 42
model:
  architecture:
    name: resnet50
    num_classes: 2
  initialization:
    pretrained: true
data:
  paths:
    train: data/0.Normal
    val: data/1.Abnormal
  loading:
    batch_size: 32
    num_workers: 4
training:
  optimizer:
    type: adam
    learning_rate: 0.001
  epochs: 10
output:
  base_dir: outputs
```

**Diffusion Example:**

```yaml
experiment: diffusion
mode: train # or generate
compute:
  device: cuda
  seed: null
model:
  architecture:
    image_size: 40
    in_channels: 3
    model_channels: 64
  diffusion:
    num_timesteps: 1000
    beta_schedule: cosine
data:
  paths:
    train: data/train
  loading:
    batch_size: 64
training:
  epochs: 100
  learning_rate: 0.0002
output:
  base_dir: outputs
```

### Config Organization

- `src/experiments/<experiment>/default.yaml`: Default configs shipped with each experiment
- `configs/`: User folder for custom experiment configurations (user-managed, can be empty)
- `tests/fixtures/configs/`: Minimal test configs for fast testing
- Each config must specify `experiment` field for routing to correct experiment implementation

## Research Workflow

### Typical Workflow

1. **Generate Synthetic Data** (optional):

   ```bash
   # Set mode: generate in config file
   python -m src.main configs/diffusion/generate.yaml
   ```

2. **Train Baseline Classifier**:

   ```bash
   # Train on real data only
   python -m src.main src/experiments/classifier/default.yaml
   ```

3. **Train with Synthetic Augmentation**:

   ```bash
   # Train on real + synthetic data (update data paths in config)
   python -m src.main configs/classifier/with_synth.yaml
   ```

4. **Compare Results**:
   - Use `analyze_comparison.py` tools in classifier experiment
   - Compare metrics logged in outputs directory

### Experiment Tracking

- All results saved to `outputs/` with timestamp and config hash
- Logs include: metrics CSV, tensorboard logs, model checkpoints
- Config file copied to output directory for reproducibility
- Generated samples saved for qualitative evaluation

## Integration with Existing Code

### Migration Status

Old code from `src/deprecated/` has been successfully refactored:

- ✓ `ddpm_train.py` → `src/experiments/diffusion/trainer.py`
- ✓ `ddpm.py` → `src/experiments/diffusion/model.py`
- ✓ `inception_v3.py`, `resnet.py` → `src/experiments/classifier/models/`
- ✓ `train.py` → `src/base/trainer.py` (common logic)
- ✓ `analyze_comparison.py` → `src/experiments/classifier/analyze_comparison.py`
- ✓ `stats.py` → `src/utils/metrics.py`
- ✓ `util.py` → `src/utils/` (split by concern)

The `src/deprecated/` directory is maintained for reference but is no longer used in the active codebase.

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
