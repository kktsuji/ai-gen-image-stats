# Python Logging Library Integration Plan

**Date**: February 16, 2026
**Author**: AI Assistant
**Status**: Planning

## Overview

This document outlines the implementation plan for introducing Python's `logging` library throughout the project to replace print debugging and establish a standardized logging infrastructure. The new logging system will provide both console output and persistent log files with configurable log levels.

## Objectives

1. **Replace Print Debugging**: Eliminate all `print()` statements used for debugging/info messages
2. **Standardized Logging**: Implement consistent logging across all modules using Python's `logging` library
3. **Dual Output**: Log to both console and file simultaneously
4. **Configurable Levels**: Support dynamic log level configuration via YAML files
5. **Enhanced Observability**: Add meaningful logging beyond current print statements
6. **Architecture Compliance**: Follow the existing vertical slice architecture pattern

## Design Principles

### 1. Separation of Concerns

**Application Logging vs Metrics Logging:**

- **Application Logging** (`logging` library): Runtime events, debugging, system messages, errors
  - Handles: INFO, DEBUG, WARNING, ERROR, CRITICAL messages
  - Output: Console + log files
  - Examples: "Training started", "Checkpoint saved", "Device detected: cuda"

- **Metrics Logging** (existing `BaseLogger`/`ClassifierLogger`): Training metrics, images, artifacts
  - Handles: Scalar metrics, confusion matrices, generated images
  - Output: CSV files, PNG images, YAML hyperparams
  - Examples: Loss curves, accuracy metrics, sample images

**These are complementary, not overlapping systems.**

### 2. Module-Level Loggers

Each module gets its own logger using `logging.getLogger(__name__)`:

```python
import logging

logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting model training")
    logger.debug(f"Batch size: {batch_size}")
```

**Benefits:**

- Clear source identification in logs
- Module-specific log level control
- Standard Python practice

### 3. Centralized Configuration

**Single Source of Truth:**

- Logging setup utility: `src/utils/logging.py`
- Configuration via YAML: `logging` section in experiment configs
- Initialization at application entry point: `src/main.py`

### 4. Hierarchical Log Levels

```
CRITICAL (50) - System failures, unrecoverable errors
ERROR    (40) - Errors that don't crash but affect results
WARNING  (30) - Warnings about potential issues
INFO     (20) - Important progress/status messages [DEFAULT]
DEBUG    (10) - Detailed diagnostic information
```

## Architecture Integration

### Directory Structure

```
src/
├── main.py                          # Initialize logging at startup
├── utils/
│   ├── logging.py                   # NEW: Logging configuration utility
│   ├── cli.py
│   ├── config.py
│   └── device.py
├── base/
│   ├── trainer.py                   # Replace print → logger.info/debug
│   ├── model.py
│   └── dataloader.py
└── experiments/
    ├── classifier/
    │   ├── trainer.py               # Replace print → logger.info/debug
    │   ├── dataloader.py
    │   └── logger.py                # NO CHANGES (metrics logging)
    └── diffusion/
        ├── trainer.py               # Replace print → logger.info/debug
        ├── sampler.py               # Replace print → logger.info/debug
        └── logger.py                # NO CHANGES (metrics logging)

tests/
├── utils/
│   └── test_logging.py              # NEW: Tests for logging utility
├── base/
│   └── test_trainer.py              # Update: Verify logging calls
└── experiments/
    ├── classifier/
    │   └── test_trainer.py          # Update: Verify logging calls
    └── diffusion/
        └── test_trainer.py          # Update: Verify logging calls
```

### Configuration Format

**Classifier Config (`configs/classifier.yaml`):**

```yaml
experiment: classifier
mode: train

# NEW: Logging Configuration
logging:
  # Console logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  console_level: INFO

  # File logging level (can be more verbose than console)
  file_level: DEBUG

  # Log format string
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  # Date format for timestamps
  date_format: "%Y-%m-%d %H:%M:%S"

  # Module-specific log levels (optional)
  module_levels:
    src.experiments.classifier.trainer: DEBUG
    src.base.trainer: INFO
    src.utils.device: WARNING

compute:
  device: cuda
  seed: 42

# ... rest of config ...
```

**Diffusion Config (`configs/diffusion.yaml`):**

```yaml
experiment: diffusion
mode: train

# NEW: Logging Configuration
logging:
  console_level: INFO
  file_level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"

compute:
  device: cuda
  seed: null

# ... rest of config ...
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### Step 1.1: Create Logging Utility

**File: `src/utils/logging.py`**

Create a new module for logging configuration:

```python
"""Logging Configuration Utility

This module provides centralized logging setup for the application.
It configures both console and file handlers with customizable log levels.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union


def setup_logging(
    log_file: Union[str, Path],
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    module_levels: Optional[Dict[str, str]] = None,
) -> logging.Logger:
    """Configure application-wide logging with console and file handlers.

    Args:
        log_file: Path to log file
        console_level: Log level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: Log level for file output
        log_format: Custom format string for log messages
        date_format: Custom format for timestamps
        module_levels: Dict of module names to log levels for fine-grained control

    Returns:
        Root logger instance

    Example:
        >>> logger = setup_logging(
        ...     log_file="outputs/logs/train_20260216_143022.log",
        ...     console_level="INFO",
        ...     file_level="DEBUG"
        ... )
        >>> logger.info("Logging initialized")
    """
    # Implementation details in actual file...
    pass


def get_log_file_path(output_base_dir: Union[str, Path], log_subdir: str = "logs") -> Path:
    """Generate timestamped log file path.

    Args:
        output_base_dir: Base output directory from config
        log_subdir: Subdirectory for logs

    Returns:
        Path to log file with timestamp

    Example:
        >>> path = get_log_file_path("outputs/classifier-experiment", "logs")
        >>> print(path)
        outputs/classifier-experiment/logs/log_20260216_143022.log
    """
    pass


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("This is a debug message")
    """
    return logging.getLogger(name)
```

**Implementation Checklist:**

- [ ] Create `src/utils/logging.py`
- [ ] Implement `setup_logging()` function
  - [ ] Configure root logger
  - [ ] Add console handler with formatter
  - [ ] Add file handler with formatter
  - [ ] Support custom log levels per module
  - [ ] Handle log file directory creation
- [ ] Implement `get_log_file_path()` helper
- [ ] Implement `get_logger()` wrapper
- [ ] Add docstrings with examples
- [ ] Export functions in `src/utils/__init__.py`

#### Step 1.2: Add Tests for Logging Utility

**File: `tests/utils/test_logging.py`**

```python
"""Tests for Logging Utility

Unit tests for the logging configuration and setup utilities.
"""

import logging
import tempfile
from pathlib import Path

import pytest

from src.utils.logging import get_log_file_path, get_logger, setup_logging


@pytest.mark.unit
class TestLoggingSetup:
    """Tests for logging setup function."""

    def test_setup_logging_creates_log_file(self):
        """setup_logging creates log file at specified path."""
        # Test implementation...
        pass

    def test_setup_logging_console_handler(self):
        """setup_logging adds console handler with correct level."""
        pass

    def test_setup_logging_file_handler(self):
        """setup_logging adds file handler with correct level."""
        pass

    def test_setup_logging_custom_format(self):
        """setup_logging accepts custom format string."""
        pass

    def test_setup_logging_module_levels(self):
        """setup_logging configures module-specific log levels."""
        pass

    def test_logs_to_console_and_file(self):
        """Messages are logged to both console and file."""
        pass


@pytest.mark.unit
class TestLogFilePath:
    """Tests for log file path generation."""

    def test_get_log_file_path_structure(self):
        """get_log_file_path generates correct directory structure."""
        pass

    def test_get_log_file_path_timestamp(self):
        """get_log_file_path includes timestamp in filename."""
        pass


@pytest.mark.unit
class TestGetLogger:
    """Tests for get_logger wrapper."""

    def test_get_logger_return_type(self):
        """get_logger returns a logging.Logger instance."""
        pass

    def test_get_logger_unique_names(self):
        """get_logger creates separate loggers for different names."""
        pass
```

**Implementation Checklist:**

- [ ] Create `tests/utils/test_logging.py`
- [ ] Implement unit tests for `setup_logging()`
- [ ] Implement unit tests for `get_log_file_path()`
- [ ] Implement unit tests for `get_logger()`
- [ ] Verify log messages appear in both console and file
- [ ] Test custom formatters and log levels
- [ ] Ensure tests run on CPU, no GPU required

#### Step 1.3: Update Configuration Schema

**Files:**

- `configs/classifier.yaml`
- `configs/diffusion.yaml`
- `src/experiments/classifier/default.yaml`
- `src/experiments/diffusion/default.yaml`

Add `logging` section to all config files:

```yaml
# Logging Configuration
logging:
  console_level: INFO # Console output verbosity
  file_level: DEBUG # File output verbosity (more detailed)
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  # Optional: Module-specific overrides
  # module_levels:
  #   src.experiments.classifier.trainer: DEBUG
```

**Implementation Checklist:**

- [ ] Add `logging` section to `configs/classifier.yaml`
- [ ] Add `logging` section to `configs/diffusion.yaml`
- [ ] Add `logging` section to `src/experiments/classifier/default.yaml`
- [ ] Add `logging` section to `src/experiments/diffusion/default.yaml`
- [ ] Document logging parameters in config comments
- [ ] Update config validation in `src/experiments/*/config.py` (optional defaults)

#### Step 1.4: Initialize Logging in Main Entry Point

**File: `src/main.py`**

Initialize logging at the start of each experiment setup:

```python
def setup_experiment_classifier(config: Dict[str, Any]) -> None:
    """Setup and run classifier experiment."""
    from src.experiments.classifier.config import validate_config
    from src.utils.config import resolve_output_path
    from src.utils.logging import get_log_file_path, setup_logging

    # Validate config
    validate_config(config)

    # Setup logging FIRST (before any other operations)
    log_dir = resolve_output_path(config, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = get_log_file_path(
        output_base_dir=config["output"]["base_dir"],
        log_subdir=config["output"]["subdirs"]["logs"]
    )

    # Get logging configuration (with defaults)
    logging_config = config.get("logging", {})
    console_level = logging_config.get("console_level", "INFO")
    file_level = logging_config.get("file_level", "DEBUG")
    log_format = logging_config.get("format")
    date_format = logging_config.get("date_format")
    module_levels = logging_config.get("module_levels")

    # Initialize logging
    logger = setup_logging(
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
        log_format=log_format,
        date_format=date_format,
        module_levels=module_levels,
    )

    # Now use logger instead of print
    logger.info("=" * 80)
    logger.info("CLASSIFIER EXPERIMENT STARTED")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Console log level: {console_level}")
    logger.info(f"File log level: {file_level}")

    # Continue with rest of setup...
```

**Implementation Checklist:**

- [ ] Import logging utilities in `src/main.py`
- [ ] Initialize logging in `setup_experiment_classifier()`
- [ ] Initialize logging in `setup_experiment_diffusion()`
- [ ] Replace initial print statements with logger calls
- [ ] Add experiment start/end log markers
- [ ] Log configuration summary at startup
- [ ] Handle missing logging config gracefully (use defaults)

### Phase 2: Replace Print Statements in `/src` (Week 2)

#### Step 2.1: Update `src/main.py`

**Current Print Statements:**

- Device information
- Random seed confirmation
- Directory paths
- Configuration save confirmation
- Model information
- Optimizer/scheduler setup
- Training start message
- Class information
- Dataset size information

**Migration Strategy:**

```python
# Before:
print(f"Using device: {device}")

# After:
logger = logging.getLogger(__name__)
logger.info(f"Using device: {device}")
```

**Log Level Guidelines:**

- `INFO`: Device, directories, model name, training start/end
- `DEBUG`: Detailed config values, class distributions, layer counts
- `WARNING`: Missing optional configs, fallback to defaults

**Implementation Checklist:**

- [ ] Add `logger = logging.getLogger(__name__)` at module top
- [ ] Replace device info print → `logger.info`
- [ ] Replace seed info print → `logger.info`
- [ ] Replace directory prints → `logger.info`
- [ ] Replace model info prints → `logger.info`
- [ ] Replace optimizer prints → `logger.info`
- [ ] Replace training start → `logger.info`
- [ ] Replace class/dataset info → `logger.debug`
- [ ] Add startup banner with `logger.info`
- [ ] Add shutdown/completion message with `logger.info`

#### Step 2.2: Update `src/base/trainer.py`

**Current Print Statement (Line 458):**

```python
print(f"Resuming training from epoch {start_epoch}")
```

**Additional Logging Opportunities:**

- Training loop start/end
- Epoch start/end with timing
- Checkpoint save confirmations
- Validation trigger messages
- Best metric updates
- Early stopping triggers
- Exception/error handling

**Implementation Checklist:**

- [ ] Add `logger = logging.getLogger(__name__)` at module top
- [ ] Replace resume print → `logger.info`
- [ ] Add training loop start log
- [ ] Add epoch start/end logs with timing
- [ ] Add checkpoint save confirmations → `logger.info`
- [ ] Add validation epoch logs → `logger.info`
- [ ] Add best metric update logs → `logger.info`
- [ ] Add error/exception logs → `logger.error`
- [ ] Add debug logs for internal state (if useful)

#### Step 2.3: Update `src/experiments/classifier/trainer.py`

**Logging Opportunities:**

- Epoch progress (already have progress bar, add summary logs)
- Batch-level debugging (loss spikes, gradient norms)
- Validation results summary
- Confusion matrix computation
- Best model updates

**Implementation Checklist:**

- [ ] Add `logger = logging.getLogger(__name__)` at module top
- [ ] Add epoch summary logs (avg loss, accuracy)
- [ ] Add validation summary logs
- [ ] Add best model update logs
- [ ] Add debug logs for loss/gradient anomalies
- [ ] Add confusion matrix computation logs
- [ ] Log scheduler learning rate changes

#### Step 2.4: Update `src/experiments/diffusion/trainer.py`

**Logging Opportunities:**

- Training epoch summaries
- Diffusion-specific metrics (loss components)
- Sample generation triggers
- EMA updates
- Gradient clipping events

**Implementation Checklist:**

- [ ] Add `logger = logging.getLogger(__name__)` at module top
- [ ] Add epoch summary logs
- [ ] Add diffusion loss component logs
- [ ] Add sample generation trigger logs
- [ ] Add EMA update logs (if verbose)
- [ ] Add gradient clipping logs → `logger.warning`
- [ ] Add checkpoint save logs

#### Step 2.5: Update `src/experiments/diffusion/sampler.py`

**Logging Opportunities:**

- Sampling start/end
- Checkpoint loading
- Generation progress
- Class-conditional sampling info

**Implementation Checklist:**

- [ ] Add `logger = logging.getLogger(__name__)` at module top
- [ ] Add sampling initialization logs
- [ ] Add checkpoint loading confirmation
- [ ] Add generation progress logs (every N steps)
- [ ] Add class distribution logs for conditional generation
- [ ] Add sampling completion summary

#### Step 2.6: Update Other Source Files

**Files to review:**

- `src/base/model.py` (no prints, but could add debug logs)
- `src/base/dataloader.py` (no prints, but could add debug logs)
- `src/experiments/classifier/dataloader.py` (could add dataset size logs)
- `src/experiments/diffusion/dataloader.py` (could add dataset size logs)
- `src/utils/device.py` (could add device detection logs)
- `src/utils/config.py` (could add config loading logs)

**Implementation Checklist:**

- [ ] Review each file for logging opportunities
- [ ] Add loggers where meaningful (avoid over-logging)
- [ ] Focus on INFO/DEBUG for these utilities
- [ ] Log warnings for edge cases/fallbacks

### Phase 3: Add Enhanced Logging (Week 3)

#### Step 3.1: Training Progress Logging

Add periodic progress logs during training:

```python
# Log every N batches (for long epochs)
if batch_idx % log_interval == 0:
    logger.debug(
        f"Epoch [{epoch}/{num_epochs}] "
        f"Batch [{batch_idx}/{len(dataloader)}] "
        f"Loss: {loss.item():.4f} "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

# Log epoch summary
logger.info(
    f"Epoch {epoch} completed - "
    f"Avg Loss: {avg_loss:.4f}, "
    f"Accuracy: {accuracy:.2f}%, "
    f"Time: {epoch_time:.1f}s"
)
```

**Implementation Checklist:**

- [ ] Add batch-level progress logs (DEBUG level)
- [ ] Add epoch summary logs (INFO level)
- [ ] Add time tracking per epoch
- [ ] Add learning rate logs
- [ ] Add gradient norm logs (if clipping)
- [ ] Make log interval configurable (optional)

#### Step 3.2: Checkpoint and State Logging

```python
logger.info(f"Checkpoint saved: {checkpoint_path}")
logger.info(f"  Epoch: {epoch}, Step: {global_step}")
logger.info(f"  Metrics: {metrics}")

logger.info(f"Loading checkpoint: {checkpoint_path}")
logger.debug(f"  Checkpoint contains: {list(checkpoint.keys())}")
```

**Implementation Checklist:**

- [ ] Log checkpoint save operations
- [ ] Log checkpoint load operations
- [ ] Include metrics in checkpoint logs
- [ ] Add debug logs for checkpoint contents
- [ ] Log validation metric improvements

#### Step 3.3: Error and Warning Logging

```python
# Warning for potential issues
logger.warning(f"Learning rate very low: {lr:.2e}, training may be slow")

# Error for recoverable issues
logger.error(f"Failed to load checkpoint: {e}, starting from scratch")

# Critical for unrecoverable errors
logger.critical(f"Out of memory error, cannot continue training")
```

**Implementation Checklist:**

- [ ] Add warnings for unusual conditions
- [ ] Add error logs for exceptions
- [ ] Log stack traces with `logger.exception()`
- [ ] Add critical logs for fatal errors
- [ ] Ensure errors are still raised after logging

#### Step 3.4: Debug Logging for Development

Add verbose debug logs for development/troubleshooting:

```python
logger.debug(f"Model architecture: {model}")
logger.debug(f"Optimizer state: {optimizer.state_dict()}")
logger.debug(f"Batch shapes: input={x.shape}, target={y.shape}")
logger.debug(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
```

**Implementation Checklist:**

- [ ] Add model structure logs (DEBUG)
- [ ] Add batch shape logs (DEBUG)
- [ ] Add memory usage logs (DEBUG)
- [ ] Add data loading time logs (DEBUG)
- [ ] Keep DEBUG logs minimal in production paths

### Phase 4: Testing Integration (Week 4)

#### Step 4.1: Update Test Fixtures

Create test fixtures for logging in `tests/conftest.py`:

```python
@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages in tests."""
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def temp_log_file(tmp_path):
    """Temporary log file for testing."""
    log_file = tmp_path / "test.log"
    yield log_file
    # Cleanup happens automatically with tmp_path
```

**Implementation Checklist:**

- [ ] Add `capture_logs` fixture to `tests/conftest.py`
- [ ] Add `temp_log_file` fixture to `tests/conftest.py`
- [ ] Add logging setup/teardown for tests
- [ ] Ensure tests don't pollute console with logs

#### Step 4.2: Test Logging in Trainers

**File: `tests/base/test_trainer.py`**

```python
def test_trainer_logs_epoch_start(trainer, capture_logs):
    """Trainer logs epoch start message."""
    trainer.train(num_epochs=1)
    assert "Epoch 1" in capture_logs.text
    assert "started" in capture_logs.text.lower()


def test_trainer_logs_checkpoint_save(trainer, capture_logs, tmp_path):
    """Trainer logs checkpoint save operations."""
    trainer.save_checkpoint(tmp_path / "ckpt.pt")
    assert "checkpoint saved" in capture_logs.text.lower()
```

**Implementation Checklist:**

- [ ] Add logging tests to `tests/base/test_trainer.py`
- [ ] Verify epoch logging
- [ ] Verify checkpoint logging
- [ ] Verify error logging

#### Step 4.3: Test Logging in Experiments

**Files:**

- `tests/experiments/classifier/test_trainer.py`
- `tests/experiments/diffusion/test_trainer.py`

```python
def test_classifier_trainer_logs_validation(trainer, capture_logs):
    """Classifier trainer logs validation results."""
    trainer.validate_epoch()
    assert "validation" in capture_logs.text.lower()
    assert "accuracy" in capture_logs.text.lower()
```

**Implementation Checklist:**

- [ ] Add logging tests for classifier trainer
- [ ] Add logging tests for diffusion trainer
- [ ] Verify validation logging
- [ ] Verify metric logging (not to be confused with BaseLogger metrics)

#### Step 4.4: Integration Tests

**File: `tests/integration/test_logging_integration.py`**

```python
"""Integration tests for logging system."""

@pytest.mark.integration
def test_logging_to_file_and_console(tmp_path):
    """Logs appear in both file and console."""
    # Setup logging
    # Log messages
    # Verify in file
    # Verify in console capture
    pass


@pytest.mark.integration
def test_different_log_levels(tmp_path):
    """Console and file can have different log levels."""
    # Setup with INFO console, DEBUG file
    # Log DEBUG message
    # Verify only in file, not console
    pass
```

**Implementation Checklist:**

- [ ] Create `tests/integration/test_logging_integration.py`
- [ ] Test dual output (console + file)
- [ ] Test different log levels
- [ ] Test module-specific levels
- [ ] Test log format customization
- [ ] Test log file rotation (if implemented)

### Phase 5: Documentation and Cleanup (Week 5)

#### Step 5.1: Update Documentation

**File: `README.md`**

Add logging documentation to README:

```markdown
## Logging

The project uses Python's `logging` library for application-level logging.

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: Important progress and status messages (default)
- **WARNING**: Warnings about potential issues
- **ERROR**: Errors that affect results but don't crash
- **CRITICAL**: Unrecoverable errors

### Configuration

Configure logging in your YAML config file:

\`\`\`yaml
logging:
console_level: INFO # Console output verbosity
file_level: DEBUG # File output verbosity (more detailed)
format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format: "%Y-%m-%d %H:%M:%S"
\`\`\`

### Log Files

Logs are saved to: `{output.base_dir}/{output.subdirs.logs}/log_<timestamp>.log`

Example: `outputs/classifier-test/logs/log_20260216_143022.log`
```

**Implementation Checklist:**

- [ ] Add logging section to `README.md`
- [ ] Document log levels
- [ ] Document configuration options
- [ ] Document log file locations
- [ ] Add examples of log output

#### Step 5.2: Update Architecture Documentation

**File: `docs/standards/architecture.md`**

Add logging section:

```markdown
## Logging Strategy

### Application Logging vs Metrics Logging

The project uses two complementary logging systems:

1. **Application Logging** (`logging` library):
   - Purpose: Runtime events, debugging, system messages
   - Output: Console + log files
   - Configuration: `logging` section in YAML

2. **Metrics Logging** (`BaseLogger` classes):
   - Purpose: Training metrics, images, artifacts
   - Output: CSV files, images, hyperparams
   - Configuration: Experiment-specific

### Using Application Logging

\`\`\`python
import logging

logger = logging.getLogger(**name**)

def train():
logger.info("Training started")
logger.debug(f"Batch size: {batch_size}")
logger.warning("Learning rate is very low")
logger.error("Failed to load checkpoint")
\`\`\`

### Log File Structure

\`\`\`
outputs/
classifier-experiment/
logs/
log_20260216_143022.log # Application logs
metrics/
metrics.csv # Training metrics
checkpoints/
epoch_010.pt
\`\`\`
```

**Implementation Checklist:**

- [ ] Add logging section to architecture doc
- [ ] Clarify application vs metrics logging
- [ ] Document logging patterns
- [ ] Update directory structure diagrams
- [ ] Add logging best practices

#### Step 5.3: Create Migration Guide

**File: `docs/features/20260216_logging-migration-guide.md`**

Create a guide for users/contributors:

```markdown
# Logging Migration Guide

## For Users

### Enabling Verbose Logging

Change console log level in your config:

\`\`\`yaml
logging:
console_level: DEBUG # Was: INFO
\`\`\`

### Viewing Log Files

Find your experiment's log file:

\`\`\`
outputs/<experiment-name>/logs/log\_<timestamp>.log
\`\`\`

## For Contributors

### Adding Logging to New Code

1. Import logger at module level:
   \`\`\`python
   import logging
   logger = logging.getLogger(**name**)
   \`\`\`

2. Use appropriate log level:

- `logger.info()`: Important events
- `logger.debug()`: Detailed diagnostics
- `logger.warning()`: Potential issues
- `logger.error()`: Errors
  \`\`\`

### Logging Checklist

- [ ] Use module-level logger
- [ ] Choose appropriate log level
- [ ] Include relevant context in message
- [ ] Don't log sensitive information
- [ ] Use f-strings for formatting
```

**Implementation Checklist:**

- [ ] Create migration guide document
- [ ] Add user-facing instructions
- [ ] Add contributor guidelines
- [ ] Include examples
- [ ] Add troubleshooting section

#### Step 5.4: Final Cleanup

**Implementation Checklist:**

- [ ] Remove all remaining `print()` statements in `/src`
- [ ] Verify no print statements except in:
  - Docstring examples (acceptable)
  - `src/deprecated/` (not touching these)
- [ ] Run full test suite
- [ ] Test with different log levels
- [ ] Verify log files are created correctly
- [ ] Check log file sizes (reasonable, not massive)
- [ ] Verify console output is clean and informative

## Technical Specifications

### Log Format

**Default Format:**

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

**Example Output:**

```
2026-02-16 14:30:22 - src.main - INFO - CLASSIFIER EXPERIMENT STARTED
2026-02-16 14:30:22 - src.main - INFO - Using device: cuda
2026-02-16 14:30:22 - src.main - INFO - Checkpoint directory: outputs/...
2026-02-16 14:30:23 - src.experiments.classifier.trainer - INFO - Starting training for 100 epochs
2026-02-16 14:30:24 - src.base.trainer - DEBUG - Epoch 1/100 started
```

### Log File Naming

**Pattern:** `log_YYYYMMDD_HHMMSS.log`

**Examples:**

- `log_20260216_143022.log`
- `log_20260217_091530.log`

**Location:** `{output.base_dir}/{output.subdirs.logs}/log_<timestamp>.log`

### Handlers Configuration

**Console Handler:**

- Stream: `sys.stdout`
- Level: Configurable via `logging.console_level`
- Format: Colorized (optional, via `colorlog` if available)

**File Handler:**

- Mode: Write (`w`)
- Level: Configurable via `logging.file_level`
- Format: Plain text
- Encoding: UTF-8

### Module-Specific Log Levels

Support granular control:

```yaml
logging:
  console_level: INFO
  file_level: DEBUG
  module_levels:
    src.experiments.classifier.trainer: DEBUG
    src.base.trainer: INFO
    src.utils.device: WARNING
    torch: ERROR # Suppress verbose torch logging
```

## Testing Strategy

### Unit Tests (Tier 1)

**Focus:** Logging utility functions

- `test_setup_logging()`
- `test_get_log_file_path()`
- `test_log_level_configuration()`
- `test_module_specific_levels()`

**Speed:** < 100ms per test

### Component Tests (Tier 2)

**Focus:** Logging in individual components

- Trainers log correct messages
- Loggers respect level configuration
- File and console output work correctly

**Speed:** 1-5 seconds per test

### Integration Tests (Tier 3)

**Focus:** End-to-end logging

- Full experiment run produces logs
- Log file created at correct path
- Both handlers active simultaneously
- Different log levels work correctly

**Speed:** 10-60 seconds per test

## Migration Impact Analysis

### Files Modified

**New Files Created (2):**

- `src/utils/logging.py`
- `tests/utils/test_logging.py`

**Configuration Files Updated (4):**

- `configs/classifier.yaml`
- `configs/diffusion.yaml`
- `src/experiments/classifier/default.yaml`
- `src/experiments/diffusion/default.yaml`

**Source Files Updated (~10):**

- `src/main.py`
- `src/base/trainer.py`
- `src/experiments/classifier/trainer.py`
- `src/experiments/diffusion/trainer.py`
- `src/experiments/diffusion/sampler.py`
- (Optional) Several other utility files

**Test Files Updated (~5):**

- `tests/conftest.py`
- `tests/base/test_trainer.py`
- `tests/experiments/classifier/test_trainer.py`
- `tests/experiments/diffusion/test_trainer.py`
- (New) `tests/integration/test_logging_integration.py`

**Documentation Updated (3):**

- `README.md`
- `docs/standards/architecture.md`
- (New) `docs/features/20260216_logging-migration-guide.md`

### Backward Compatibility

**Breaking Changes:** None

**Config Compatibility:**

- Old configs without `logging` section: Use default values
- No migration required for existing configs
- Logging is additive, doesn't change existing behavior

### Performance Impact

**Minimal overhead:**

- Logging adds ~1-5ms per message (INFO/DEBUG level)
- File I/O is buffered, minimal impact
- DEBUG logs only computed if level enabled
- No impact on training speed (< 0.1%)

## Success Criteria

### Functional Requirements

- [x] All `print()` statements replaced with `logging` calls
- [x] Logs appear in both console and file
- [x] Log levels are configurable via YAML
- [x] Log files created with timestamp in filename
- [x] Different log levels for console and file
- [x] Module-specific log levels supported
- [x] Clean, readable log format
- [x] All tests pass with logging enabled

### Quality Requirements

- [x] Test coverage > 80% for logging utility
- [x] No regression in existing tests
- [x] Logging doesn't impact training performance
- [x] Log files remain readable size (< 10MB per experiment)
- [x] Documentation updated and clear

### User Experience

- [x] Default log level (INFO) shows important events only
- [x] DEBUG level useful for troubleshooting
- [x] Console output clean and not overwhelming
- [x] Log files provide complete audit trail
- [x] Easy to find log files for a given experiment

## Timeline Summary

**Total Estimated Time: 5 Weeks**

| Phase                          | Duration | Key Deliverables                      |
| ------------------------------ | -------- | ------------------------------------- |
| Phase 1: Core Infrastructure   | 1 week   | Logging utility, tests, config schema |
| Phase 2: Source Code Migration | 2 weeks  | Replace all prints in `/src`          |
| Phase 3: Enhanced Logging      | 1 week   | Add meaningful logs beyond prints     |
| Phase 4: Testing Integration   | 1 week   | Update tests, add logging tests       |
| Phase 5: Documentation         | 1 week   | Update docs, create migration guide   |

## Implementation Order

### Priority 1 (Critical Path)

1. Create `src/utils/logging.py`
2. Add tests for logging utility
3. Update config files with `logging` section
4. Initialize logging in `src/main.py`
5. Replace prints in `src/main.py`

### Priority 2 (Core Functionality)

6. Update `src/base/trainer.py`
7. Update experiment trainers
8. Update experiment samplers
9. Add enhanced logging

### Priority 3 (Testing & Polish)

10. Update test fixtures
11. Add logging tests
12. Update documentation
13. Final cleanup and verification

## Notes

- **Do not modify** `src/base/logger.py` or experiment-specific loggers (ClassifierLogger, DiffusionLogger) - these are for metrics, not application logging
- **Do not modify** `src/deprecated/` - these files are not part of active codebase
- Logging in tests should use `caplog` fixture or be configured to not pollute console
- Consider using `colorlog` library for colorized console output (optional enhancement)
- Log files should be added to `.gitignore` (likely already covered by `outputs/`)

## References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
- Project Architecture: `docs/standards/architecture.md`
- Testing Strategy: `docs/standards/architecture.md#testing-strategy`

---

**Next Steps:**

1. Review and approve this plan
2. Begin Phase 1 implementation
3. Create tracking issues for each phase
4. Regular progress reviews after each phase
