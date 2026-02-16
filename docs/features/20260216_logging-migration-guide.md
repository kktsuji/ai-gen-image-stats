# Logging Migration Guide

**Date**: February 17, 2026  
**Author**: AI Assistant  
**Status**: Active

## Overview

This guide helps users and contributors understand and use the new Python `logging` library integration. The project now uses standardized application logging for runtime events while maintaining separate metrics logging for training results.

## For Users

### Understanding the Two Logging Systems

The project uses two complementary logging systems:

1. **Application Logging** (Python `logging`): What the system is doing
   - Runtime events, debugging, errors
   - Output: Console + timestamped log files
   - Configure: `logging` section in YAML

2. **Metrics Logging** (BaseLogger): How well the model is performing
   - Training metrics, generated images, evaluation results
   - Output: CSV files, PNG images
   - Configure: Experiment-specific settings

### Viewing Logs

#### Console Output

By default, you'll see INFO-level messages in the console:

```bash
$ python -m src.main configs/classifier.yaml
2026-02-17 14:30:22 - src.main - INFO - CLASSIFIER EXPERIMENT STARTED
2026-02-17 14:30:22 - src.main - INFO - Using device: cuda
2026-02-17 14:30:23 - src.experiments.classifier.trainer - INFO - Starting training for 100 epochs
2026-02-17 14:30:45 - src.base.trainer - INFO - Epoch 1 completed - Avg Loss: 0.6234, Accuracy: 65.23%
```

#### Log Files

All logs (including DEBUG messages) are saved to timestamped files:

```
outputs/
└── your-experiment/
    └── logs/
        └── log_20260217_143022.log
```

**Finding Your Log File:** Look for the log file path in the console output at the start of your experiment.

### Configuring Log Levels

Add a `logging` section to your YAML config file:

```yaml
# configs/my_experiment.yaml

logging:
  console_level: INFO # What you see in the terminal
  file_level: DEBUG # What's saved to the log file


# ... rest of your config ...
```

### Common Use Cases

#### Verbose Output for Debugging

To see detailed diagnostic information in the console:

```yaml
logging:
  console_level: DEBUG
  file_level: DEBUG
```

This will show:

- Batch-level details
- Memory usage
- Model architecture information
- Data loading details

#### Quiet Console, Detailed Logs

For cleaner console output while maintaining detailed file logs:

```yaml
logging:
  console_level: WARNING # Only warnings and errors in console
  file_level: DEBUG # Everything in log file
```

#### Module-Specific Logging

Control verbosity for specific components:

```yaml
logging:
  console_level: INFO
  file_level: DEBUG
  module_levels:
    src.experiments.classifier.trainer: DEBUG # Verbose trainer logs
    src.utils.device: WARNING # Quiet device detection
    torch: ERROR # Suppress PyTorch messages
```

### Default Behavior

If you omit the `logging` section, the defaults are:

```yaml
logging:
  console_level: INFO
  file_level: DEBUG
  format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
```

### Troubleshooting

#### Not Seeing Expected Messages

**Problem:** Console is too quiet, missing important messages.

**Solution:** Check your `console_level` setting. Set it to `DEBUG` or `INFO`.

```yaml
logging:
  console_level: INFO # or DEBUG for more detail
```

#### Too Many Messages

**Problem:** Console is cluttered with too much information.

**Solution:** Increase the console level to `WARNING` or `ERROR`.

```yaml
logging:
  console_level: WARNING # Only show warnings and errors
  file_level: DEBUG # Keep detailed logs in file
```

#### Can't Find Log Files

**Problem:** Don't know where log files are saved.

**Solution:** Look for the log file path in the first few lines of console output:

```
INFO - Log file: outputs/your-experiment/logs/log_20260217_143022.log
```

Or check: `{output.base_dir}/{output.subdirs.logs}/`

#### Old Configs Don't Work

**Problem:** Running old config files without `logging` section.

**Solution:** Old configs work fine! Logging uses sensible defaults if the section is missing. To customize, add the `logging` section as shown above.

## For Contributors

### Adding Logging to New Code

#### 1. Import and Create Module-Level Logger

At the top of your module:

```python
import logging

logger = logging.getLogger(__name__)
```

**Why module-level?**

- Provides clear source identification in logs
- Follows Python best practices
- Enables module-specific log level control

#### 2. Choose Appropriate Log Level

Use this decision tree:

```
Is it an unrecoverable error?
  YES → logger.critical()
  NO ↓

Is it an error that affects results?
  YES → logger.error() or logger.exception()
  NO ↓

Is it a potential issue/warning?
  YES → logger.warning()
  NO ↓

Is it important progress/status?
  YES → logger.info()
  NO ↓

Is it detailed diagnostic info?
  YES → logger.debug()
```

#### 3. Log Examples

**Training Progress:**

```python
def train(self, num_epochs: int) -> None:
    """Train the model."""
    logger.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs} started")

        # Training loop...
        avg_loss, accuracy = self._train_epoch(epoch)

        # Epoch summary (INFO level)
        logger.info(
            f"Epoch {epoch} completed - "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%, "
            f"Time: {epoch_time:.1f}s"
        )

        # Detailed diagnostics (DEBUG level)
        logger.debug(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        logger.debug(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
```

**Checkpoint Operations:**

```python
def save_checkpoint(self, path: Path, epoch: int, metrics: Dict) -> None:
    """Save model checkpoint."""
    logger.info(f"Saving checkpoint to: {path}")
    logger.debug(f"  Epoch: {epoch}")
    logger.debug(f"  Metrics: {metrics}")

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
        logger.info("Checkpoint saved successfully")

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        logger.exception("Full exception details:")
        raise
```

**Error Handling:**

```python
def load_checkpoint(self, path: Path) -> Dict:
    """Load model checkpoint."""
    if not path.exists():
        logger.warning(f"Checkpoint not found at {path}, starting from scratch")
        return None

    try:
        logger.info(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path)
        logger.debug(f"Checkpoint contains keys: {list(checkpoint.keys())}")
        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.exception("Full exception details:")  # Includes stack trace
        raise
```

**Configuration Loading:**

```python
def load_config(self, config_path: Path) -> Dict:
    """Load experiment configuration."""
    logger.info(f"Loading configuration from: {config_path}")

    if not config_path.exists():
        logger.critical(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text())

    if "experiment" not in config:
        logger.error("Missing required field: experiment")
        raise ValueError("Config must specify 'experiment' field")

    logger.debug(f"Loaded config: {config}")
    return config
```

### Testing Logging

#### Verifying Log Messages in Tests

Use pytest's `caplog` fixture:

```python
import logging
import pytest

def test_trainer_logs_epoch_start(trainer, caplog):
    """Trainer logs epoch start at INFO level."""
    caplog.set_level(logging.INFO)

    trainer.train(num_epochs=1)

    # Verify message was logged
    assert "Epoch 1" in caplog.text
    assert "started" in caplog.text.lower()

def test_trainer_logs_checkpoint_save(trainer, caplog, tmp_path):
    """Trainer logs checkpoint save operations."""
    caplog.set_level(logging.INFO)
    checkpoint_path = tmp_path / "test.pt"

    trainer.save_checkpoint(checkpoint_path, epoch=1, metrics={})

    assert "checkpoint saved" in caplog.text.lower()
    assert str(checkpoint_path) in caplog.text
```

#### Testing Log Levels

```python
def test_debug_messages_not_shown_at_info(trainer, caplog):
    """DEBUG messages are only shown when level is DEBUG."""
    caplog.set_level(logging.INFO)

    trainer.train(num_epochs=1)

    # DEBUG message should not appear
    assert "GPU memory" not in caplog.text

    # INFO message should appear
    assert "Epoch 1 completed" in caplog.text
```

#### Using Test Fixtures

```python
# tests/conftest.py

@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture all log messages."""
    caplog.set_level(logging.DEBUG)
    return caplog

@pytest.fixture
def temp_log_file(tmp_path):
    """Temporary log file for testing."""
    return tmp_path / "test.log"

# In your test:
def test_with_log_capture(trainer, capture_logs):
    """Use the fixture to capture logs."""
    trainer.train(num_epochs=1)
    assert "training" in capture_logs.text.lower()
```

### Code Review Checklist

When reviewing code with logging:

- [ ] Uses module-level logger (`logger = logging.getLogger(__name__)`)
- [ ] Appropriate log level chosen (INFO for important events, DEBUG for details)
- [ ] Log messages include relevant context (epoch, loss, paths, etc.)
- [ ] No sensitive information logged (API keys, passwords, etc.)
- [ ] Uses f-strings for formatting, not string concatenation
- [ ] Uses `logger.exception()` in exception handlers (includes stack trace)
- [ ] No `print()` statements for application logging
- [ ] Not over-logging (no per-batch INFO messages in tight loops)
- [ ] Tests verify important log messages

### Best Practices Summary

**DO:**

✅ Use module-level loggers: `logger = logging.getLogger(__name__)`  
✅ Choose appropriate log levels (INFO for events, DEBUG for details)  
✅ Include context in messages: `logger.info(f"Epoch {epoch} completed - Loss: {loss:.4f}")`  
✅ Use `logger.exception()` in exception handlers  
✅ Test important log messages with `caplog`  
✅ Use f-strings for formatting: `f"Epoch {epoch}"`

**DON'T:**

❌ Use `print()` for application logging  
❌ Create logger instances in functions  
❌ Log sensitive information (keys, passwords)  
❌ Log excessively at INFO level (e.g., every batch)  
❌ Use string concatenation: `"Epoch " + str(epoch)`  
❌ Confuse application logging with metrics logging

### When to Use Which Logger

**Use Application Logging** (`logging.getLogger(__name__)`) for:

- System events (startup, shutdown)
- Training progress (epoch summaries)
- Checkpoint operations
- Configuration loading
- Error messages
- Debug diagnostics

**Use Metrics Logging** (BaseLogger subclasses) for:

- Training loss curves
- Validation accuracy
- Confusion matrices
- Generated sample images
- Evaluation metrics (FID, IS, etc.)
- Hyperparameter tracking

### Migration Pattern

If you're updating old code with `print()` statements:

```python
# Before:
print(f"Starting training for {num_epochs} epochs")
print(f"Using device: {device}")

# After:
import logging
logger = logging.getLogger(__name__)

logger.info(f"Starting training for {num_epochs} epochs")
logger.info(f"Using device: {device}")
```

### Performance Considerations

Logging has minimal overhead:

- ~1-5ms per log message
- Negligible impact on training speed (< 0.1%)
- File I/O is buffered

**Optimization tip** for expensive computations:

```python
# Only compute if DEBUG logging is enabled
if logger.isEnabledFor(logging.DEBUG):
    detailed_stats = expensive_computation()
    logger.debug(f"Stats: {detailed_stats}")
```

## Quick Reference

### Log Level Reference

| Level    | Value | When to Use                | Example                          |
| -------- | ----- | -------------------------- | -------------------------------- |
| DEBUG    | 10    | Detailed diagnostics       | Batch shapes, memory usage       |
| INFO     | 20    | Important events (default) | Training start, epoch completion |
| WARNING  | 30    | Potential issues           | Missing optional config, low LR  |
| ERROR    | 40    | Errors affecting results   | Failed checkpoint load           |
| CRITICAL | 50    | Unrecoverable errors       | Out of memory, missing files     |

### Configuration Quick Start

**Minimal Config:**

```yaml
logging:
  console_level: INFO
  file_level: DEBUG
```

**Verbose Config:**

```yaml
logging:
  console_level: DEBUG
  file_level: DEBUG
```

**Quiet Config:**

```yaml
logging:
  console_level: WARNING
  file_level: DEBUG
```

**Custom Format:**

```yaml
logging:
  console_level: INFO
  file_level: DEBUG
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  date_format: "%H:%M:%S"
```

### Common Patterns

**Training Loop:**

```python
logger.info(f"Epoch {epoch}/{num_epochs} started")
# ... training ...
logger.info(f"Epoch {epoch} completed - Loss: {loss:.4f}, Acc: {acc:.2f}%")
```

**Checkpoint Save:**

```python
logger.info(f"Saving checkpoint: {path}")
# ... save ...
logger.info("Checkpoint saved successfully")
```

**Error Handling:**

```python
try:
    # ... operation ...
except Exception as e:
    logger.error(f"Operation failed: {e}")
    logger.exception("Full traceback:")
    raise
```

## Support

For questions or issues:

1. Check this guide
2. Review examples in `src/main.py`, `src/base/trainer.py`
3. Check tests in `tests/utils/test_logging.py`
4. Refer to [Python Logging Documentation](https://docs.python.org/3/library/logging.html)

---

**Last Updated**: February 17, 2026  
**Version**: 1.0
