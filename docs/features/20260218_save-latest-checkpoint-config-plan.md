# Implementation Plan: Configurable Save-Latest-Checkpoint

**Date:** 2026-02-18  
**Status:** Draft  
**Scope:** `BaseTrainer`, diffusion and classifier experiment trainers, both default configs, config validators, `main.py`, tests

---

## Background

`BaseTrainer.train()` and `resume_training()` currently always write `latest_checkpoint.pth` at the end of every epoch in an unconditional block (see `src/base/trainer.py` lines 341–349 and the equivalent block in `resume_training()`).  
The two experiment-specific trainers (`src/experiments/classifier/trainer.py` and `src/experiments/diffusion/trainer.py`) both override `train()` and carry the same unconditional block.

This plan introduces a boolean flag `training.checkpointing.save_latest` that lets users opt in or out of this behaviour via the YAML config. Default is `true` (preserves the existing behaviour).

---

## 1. TODO List

- [x] Phase 1: Config layer
  - [x] Task 1.1: Add `save_latest` field to `src/experiments/classifier/default.yaml`
  - [x] Task 1.2: Add `save_latest` field to `src/experiments/diffusion/default.yaml`
  - [x] Task 1.3: Add validation for `save_latest` in `src/experiments/classifier/config.py`
  - [x] Task 1.4: Add validation for `save_latest` in `src/experiments/diffusion/config.py`

- [x] Phase 2: Base trainer
  - [x] Task 2.1: Add `save_latest_checkpoint: bool = True` parameter to `BaseTrainer.train()`
  - [x] Task 2.2: Gate the latest-checkpoint block in `BaseTrainer.train()` behind the flag
  - [x] Task 2.3: Add `save_latest_checkpoint: bool = True` parameter to `BaseTrainer.resume_training()`
  - [x] Task 2.4: Gate the latest-checkpoint block in `BaseTrainer.resume_training()` behind the flag

- [ ] Phase 3: Experiment trainers
  - [ ] Task 3.1: Add `save_latest_checkpoint` parameter to `ClassifierTrainer.train()`
  - [ ] Task 3.2: Gate the latest-checkpoint block in `ClassifierTrainer.train()` behind the flag
  - [ ] Task 3.3: Add `save_latest_checkpoint` parameter to `DiffusionTrainer.train()`
  - [ ] Task 3.4: Gate the latest-checkpoint block in `DiffusionTrainer.train()` behind the flag

- [ ] Phase 4: Wiring in `main.py`
  - [ ] Task 4.1: Read `training.checkpointing.save_latest` from config and pass it to `trainer.train()`

- [ ] Phase 5: Tests
  - [ ] Task 5.1: Unit tests — validate config parsing of `save_latest`
  - [ ] Task 5.2: Component tests — verify `latest_checkpoint.pth` is (not) written based on flag

---

## 2. Details of Each Phase

---

### Phase 1: Config layer

#### Task 1.1 & 1.2 — Add `save_latest` to both default YAMLs

Add a new field to the `training.checkpointing` section in both experiment default configs.

**`src/experiments/classifier/default.yaml`** — inside `training.checkpointing`:

```yaml
checkpointing:
  save_frequency: 10 # Save checkpoint every N epochs
  save_best_only: true # Only save checkpoint when validation metric improves
  save_optimizer: true # Include optimizer state in checkpoints
  save_latest: true # Always save latest_checkpoint.pth after every epoch
```

**`src/experiments/diffusion/default.yaml`** — inside `training.checkpointing`:

```yaml
checkpointing:
  save_frequency: 10 # Save every N epochs
  save_best_only: false # Save all checkpoints
  save_optimizer: true # Include optimizer state in checkpoints
  save_latest: true # Always save latest_checkpoint.pth after every epoch
```

Setting `save_latest: false` disables writing `latest_checkpoint.pth` entirely.  
Setting `save_latest: true` (or omitting the field) preserves the existing behaviour.

---

#### Task 1.3 — Validate `save_latest` in `src/experiments/classifier/config.py`

Locate the `checkpointing` validation block and append:

```python
if "save_latest" in ckpt and not isinstance(ckpt["save_latest"], bool):
    raise ValueError("training.checkpointing.save_latest must be a boolean")
```

---

#### Task 1.4 — Validate `save_latest` in `src/experiments/diffusion/config.py`

Locate the block that validates `training.checkpointing` (around line 373) and append the same check:

```python
if "save_latest" in ckpt and not isinstance(ckpt["save_latest"], bool):
    raise ValueError("training.checkpointing.save_latest must be a boolean")
```

---

### Phase 2: Base trainer (`src/base/trainer.py`)

#### Task 2.1 & 2.2 — `BaseTrainer.train()`

Add `save_latest_checkpoint: bool = True` to the signature:

```python
def train(
    self,
    num_epochs: int,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_frequency: int = 1,
    validate_frequency: int = 1,
    save_best: bool = True,
    best_metric: str = "loss",
    best_metric_mode: str = "min",
    save_latest_checkpoint: bool = True,   # ← new
) -> None:
```

Replace the unconditional latest-checkpoint block:

```python
# Before (always runs)
latest_path = checkpoint_dir / "latest_checkpoint.pth"
self.save_checkpoint(
    latest_path,
    epoch=self._current_epoch,
    is_best=False,
    metrics={**train_metrics, **(val_metrics if val_metrics else {})},
)
logger.debug(f"Latest checkpoint updated: {latest_path}")
```

With:

```python
# After (conditional)
if save_latest_checkpoint:
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    self.save_checkpoint(
        latest_path,
        epoch=self._current_epoch,
        is_best=False,
        metrics={**train_metrics, **(val_metrics if val_metrics else {})},
    )
    logger.debug(f"Latest checkpoint updated: {latest_path}")
```

Update the docstring to document the new parameter:

```
save_latest_checkpoint: If True, writes latest_checkpoint.pth after every epoch.
                        If False, only periodic and best checkpoints are written.
```

---

#### Task 2.3 & 2.4 — `BaseTrainer.resume_training()`

Apply the identical signature extension and conditional block to `resume_training()`, which contains a verbatim copy of the same latest-checkpoint block.

---

### Phase 3: Experiment trainers

Both experiment trainers override `train()` and contain their own copy of the latest-checkpoint block. The changes mirror Phase 2 exactly.

#### Task 3.1 & 3.2 — `ClassifierTrainer.train()` (`src/experiments/classifier/trainer.py`)

1. Add `save_latest_checkpoint: bool = True` to the signature.
2. Wrap the latest-checkpoint block (around line 363) with `if save_latest_checkpoint:`.

#### Task 3.3 & 3.4 — `DiffusionTrainer.train()` (`src/experiments/diffusion/trainer.py`)

1. Add `save_latest_checkpoint: bool = True` to the signature.
2. Wrap the latest-checkpoint block (around line 582) with `if save_latest_checkpoint:`.

---

### Phase 4: Wiring in `main.py`

#### Task 4.1 — Read config value and pass to `trainer.train()`

The experiment entry points (launched from `src/main.py`) already read `training.checkpointing` fields and forward them to the trainer's `train()` call. Extend the same pattern:

```python
# In the section that builds the train() call arguments
checkpointing_cfg = config["training"]["checkpointing"]
save_latest = checkpointing_cfg.get("save_latest", True)   # default True if omitted

trainer.train(
    num_epochs=...,
    checkpoint_dir=...,
    checkpoint_frequency=checkpointing_cfg["save_frequency"],
    ...
    save_latest_checkpoint=save_latest,
)
```

> **Note:** If `save_latest` is absent from the config (e.g., older user configs), the default `True` preserves backward compatibility.

---

### Phase 5: Tests

#### Task 5.1 — Unit tests for config validation

File: `tests/experiments/classifier/test_config.py` and `tests/experiments/diffusion/test_config.py`

Add test cases:

```python
@pytest.mark.unit
def test_save_latest_valid_true(minimal_config):
    """save_latest: true is accepted."""
    minimal_config["training"]["checkpointing"]["save_latest"] = True
    validate_config(minimal_config)  # should not raise

@pytest.mark.unit
def test_save_latest_valid_false(minimal_config):
    """save_latest: false is accepted."""
    minimal_config["training"]["checkpointing"]["save_latest"] = False
    validate_config(minimal_config)  # should not raise

@pytest.mark.unit
def test_save_latest_invalid_type(minimal_config):
    """save_latest with non-bool raises ValueError."""
    minimal_config["training"]["checkpointing"]["save_latest"] = "yes"
    with pytest.raises(ValueError, match="save_latest must be a boolean"):
        validate_config(minimal_config)
```

#### Task 5.2 — Component tests verifying file creation

File: `tests/base/test_trainer.py` (or the appropriate component test module)

```python
@pytest.mark.component
def test_latest_checkpoint_written_when_enabled(mock_trainer, tmp_path):
    """latest_checkpoint.pth is created when save_latest_checkpoint=True."""
    mock_trainer.train(
        num_epochs=1,
        checkpoint_dir=tmp_path,
        save_latest_checkpoint=True,
    )
    assert (tmp_path / "latest_checkpoint.pth").exists()


@pytest.mark.component
def test_latest_checkpoint_not_written_when_disabled(mock_trainer, tmp_path):
    """latest_checkpoint.pth is NOT created when save_latest_checkpoint=False."""
    mock_trainer.train(
        num_epochs=1,
        checkpoint_dir=tmp_path,
        save_latest_checkpoint=False,
    )
    assert not (tmp_path / "latest_checkpoint.pth").exists()
```

---

## Compatibility Notes

| Scenario                          | Behaviour                                                                     |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `save_latest` omitted from config | Defaults to `True` — no behaviour change                                      |
| `save_latest: true`               | `latest_checkpoint.pth` written every epoch (existing behaviour)              |
| `save_latest: false`              | `latest_checkpoint.pth` never written; only periodic + best checkpoints saved |
| `checkpoint_dir: null`            | Latest-checkpoint block is already skipped entirely; flag is irrelevant       |

The change is backwards-compatible: existing config files without `save_latest` default to `True`.

---

## Affected Files Summary

| File                                          | Change                                                             |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `src/experiments/classifier/default.yaml`     | Add `save_latest: true` to `training.checkpointing`                |
| `src/experiments/diffusion/default.yaml`      | Add `save_latest: true` to `training.checkpointing`                |
| `src/experiments/classifier/config.py`        | Validate `save_latest` is a bool                                   |
| `src/experiments/diffusion/config.py`         | Validate `save_latest` is a bool                                   |
| `src/base/trainer.py`                         | Add param + conditional guard in `train()` and `resume_training()` |
| `src/experiments/classifier/trainer.py`       | Add param + conditional guard in `train()`                         |
| `src/experiments/diffusion/trainer.py`        | Add param + conditional guard in `train()`                         |
| `src/main.py`                                 | Read `save_latest` from config, pass to `trainer.train()`          |
| `tests/experiments/classifier/test_config.py` | Unit tests for `save_latest` validation                            |
| `tests/experiments/diffusion/test_config.py`  | Unit tests for `save_latest` validation                            |
| `tests/base/test_trainer.py`                  | Component tests for file presence/absence                          |
