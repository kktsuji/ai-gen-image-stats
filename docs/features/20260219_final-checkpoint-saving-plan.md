# Final Checkpoint Saving After Training Completion

## Overview

After training finishes, save a `final_model.pth` checkpoint unconditionally — regardless of any configuration settings. This guarantees that the last-epoch model state is always preserved as a clearly named artifact, distinct from `best_model.pth` (best metric) and `latest_checkpoint.pth` (rolling latest epoch).

**Scope of changes:**

- `src/base/trainer.py` — `BaseTrainer.train()` and `BaseTrainer.resume_training()`
- `src/experiments/classifier/trainer.py` — `ClassifierTrainer.train()`
- `src/experiments/diffusion/trainer.py` — `DiffusionTrainer.train()`

No config parameter is added; saving `final_model.pth` is always done when a `checkpoint_dir` is provided, after the epoch loop completes.

**File saved:** `<checkpoint_dir>/final_model.pth`

**Expected outcome:** Every completed training run (including resumed runs) leaves a `final_model.pth` in the checkpoint directory containing the full model/optimizer/metrics state of the final epoch.

**Estimated effort:** Small — ~4 targeted code additions + corresponding tests.

---

## Implementation Checklist

- [ ] Phase 1: Base trainer
  - [ ] Task 1.1: Add final checkpoint save after the epoch loop in `BaseTrainer.train()`
  - [ ] Task 1.2: Add final checkpoint save after the epoch loop in `BaseTrainer.resume_training()`
- [ ] Phase 2: Classifier trainer
  - [ ] Task 2.1: Add final checkpoint save after the epoch loop in `ClassifierTrainer.train()`
- [ ] Phase 3: Diffusion trainer
  - [ ] Task 3.1: Add final checkpoint save after the epoch loop in `DiffusionTrainer.train()`
- [ ] Phase 4: Tests
  - [ ] Task 4.1: Add / update tests for `BaseTrainer.train()` to assert `final_model.pth` exists
  - [ ] Task 4.2: Add / update tests for `ClassifierTrainer.train()` to assert `final_model.pth` exists
  - [ ] Task 4.3: Add / update tests for `DiffusionTrainer.train()` to assert `final_model.pth` exists
- [ ] Phase 5: Confirm all tests pass
  - [ ] Task 5.1: Run the full test suite and verify no regressions
- [ ] Phase 6: Update documentation
  - [ ] Task 6.1: Update `README.md` — add `final_model.pth` to the checkpoint output structure example
  - [ ] Task 6.2: Update `docs/standards/architecture.md` — add `final_model.pth` to the checkpoint directory example in the Log File Structure section

---

## Phase Details

### Phase 1: Base trainer (`src/base/trainer.py`)

#### Task 1.1 — `BaseTrainer.train()`

After the `for epoch in range(num_epochs):` loop ends, add the following block (still inside the method, but outside the loop):

```python
# Save final checkpoint after all epochs complete
if checkpoint_dir is not None:
    final_path = checkpoint_dir / "final_model.pth"
    self.save_checkpoint(
        final_path,
        epoch=self._current_epoch,
        is_best=False,
        metrics={
            **train_metrics,
            **(val_metrics if val_metrics else {}),
        },
    )
    logger.info(f"Final model checkpoint saved: {final_path}")
```

`train_metrics` and `val_metrics` are naturally in scope from the last iteration of the loop. If `num_epochs == 0` the loop body never runs; in that edge case, no checkpoint is saved (same behavior as the other checkpoint types).

#### Task 1.2 — `BaseTrainer.resume_training()`

Apply the identical block after the `for epoch in range(num_epochs):` loop in `resume_training()`, using the same variable names (`train_metrics`, `val_metrics`, `checkpoint_dir`, `self._current_epoch`).

---

### Phase 2: Classifier trainer (`src/experiments/classifier/trainer.py`)

#### Task 2.1 — `ClassifierTrainer.train()`

`ClassifierTrainer` overrides `train()` independently. Apply the same final-checkpoint block after its epoch loop, exactly as in Task 1.1.

---

### Phase 3: Diffusion trainer (`src/experiments/diffusion/trainer.py`)

#### Task 3.1 — `DiffusionTrainer.train()`

`DiffusionTrainer` overrides `train()` independently. Apply the same final-checkpoint block after its epoch loop, exactly as in Task 1.1.

> **Note:** `DiffusionTrainer.save_checkpoint()` already handles EMA and AMP states, so calling `self.save_checkpoint()` here is sufficient — no extra logic is needed.

---

### Phase 4: Tests

#### Task 4.1 — `tests/base/test_trainer.py` (or equivalent)

Add an assertion in the existing `test_training_with_checkpointing` (or a new test) that `final_model.pth` is created after `trainer.train(...)`.

```python
assert (checkpoint_dir / "final_model.pth").exists()
```

#### Task 4.2 — `tests/experiments/classifier/test_trainer.py`

Add the same assertion to the existing `test_training_with_checkpointing` test for the classifier trainer.

#### Task 4.3 — `tests/experiments/diffusion/test_trainer.py`

Add the same assertion to the existing checkpoint test for the diffusion trainer.

---

### Phase 5: Confirm all tests pass

#### Task 5.1 — Run the full test suite

After all code and test changes are in place, run the full test suite to confirm no regressions:

```bash
venv/bin/python -m pytest
```

All tests must pass, including:

- Unit / component tests in `tests/base/`
- Integration tests in `tests/experiments/classifier/` and `tests/experiments/diffusion/`
- Any smoke tests in `tests/integration/`

---

### Phase 6: Update documentation

#### Task 6.1 — `README.md`

In the **Output Structure** section, add `final_model.pth` to the checkpoints directory example:

```
outputs/experiment_name/
├── checkpoints/
│   ├── checkpoint_epoch_N.pth
│   ├── best_model.pth
│   ├── latest_checkpoint.pth
│   └── final_model.pth        # ← new: saved once after all epochs complete
```

#### Task 6.2 — `docs/standards/architecture.md`

In the **Log File Structure** section, update the checkpoints directory example to include `final_model.pth`:

```
├── checkpoints/
│   ├── checkpoint_epoch_N.pth
│   ├── best_model.pth
│   ├── latest_checkpoint.pth
│   └── final_model.pth        # ← new: saved once after all epochs complete
```
