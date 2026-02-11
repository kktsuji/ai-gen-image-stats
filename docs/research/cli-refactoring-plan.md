# CLI Refactoring Plan: Config-Only Mode

**Date:** February 12, 2026  
**Author:** GitHub Copilot  
**Status:** Planning

## Executive Summary

This document outlines the plan to refactor the CLI from a hybrid model (CLI params + JSON config + defaults) to a strict config-only model (JSON config only, no defaults).

See the [TODO List](#executable-step-by-step-todo-list) below for the main actionable content.

### Current Behavior vs Target Behavior

**Current (multiple input methods):**

```bash
python -m src.main --experiment classifier --model resnet50 --epochs 10
python -m src.main --experiment classifier --config config.json --batch-size 64
python -m src.main --experiment classifier  # Uses defaults
```

Priority: CLI params > JSON config > Code defaults

**Target (config-only, simplified):**

```bash
# Simple positional argument for config file
python -m src.main configs/classifier/baseline.json
python -m src.main configs/diffusion/default.json
```

Priority: JSON config values ONLY (no defaults, no CLI overrides)

### Quick Reference

| Aspect              | Detail                             |
| ------------------- | ---------------------------------- |
| **Estimated Time**  | 19 hours (2-3 sessions)            |
| **Files Modified**  | 10+ files                          |
| **Breaking Change** | YES                                |
| **New CLI Format**  | `python -m src.main <config.json>` |

**Navigation:**

- [Executable TODO List](#executable-step-by-step-todo-list) - Main task list
- [Risk Assessment](#risk-assessment) - Risks and mitigations
- [Timeline & Success Criteria](#estimated-timeline) - Planning info
- [Appendix A](#appendix-a-investigation-summary) - Investigation details
- [Appendix B](#appendix-b-detailed-change-plan) - Technical specifications

---

## Executable Step-by-Step TODO List

### Prerequisites

- [ ] Create feature branch: `git checkout -b feature/cli-config-only`
- [ ] Ensure virtual environment is activated
- [ ] Run existing tests to establish baseline: `pytest`

### Step 1: Create Test Config Files

**Goal:** Set up test fixtures before making changes

- [x] 1.1 Create test config directory structure

  ```bash
  mkdir -p tests/fixtures/configs/classifier tests/fixtures/configs/diffusion
  ```

- [x] 1.2 Create `tests/fixtures/configs/classifier/valid_minimal.json`
  - Copy from `configs/classifier/baseline.json`
  - Verify all required fields present

- [x] 1.3 Create `tests/fixtures/configs/classifier/invalid_missing_model.json`
  - Copy from valid config, remove `model` section
  - Use for error testing

- [x] 1.4 Create `tests/fixtures/configs/diffusion/valid_minimal.json`
  - Copy from `configs/diffusion/default.json`

- [x] 1.5 Create `tests/fixtures/configs/diffusion/invalid_missing_data.json`
  - Copy from valid config, remove `data.train_path` field

**Validation:** Files exist and contain valid JSON

---

### Step 2: Update Experiment Config Validators (Strict Mode)

**Goal:** Make validation strict BEFORE changing CLI

- [x] 2.1 Update `src/experiments/classifier/config.py` `validate_config()`
  - Add comprehensive required field checks for ALL nested sections
  - Provide clear error messages with field paths
  - Don't allow None for critical fields
  - See [Appendix B.3.1](#31-update-srcexperimentsclassifierconfigpy) for required fields list

- [x] 2.2 Update `src/experiments/diffusion/config.py` `validate_config()`
  - Similar structure to classifier
  - See [Appendix B.3.2](#32-update-srcexperimentsdiffusionconfigpy) for required fields list

- [x] 2.3 Test strict validation
  ```bash
  pytest tests/experiments/classifier/test_config.py -v
  pytest tests/experiments/diffusion/test_config.py -v
  ```

**Validation:** Strict validators catch missing fields

---

### Step 3: Update CLI Parser (Config-Only Mode)

**Goal:** Simplify CLI to accept only config file

- [x] 3.1 Backup: `cp src/utils/cli.py src/utils/cli.py.backup`

- [x] 3.2 Update `create_parser()` in `src/utils/cli.py`
  - Remove ALL 20+ specific parameter arguments
  - Add positional: `config_path` (required)
  - Remove `--experiment` (read from config)
  - See [Appendix B.1.1](#11-update-srcutilsclipy) for details

- [x] 3.3 Simplify/remove `args_to_dict()` (may not be needed)

- [x] 3.4 Update `parse_args()` in `src/utils/cli.py`
  - Get config_path from positional arg
  - Load with `load_config(config_path)`
  - Verify 'experiment' field exists
  - Validate experiment type (classifier/diffusion/gan)
  - Return config directly (NO merging)

- [x] 3.5 Update `validate_config()` if needed

**Validation:** `python -c "from src.utils.cli import parse_args; print('OK')"`

---

### Step 4: Update Main Entry Point

**Goal:** Remove default config merging

- [ ] 4.1 Update `src/main.py` docstring
  - Show only config-file examples
  - Remove CLI parameter examples

- [ ] 4.2 Update `setup_experiment_classifier()`
  - Remove: `config = merge_configs(classifier_defaults, config)`
  - Keep: `validate_classifier_config(config)`

- [ ] 4.3 Update `setup_experiment_diffusion()`
  - Remove: `config = merge_configs(diffusion_defaults, config)`
  - Keep: `validate_diffusion_config(config)`

- [ ] 4.4 Test: `python -c "from src.main import main; print('OK')"`

**Validation:** No import errors

---

### Step 5: Test with Real Config Files

**Goal:** Verify changes work with actual configs

- [ ] 5.1 Test classifier: `python -m src.main configs/classifier/baseline.json`

- [ ] 5.2 Test diffusion: `python -m src.main configs/diffusion/default.json`

- [ ] 5.3 Test error cases:
  ```bash
  python -m src.main  # Should error: config required
  python -m src.main nonexistent.json  # Should error: file not found
  python -m src.main configs/classifier/baseline.json --epochs 10  # Should error: unrecognized arg
  ```

**Validation:** CLI works with configs, rejects invalid usage

---

### Step 6: Update Tests

**Goal:** Fix broken tests, add new tests

- [ ] 6.1 Run tests to see failures: `pytest tests/utils/test_cli.py -v`

- [ ] 6.2 Update `TestCreateParser` class
  - Add: `test_parser_has_positional_config_argument()`
  - Add: `test_config_path_is_required()`

- [ ] 6.3 Remove obsolete test classes:
  - `TestParseTrainingArguments`, `TestParseModelArguments`
  - `TestParseDataArguments`, `TestParseOutputArguments`
  - `TestParseModeArguments`, `TestParseDeviceArguments`
  - `TestArgsToDict`

- [ ] 6.4 Update `TestParseArgs` class
  - Remove CLI-only, defaults, overrides, priority tests
  - Keep: `test_parse_args_with_config_file()` (update it)
  - Add: `test_parse_args_requires_config()`
  - Add: `test_parse_args_config_file_not_found()`

- [ ] 6.5 Update `TestValidateConfig` for strict validation

- [ ] 6.6 Add `TestConfigOnlyMode` class:
  - `test_requires_config_file()`
  - `test_no_cli_overrides_allowed()`
  - `test_loads_config_only()`
  - `test_experiment_from_config()`

- [ ] 6.7 Run: `pytest tests/utils/test_cli.py -v`

- [ ] 6.8 Check: `pytest tests/test_main.py -v`

**Validation:** All tests pass

---

### Step 7: Update Documentation

**Goal:** Update user-facing documentation

- [ ] 7.1 Update `README.md`
  - Update "Quick Start" section
  - Remove CLI parameter examples
  - Show only: `python -m src.main configs/<file>.json`

- [ ] 7.2 Update `docs/standards/architecture.md`
  - Update "CLI Interface" section
  - Remove priority explanation
  - Add strict validation requirements

- [ ] 7.3 Create `docs/research/cli-migration-guide.md`
  - Before/after examples
  - Migration instructions

**Validation:** Docs are clear and accurate

---

### Step 8: Final Testing

**Goal:** Comprehensive validation

- [ ] 8.1-8.4 Run all test suites:

  ```bash
  pytest -m unit -v
  pytest -m component -v
  pytest -m integration -v
  pytest -v  # Full suite
  ```

- [ ] 8.5 Test real workflows:

  ```bash
  python -m src.main configs/classifier/baseline.json
  python -m src.main configs/diffusion/default.json
  ```

- [ ] 8.6 Test error cases (missing config, extra args, missing fields)

**Validation:** All tests pass, error messages clear

---

### Step 9: Code Review and Cleanup

- [ ] 9.1 Remove backup: `rm src/utils/cli.py.backup`
- [ ] 9.2 Check for unused imports
- [ ] 9.3 Format code: `black src/utils/cli.py src/main.py`
- [ ] 9.4 Review changes: `git diff main`
- [ ] 9.5 Update CHANGELOG if exists

**Validation:** Code is clean

---

### Step 10: Commit and Document

- [ ] 10.1 Stage all changes
- [ ] 10.2 Commit with message:

  ```
  Refactor CLI to config-only mode

  - Remove CLI parameter arguments
  - Require JSON config file as positional arg
  - Implement strict validation (no defaults)
  - Update tests and documentation

  BREAKING CHANGE: CLI parameters no longer accepted
  ```

- [ ] 10.3 Create PR with link to this document

**Validation:** Changes committed

---

## Risk Assessment

### High Risk

- **Breaking Change:** Existing scripts will break
  - Mitigation: Create migration guide, clear documentation

### Medium Risk

- **Test Coverage:** 667 lines to refactor
  - Mitigation: Incremental refactoring, 4-5 hours allocated
- **Config Completeness:** May have missing fields
  - Mitigation: Audit configs first (Step 1)

### Low Risk

- **Documentation:** Multiple files to update
  - Mitigation: 2 hours allocated

---

## Estimated Timeline

| Phase     | Task           | Time         |
| --------- | -------------- | ------------ |
| 1         | Test configs   | 30 min       |
| 2         | Validators     | 4 hours      |
| 3         | CLI parser     | 3 hours      |
| 4         | Main.py        | 1 hour       |
| 5         | Manual testing | 1 hour       |
| 6         | Tests          | 5 hours      |
| 7         | Docs           | 2 hours      |
| 8         | Final testing  | 1 hour       |
| 9         | Review         | 1 hour       |
| 10        | Commit         | 30 min       |
| **Total** |                | **19 hours** |

---

## Success Criteria

- [ ] All tests pass
- [ ] CLI: `python -m src.main <config.json> [--verbose]` only
- [ ] Config file required as positional arg
- [ ] Experiment type from config, not CLI
- [ ] Strict validation (no None/missing fields)
- [ ] Clear error messages
- [ ] Documentation updated
- [ ] Migration guide created

---

## Notes

- Consider `--validate-config` flag for dry-run validation
- Consider JSON schema files for validation
- May want example configs for common cases

---

## References

- [src/utils/cli.py](../../src/utils/cli.py) - Current CLI
- [src/main.py](../../src/main.py) - Main entry
- [configs/](../../configs/) - Config examples
- [docs/standards/architecture.md](../../docs/standards/architecture.md) - Architecture

---

# Appendices

## Appendix A: Investigation Summary

### Files Analyzed

**Core CLI:**

- `src/utils/cli.py` (395 lines) - Argparse with 20+ arguments
- `src/utils/config.py` - Config loading and merging
- `src/main.py` (629 lines) - Uses parse_args(), merges defaults

**Experiment Configs:**

- `src/experiments/classifier/config.py` - Defaults and validation
- `src/experiments/diffusion/config.py` - Defaults and validation

**Config Files:**

- `configs/classifier/baseline.json` - Complete classifier config
- `configs/classifier/inceptionv3.json` - InceptionV3 variant
- `configs/diffusion/default.json` - Complete diffusion config

**Tests:**

- `tests/utils/test_cli.py` (667 lines) - Major refactoring needed
- `tests/integration/*` - Use components directly, no changes needed

**Docs:**

- `README.md` - Usage examples need updating
- `docs/standards/architecture.md` - CLI interface docs need updating

### Key Findings

**CLI Arguments to Remove (20+):**

- Training: `--epochs`, `--batch-size`, `--lr`, `--optimizer`, `--seed`
- Model: `--model`, `--pretrained`, `--num-classes`
- Data: `--train-path`, `--val-path`, `--num-workers`
- Output: `--output-dir`, `--checkpoint-dir`, `--log-dir`
- Other: `--device`, `--num-samples`, etc.

**CLI Arguments to Keep (1):**

- `config_path` (positional, required) - path to JSON config

**Default Merging to Remove:**

- `src/main.py` line ~58: Classifier defaults
- `src/main.py` line ~260: Diffusion defaults
- After: NO fallbacks, error if missing

**Validation Changes:**

- Current: Lenient (allows None)
- Target: Strict (all required fields checked)
- Must: Clear error messages with field paths

---

## Appendix B: Detailed Change Plan

### Phase 1: Core CLI Changes

#### 1.1 Update `src/utils/cli.py`

**`create_parser()` changes:**

```python
# Remove all 20+ specific parameter arguments
# Keep only:
# - config_path (positional argument, required)
# Experiment type will be read from config file
```

**`args_to_dict()` changes:**

```python
# This function may no longer be needed
```

**`parse_args()` changes:**

```python
# Remove defaults parameter
# Get config_path from positional argument
# Load config file: load_config(config_path)
# Read experiment from config['experiment']
# Validate experiment type (classifier/diffusion/gan)
# Return config directly (NO merging)
```

**Effort:** 2-3 hours (entire file refactor)

---

### Phase 2: Main Entry Point

#### 2.1 Update `src/main.py`

**Docstring:** Remove CLI parameter examples, show config-only

**`setup_experiment_classifier()`:**

```python
# Remove: config = merge_configs(classifier_defaults, config)
# Keep: validate_classifier_config(config)
```

**`setup_experiment_diffusion()`:**

```python
# Remove: config = merge_configs(diffusion_defaults, config)
# Keep: validate_diffusion_config(config)
```

**Effort:** 1 hour

---

### Phase 3: Experiment Config Validation

#### 3.1 Update `src/experiments/classifier/config.py`

**Required sections and fields:**

```python
# Top level: experiment, model, data, training, output
# model.*: name, pretrained, num_classes
# data.*: train_path, batch_size, num_workers, image_size, crop_size
# training.*: epochs, learning_rate, optimizer, device
# output.*: checkpoint_dir, log_dir
```

**Effort:** 2 hours

#### 3.2 Update `src/experiments/diffusion/config.py`

**Required sections and fields:**

```python
# Top level: experiment, model, data, training, output
# model.*: image_size, in_channels, model_channels, num_timesteps, beta_schedule
# data.*: train_path, batch_size, num_workers, image_size
# training.*: epochs, learning_rate, optimizer, device
# output.*: checkpoint_dir, log_dir
```

**Effort:** 2 hours

---

### Phase 4-6: Tests, Configs, Documentation

See main TODO list for detailed steps. Key changes:

**Tests:**

- Remove 6+ obsolete test classes (Training, Model, Data, Output, Mode, Device args)
- Add TestConfigOnlyMode class
- Update TestCreateParser for positional arg
- Effort: 5+ hours

**Configs:**

- Audit existing configs for completeness
- Create test fixtures (valid & invalid)
- Effort: 1 hour

**Documentation:**

- README.md: Update Quick Start
- architecture.md: Update CLI Interface section
- Create migration guide
- Effort: 2 hours

**Examples:**

````markdown
### Training a Classifier

```bash
python -m src.main configs/classifier/baseline.json
python -m src.main configs/classifier/inceptionv3.json
```

### Training a Diffusion Model

```bash
python -m src.main configs/diffusion/default.json
```

Note: All parameters must be in the config file. CLI overrides are not supported.
````

---

**End of Document**
