# Refactoring Implementation Plan

This document outlines a step-by-step implementation plan to refactor the codebase according to the [Architecture Specification](../standards/architecture.md). Each step is designed to be small enough for a single feature branch.

## Overview

The refactoring follows a **Vertical Slice + Base Class** architecture pattern with:

- Self-contained experiment types (GAN, Diffusion, Classifier)
- Shared base classes for common functionality
- Four-tier testing strategy (unit, component, integration, smoke)
- Configuration-driven experiments with CLI support

## Implementation Phases

### Phase 1: Project Foundation

#### Step 1: Initial Project Setup

- [x] Create repository structure (root folders only)
- [x] Add `.gitignore` for Python, PyTorch, outputs
- [x] Create `requirements.txt` with core dependencies
- [x] Create `requirements-dev.txt` with test dependencies
- [x] Add basic `README.md` with project overview

#### Step 2: Base Directory Structure

- [x] Create `src/` folder with `__init__.py`
- [x] Create `src/base/` folder with `__init__.py`
- [x] Create `src/utils/` folder with `__init__.py`
- [x] Create `src/data/` folder with `__init__.py`
- [x] Create `src/experiments/` folder with `__init__.py`

#### Step 3: Test Infrastructure Setup

- [x] Create `tests/` folder with `conftest.py`
- [x] Create `tests/fixtures/` for test data
- [x] Configure pytest markers in `pytest.ini`
- [x] Add test running documentation to README

### Phase 2: Utilities Layer

#### Step 4: Device Management Utility

- [x] Implement `src/utils/device.py` with CPU/GPU detection
- [x] Add `tests/utils/test_device.py` with unit tests
- [x] Ensure CPU-only testing works

#### Step 5: Configuration Management

- [x] Implement `src/utils/config.py` for JSON loading
- [x] Add config merging logic (file → defaults)
- [x] Create `tests/fixtures/configs/` with sample configs
- [x] Add `tests/utils/test_config.py` with unit tests

#### Step 6: CLI Argument Parsing

- [x] Implement `src/utils/cli.py` with argument parser
- [x] Add CLI override logic (CLI → config → defaults)
- [x] Add `tests/utils/test_cli.py` with unit tests

#### Step 7: Common Metrics

- [x] Implement `src/utils/metrics.py` with basic metrics
- [x] Add FID, IS, PR-AUC, ROC-AUC placeholders
- [x] Add `tests/utils/test_metrics.py` with unit tests

### Phase 3: Data Layer

#### Step 8: Data Transforms

- [x] Implement `src/data/transforms.py` with common augmentations
- [x] Add normalization utilities
- [x] Add `tests/data/test_transforms.py` with unit tests

#### Step 9: Dataset Implementations

- [x] Implement `src/data/datasets.py` with base dataset class
- [x] Add image folder dataset implementation
- [x] Add `tests/data/test_datasets.py` with component tests
- [x] Create mock data in `tests/fixtures/mock_data/`

#### Step 10: Custom Samplers

- [x] Implement `src/data/samplers.py` (if needed)
- [x] Add tests for samplers
- [x] Document sampler usage

### Phase 4: Base Classes

#### Step 11: Base Model Interface

- [x] Implement `src/base/model.py` as ABC
- [x] Define required methods (forward, loss, save, load)
- [x] Add `tests/base/test_model.py` with interface tests

#### Step 12: Base DataLoader Interface

- [x] Implement `src/base/dataloader.py` as ABC
- [x] Define required methods (get_train_loader, get_val_loader)
- [x] Add `tests/base/test_dataloader.py` with interface tests

#### Step 13: Base Logger Interface

- [x] Implement `src/base/logger.py` as ABC
- [x] Define logging methods (log_metrics, log_images)
- [x] Add `tests/base/test_logger.py` with unit tests

#### Step 14: Base Trainer

- [x] Implement `src/base/trainer.py` with abstract training loop
- [x] Add checkpoint save/load logic
- [x] Add validation loop structure
- [x] Add `tests/base/test_trainer.py` with component tests

### Phase 5: First Experiment Slice (Classifier)

#### Step 15: Classifier Experiment Structure

- [x] Create `src/experiments/classifier/` folder
- [x] Add `__init__.py` and structure files
- [x] Create `tests/experiments/classifier/` folder

#### Step 16: Classifier Models - InceptionV3

- [x] Create `src/experiments/classifier/models/` folder
- [x] Implement `inceptionv3.py` wrapper
- [x] Add model instantiation logic
- [x] Add `tests/experiments/classifier/models/test_inceptionv3.py` with unit tests

#### Step 16.5: InceptionV3 Selective Layer Freezing

- [x] Add `set_trainable_layers()` method to InceptionV3Classifier
- [x] Support layer name patterns for selective unfreezing (e.g., "Mixed_5", "Mixed_6")
- [x] Add `trainable_layers` parameter to model initialization
- [x] Update model to handle both `freeze_backbone` and `trainable_layers` configuration
- [x] Add unit tests for selective layer freezing
- [x] Add component tests verifying gradient flow through selected layers
- [x] Update model documentation with selective freezing examples

#### Step 17: Classifier Models - ResNet

- [x] Implement `src/experiments/classifier/models/resnet.py`
- [x] Add ResNet50, ResNet101, ResNet152 variants
- [x] Add tests for ResNet models

#### Step 18: Classifier DataLoader

- [x] Implement `src/experiments/classifier/dataloader.py`
- [x] Inherit from base dataloader
- [x] Add classification-specific preprocessing
- [x] Add `tests/experiments/classifier/test_dataloader.py`

#### Step 19: Classifier Logger

- [x] Implement `src/experiments/classifier/logger.py`
- [x] Add classification metrics logging
- [x] Add confusion matrix logging
- [x] Add tests for logger

#### Step 20: Classifier Trainer

- [x] Implement `src/experiments/classifier/trainer.py`
- [x] Inherit from base trainer
- [x] Add classification training loop
- [x] Add `tests/experiments/classifier/test_trainer.py`

#### Step 21: Classifier Configuration

- [x] Create `src/experiments/classifier/config.py` with defaults
- [x] Create `configs/classifier/baseline.json`
- [x] Create `configs/classifier/inceptionv3.json`
- [x] Add config validation tests

#### Step 22: Classifier Analysis Tools

- [x] Implement `src/experiments/classifier/analyze_comparison.py`
- [x] Port analysis logic from old codebase
- [x] Add tests for analysis tools

### Phase 6: Main Entry Point

#### Step 23: CLI Entry Point

- [x] Implement `src/main.py` with experiment dispatcher
- [x] Add experiment selection logic
- [x] Wire up classifier experiment
- [x] Add integration tests for CLI

#### Step 24: End-to-End Classifier Test

- [x] Create `tests/integration/test_classifier_pipeline.py`
- [x] Test full workflow: config → train → checkpoint
- [x] Use tiny dataset (10-20 images)

#### Step 24.5: Fix Integration Test Issues

- [x] Add scheduler parameter support to `ClassifierTrainer.__init__()`
- [x] Fix trainer initialization in all failing test cases
- [x] Update test_checkpoint_save_and_load to fix variable scoping issues
- [x] Update test_training_resumption_from_checkpoint API usage
- [x] Fix test_pipeline_with_cosine_scheduler trainer initialization
- [x] Update test_validation_metrics_recorded to use correct train() signature
- [x] Fix test_pipeline_resnet101 and test_pipeline_with_different_optimizers
- [x] Update test_config_file_driven_pipeline metrics assertion logic
- [x] Verify all 10 integration tests pass on CPU
- [x] Run all integration tests on GPU to validate CUDA operations

### Phase 7: Second Experiment Slice (Diffusion)

#### Step 25: Diffusion Experiment Structure

- [ ] Create `src/experiments/diffusion/` folder structure
- [ ] Create `tests/experiments/diffusion/` folder

#### Step 26: Diffusion Model

- [ ] Implement `src/experiments/diffusion/model.py`
- [ ] Port DDPM logic from old codebase
- [ ] Add `tests/experiments/diffusion/test_model.py`

#### Step 27: Diffusion DataLoader

- [ ] Implement `src/experiments/diffusion/dataloader.py`
- [ ] Add diffusion-specific preprocessing
- [ ] Add tests

#### Step 28: Diffusion Logger

- [ ] Implement `src/experiments/diffusion/logger.py`
- [ ] Add generated image logging
- [ ] Add diffusion metrics logging
- [ ] Add tests

#### Step 29: Diffusion Trainer

- [ ] Implement `src/experiments/diffusion/trainer.py`
- [ ] Add diffusion training loop
- [ ] Add generation mode logic
- [ ] Add tests

#### Step 30: Diffusion Configuration

- [ ] Create `src/experiments/diffusion/config.py`
- [ ] Create `configs/diffusion/default.json`
- [ ] Add config tests

#### Step 31: Wire Diffusion to Main

- [ ] Add diffusion experiment to `src/main.py`
- [ ] Add integration test
- [ ] Test generation mode

### Phase 8: Third Experiment Slice (GAN)

#### Step 32: GAN Experiment Structure

- [ ] Create `src/experiments/gan/` folder structure
- [ ] Create `tests/experiments/gan/` folder

#### Step 33: GAN Models

- [ ] Implement `src/experiments/gan/model.py`
- [ ] Add Generator and Discriminator
- [ ] Add tests

#### Step 34: GAN DataLoader

- [ ] Implement `src/experiments/gan/dataloader.py`
- [ ] Add tests

#### Step 35: GAN Logger

- [ ] Implement `src/experiments/gan/logger.py`
- [ ] Add GAN-specific metrics
- [ ] Add tests

#### Step 36: GAN Trainer

- [ ] Implement `src/experiments/gan/trainer.py`
- [ ] Add adversarial training loop
- [ ] Add tests

#### Step 37: GAN Configuration

- [ ] Create `src/experiments/gan/config.py`
- [ ] Create `configs/gan/default.json`
- [ ] Add config tests

#### Step 38: Wire GAN to Main

- [ ] Add GAN experiment to `src/main.py`
- [ ] Add integration test

### Phase 9: Documentation & Polish

#### Step 39: Comprehensive Documentation

- [ ] Document all CLI commands with examples
- [ ] Add architecture diagram to README
- [ ] Document testing strategy in README
- [ ] Add contribution guidelines

#### Step 40: Example Workflows

- [ ] Create example workflow scripts
- [ ] Document typical research workflow
- [ ] Add quickstart guide

#### Step 41: Smoke Tests (Optional)

- [ ] Create `tests/smoke/test_gpu_training.py`
- [ ] Add performance benchmarks
- [ ] Document GPU testing procedures

#### Step 42: Migration Documentation

- [ ] Document migration from old codebase
- [ ] Create migration scripts if needed
- [ ] Archive old code in `src/old/`

### Phase 10: Final Integration

#### Step 43: Pre-trained Model Integration

- [ ] Document model weight loading
- [ ] Add weight download scripts
- [ ] Test weight compatibility

#### Step 44: Output Management

- [ ] Create `.gitignore` entries for `outputs/`
- [ ] Document output structure
- [ ] Add output cleanup utilities

#### Step 45: CI/CD Setup (Optional)

- [ ] Create GitHub Actions workflow
- [ ] Configure pytest runs on PR
- [ ] Add linting and formatting checks

#### Step 46: Final Testing & Validation

- [ ] Run all test tiers
- [ ] Verify CPU-only testing works
- [ ] Test on actual GPU (smoke tests)
- [ ] Validate all experiments work end-to-end

## Timeline Estimates

- **Phases 1-2**: ~1 week (foundation + utilities)
- **Phases 3-4**: ~1 week (data + base classes)
- **Phase 5**: ~2 weeks (first complete experiment)
- **Phase 6**: ~3 days (main entry point)
- **Phase 7**: ~1 week (second experiment)
- **Phase 8**: ~1 week (third experiment)
- **Phases 9-10**: ~1 week (polish + integration)

**Total: ~7-8 weeks** for complete implementation

## Branching Strategy

Each step should be implemented in a separate feature branch following this naming convention:

```
refactor/<phase>-<step>-<short-description>

Examples:
- refactor/phase1-step1-project-setup
- refactor/phase2-step4-device-management
- refactor/phase5-step16-classifier-inceptionv3
```

### Branch Workflow

1. Create feature branch from `main`
2. Implement the step with tests
3. Ensure all tests pass (`pytest -m "unit or component"`)
4. Create pull request
5. Review and merge to `main`
6. Delete feature branch

## Testing Requirements

Each step must include appropriate tests based on the four-tier strategy:

### Unit Tests (Required for all steps)

- Fast (< 100ms per test)
- CPU only
- No external dependencies
- Test pure logic and interfaces

### Component Tests (Required for model/trainer steps)

- Medium speed (1-5 seconds)
- CPU with small data
- Test integration points
- Verify shapes and basic behavior

### Integration Tests (Required for experiment completion)

- Slower (10-60 seconds)
- Mini workflows
- Test end-to-end pipelines
- Use tiny datasets (10-20 samples)

### Smoke Tests (Optional, run manually)

- Very slow (5-15 minutes)
- GPU required
- Real hardware validation
- Performance benchmarks

## Success Criteria

Each step is considered complete when:

1. ✅ All code is implemented according to architecture spec
2. ✅ All appropriate tests are written and passing
3. ✅ Code follows project style guidelines
4. ✅ Documentation is updated (docstrings, README)
5. ✅ Pull request is reviewed and approved
6. ✅ No regressions in existing functionality
7. ✅ Tests can run on CPU without GPU

## Dependencies Between Steps

### Critical Path

Steps must be completed in order within each phase, but phases can overlap:

- **Phase 1** must complete before any other phase
- **Phase 2** must complete before Phase 3-4
- **Phase 3-4** must complete before Phase 5
- **Phase 5** must complete before Phase 6
- **Phase 6** must complete before Phase 7-8

### Parallelization Opportunities

These can be worked on in parallel:

- **Phase 7 and Phase 8** (Diffusion and GAN) can be done simultaneously
- **Steps within Phase 5** (Classifier components) can be parallelized if multiple developers
- **Documentation (Phase 9)** can start as soon as Phase 6 completes

## Risk Mitigation

### Potential Blockers

1. **Old Code Compatibility**: May need to refactor old code significantly
   - Mitigation: Keep `src/old/` as reference, port incrementally

2. **Test Data Requirements**: Need appropriate test datasets
   - Mitigation: Create synthetic/mock data in `tests/fixtures/`

3. **GPU Testing**: Limited GPU access for smoke tests
   - Mitigation: Focus on CPU tests, run smoke tests manually/weekly

4. **Configuration Complexity**: Managing config merging might be tricky
   - Mitigation: Implement config utilities early (Step 5)

### Quality Gates

Before merging any step:

1. All unit tests pass on CI
2. Code coverage >= 80% for new code
3. No lint errors
4. Documentation updated
5. Peer review completed

## Notes

- Each step should take 1-3 days of focused work
- Tests are not optional - they're part of the step definition
- Keep commits atomic and well-documented
- Update this document as the refactor progresses
- Archive old code rather than deleting it
- Reference existing code in `src/old/` for logic to port

## Progress Tracking

Current Phase: **Phase 6: Main Entry Point**  
Current Step: **Step 24.5: Fix Integration Test Issues**  
Completed Steps: 24/47 (including Step 24.5)  
Last Updated: 2026-02-11

---

For detailed architecture specifications, see [Architecture Specification](../standards/architecture.md).
