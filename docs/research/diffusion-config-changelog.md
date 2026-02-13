# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING**: Restructured diffusion configuration for better clarity and organization
  - Moved `device` and `seed` to top level (common parameters used in both training and generation)
  - Moved checkpointing parameters (`checkpoint_dir`, `save_frequency`, `save_best_only`) to `training` section
  - Nested `validation` under `training` (was previously at top level)
  - Created new `training.visualization` section for training-time sampling (moved from `generation`)
  - Moved generation-specific parameters (`checkpoint`, `num_samples`, `output_dir`) to `generation` section
- Updated all configuration files to new structure (`configs/diffusion/*.yaml`)
- Updated all tests to use new configuration structure (842 tests passing)

### Added

- Mode-aware configuration validation for diffusion models
  - Training mode validates training-specific parameters
  - Generation mode requires checkpoint and validates generation-specific parameters
- New example configuration files:
  - `configs/diffusion/default.yaml` - Complete training configuration example
  - `tests/fixtures/configs/diffusion/valid_minimal.yaml` - Minimal valid configuration
- Comprehensive migration guide: `docs/research/diffusion-config-migration-guide.md`
- Configuration reference in README with examples of new structure
- Helper validation functions:
  - `_validate_model_config()` - Model parameter validation
  - `_validate_data_config()` - Data parameter validation
  - `_validate_training_config()` - Training parameter validation
  - `_validate_generation_config()` - Generation parameter validation

### Deprecated

- Old diffusion configuration structure (removed as of February 2026)
  - Configurations from before February 2026 must be migrated to new structure
  - See `docs/research/diffusion-config-migration-guide.md` for migration instructions

### Documentation

- Added diffusion configuration section to README.md with complete examples
- Created comprehensive migration guide for updating existing configurations
- Updated docstrings in `src/experiments/diffusion/config.py` to reflect new structure
- Added configuration validation examples

### Technical Details

**Impact:** High - All diffusion configurations must be updated

**Migration Path:**

1. Review the migration guide: `docs/research/diffusion-config-migration-guide.md`
2. Update your configuration files following the before/after examples
3. Validate updated configs with `validate_config()` function
4. Test training/generation with updated configs

**Testing:** All 842 tests passing, including:

- 73 unit tests for configuration validation
- 840+ integration tests for end-to-end workflows
- Mode-aware validation for both training and generation modes

**References:**

- Optimization Report: `docs/research/diffusion-config-optimization-report.md`
- Implementation Tasks: `docs/research/diffusion-config-implementation-tasks.md`
- Migration Guide: `docs/research/diffusion-config-migration-guide.md`

---

## [0.1.0] - 2026-02-10

### Added

- Initial project setup with modular architecture
- Base classes for trainers, models, and data loaders
- Four-tier testing strategy (unit, component, integration, smoke)
- Experiment implementations:
  - Classifier experiments (InceptionV3, ResNet50/101, WRN)
  - Diffusion experiments (DDPM)
- Configuration management with YAML files
- Comprehensive test infrastructure
- Documentation:
  - Architecture specification
  - Refactoring plan
  - User and technical requirements

### Features

- CPU-first development approach
- Configuration-driven workflows
- Metrics and evaluation utilities (FID, IS, PR-AUC, ROC-AUC)
- Device management utilities
- Docker support for consistent environments

---

**Note:** This changelog tracks changes starting from the structured refactoring phase (February 2026). Earlier development history is available in git commits.
