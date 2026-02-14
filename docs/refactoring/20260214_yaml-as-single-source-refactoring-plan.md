# YAML as Single Source of Truth - Refactoring Plan

**Date**: February 14, 2026  
**Status**: Planning  
**Priority**: High  
**Complexity**: Medium

## Executive Summary

This refactoring plan addresses two related issues:

1. **Duplication Problem**: Default configurations are defined in both Python (`get_default_config()`) and YAML (`default.yaml`) files
2. **Directory Structure**: Default YAML files are scattered in `configs/` instead of being colocated with their experiment code

The goal is to:

- Establish YAML files as the single source of truth for default configuration values
- Move `default.yaml` files into `src/experiments/<experiment>/` for better cohesion
- Repurpose `configs/` directory for user-provided configurations only
- Keep Python files focused on configuration loading, validation, and manipulation logic

## Current State Analysis

### Duplicate Configuration Management

**Affected Files:**

1. **Diffusion Experiment**:
   - `src/experiments/diffusion/config.py` - `get_default_config()` returns Python dict
   - `configs/diffusion/default.yaml` - YAML with same default values _(to be moved)_

2. **Classifier Experiment**:
   - `src/experiments/classifier/config.py` - `get_default_config()` returns Python dict
   - `configs/classifier/default.yaml` - YAML with same default values _(to be moved)_

### Problems Identified

1. **Maintenance Burden**: Any default value change requires updating 2 files
2. **Risk of Inconsistency**: Python and YAML defaults can drift apart
3. **Unclear Authority**: No clear single source of truth
4. **Documentation Overhead**: Comments must be maintained in both places
5. **Testing Complexity**: Must validate both sources stay synchronized
6. **Poor Cohesion**: Default configs separated from experiment code in `configs/` directory
7. **Discovery Issues**: Developers must remember separate `configs/` directory exists

### Current Usage Patterns

The `get_default_config()` functions are currently used in:

- Configuration initialization
- Testing (providing baseline configs)
- Validation defaults
- Documentation/examples

## Target State

### YAML as Single Source of Truth

**What YAML files will contain:**

- All default parameter values
- Comprehensive inline documentation (comments)
- Valid, runnable configurations
- Version information (V2 format)

**What Python `config.py` files will contain:**

- `get_default_config()` - loads and returns corresponding `default.yaml`
- `validate_config()` - validation logic only
- `get_model_specific_config()` / `get_resolution_config()` - programmatic overrides
- Helper functions: `resolve_output_path()`, `derive_image_size_from_model()`, etc.
- Migration utilities (if needed)
- Type hints and documentation

### Benefits

1. **Single Source of Truth**: YAML is the authoritative source
2. **User-Friendly**: Non-programmers can read/edit YAML easily
3. **Clear Separation**: Data (YAML) vs. Logic (Python)
4. **Consistent UX**: Users already provide configs via YAML
5. **Better Version Control**: Config changes more visible in diffs
6. **Reduced Code**: Less Python code to maintain
7. **Better Cohesion**: All experiment-related files in one place
8. **Easier Discovery**: Developers see defaults when browsing experiment code
9. **Clearer Purpose**: `configs/` directory for user configs only

## Directory Restructuring

### Current Structure

```
.
├── src/
│   └── experiments/
│       ├── classifier/
│       │   ├── __init__.py
│       │   ├── config.py          # Has hardcoded defaults
│       │   ├── train.py
│       │   └── evaluate.py
│       └── diffusion/
│           ├── __init__.py
│           ├── config.py          # Has hardcoded defaults
│           ├── train.py
│           └── generate.py
└── configs/
    ├── classifier/
    │   ├── default.yaml           # Duplicate defaults
    │   ├── baseline.yaml
    │   └── inceptionv3.yaml
    └── diffusion/
        ├── default.yaml           # Duplicate defaults
        └── legacy.yaml
```

### Target Structure

```
.
├── src/
│   └── experiments/
│       ├── classifier/
│       │   ├── __init__.py
│       │   ├── config.py          # Loads default.yaml
│       │   ├── default.yaml       # ← MOVED HERE (Single source of truth)
│       │   ├── train.py
│       │   └── evaluate.py
│       └── diffusion/
│           ├── __init__.py
│           ├── config.py          # Loads default.yaml
│           ├── default.yaml       # ← MOVED HERE (Single source of truth)
│           ├── train.py
│           └── generate.py
└── configs/
    ├── experiments/               # User experiment configs
    │   ├── diffusion_cifar10.yaml
    │   ├── classifier_resnet50.yaml
    │   └── ablation_study_1.yaml
    ├── classifier/                # Model-specific overrides (keep existing)
    │   ├── baseline.yaml
    │   └── inceptionv3.yaml
    └── diffusion/                 # Experiment-specific overrides (keep existing)
        └── legacy.yaml
```

### Rationale for New Structure

1. **Colocation Principle**: Related files stay together
2. **Package Cohesion**: Each experiment package is self-contained
3. **Clear Ownership**: Experiment owns its defaults
4. **Simpler Imports**: `Path(__file__).parent / "default.yaml"` instead of `../../configs/...`
5. **Better Discovery**: Defaults visible when browsing experiment code
6. **Purpose Clarity**: `configs/` directory is clearly for user-provided configs

## Implementation Plan

### Phase 0: Directory Restructuring (New Phase)

#### Task 0.1: Move default.yaml Files

- Move `configs/diffusion/default.yaml` → `src/experiments/diffusion/default.yaml`
- Move `configs/classifier/default.yaml` → `src/experiments/classifier/default.yaml`
- Verify YAML files are valid after move
- Update `.gitignore` if needed

#### Task 0.2: Document New Directory Purpose

- Update README to explain new structure
- Document that `configs/` is for user configurations
- Add example user config directory structure

#### Task 0.3: Backward Compatibility (Optional)

- Consider keeping symlinks in old location for gradual migration
- Or add fallback logic to check both locations
- Document deprecation timeline

### Phase 1: Preparation (No Breaking Changes)

#### Task 1.1: Add YAML Loading to config.py Files

- Add function to find and load default YAML files
- Handle path resolution (config.py → default.yaml)
- Add caching to avoid repeated file I/O
- Add error handling for missing/invalid YAML

**Files to modify:**

- `src/experiments/diffusion/config.py`
- `src/experiments/classifier/config.py`

#### Task 1.2: Validate YAML Completeness

- Ensure all keys in Python defaults exist in YAML
- Ensure all YAML keys are valid
- Document any intentional differences
- Add validation tests

**Validation script:**

- Create `scripts/validate_default_configs.py`
- Compare Python vs YAML defaults
- Report any mismatches

#### Task 1.3: Add Deprecation Warnings

- Keep existing `get_default_config()` behavior
- Add deprecation warnings when not using YAML
- Update documentation to recommend YAML approach

### Phase 2: Core Refactoring

#### Task 2.1: Refactor get_default_config() - Diffusion

- Modify `src/experiments/diffusion/config.py::get_default_config()`
- Load `src/experiments/diffusion/default.yaml` instead of returning dict
- Use simple path resolution: `Path(__file__).parent / "default.yaml"`
- Maintain backward compatibility (same return type)
- Update docstrings

#### Task 2.2: Refactor get_default_config() - Classifier

- Modify `src/experiments/classifier/config.py::get_default_config()`
- Load `src/experiments/classifier/default.yaml` instead of returning dict
- Use simple path resolution: `Path(__file__).parent / "default.yaml"`
- Maintain backward compatibility (same return type)
- Update docstrings

#### Task 2.3: Remove Hardcoded Defaults

- Remove Python dict definitions from `get_default_config()`
- Keep only YAML loading logic
- Preserve all validation and helper functions
- Update imports if needed

#### Task 2.4: Update Utility Functions

- Ensure `src/utils/config.py` functions work with YAML-sourced defaults
- Update `load_and_merge_configs()` if needed
- Verify `merge_configs()` handles all cases
- Test edge cases

### Phase 3: Testing & Validation

#### Task 3.1: Update Unit Tests

- Update tests in `tests/experiments/diffusion/`
- Update tests in `tests/experiments/classifier/`
- Ensure tests load YAML defaults correctly
- Add tests for missing YAML files
- Add tests for invalid YAML syntax

**Files to update:**

- `tests/experiments/diffusion/test_config.py`
- `tests/experiments/classifier/test_config.py`

#### Task 3.2: Integration Testing

- Test full training workflow with YAML defaults
- Test generation workflow with YAML defaults
- Test config merging (default → user → CLI)
- Test all edge cases

#### Task 3.3: Performance Testing

- Measure YAML loading overhead
- Implement caching if needed
- Ensure no significant performance regression

### Phase 4: Documentation & Migration

#### Task 4.1: Update Documentation

- Update README.md with new approach
- Update configuration guides
- Update API documentation
- Add migration examples

**Files to update:**

- `README.md`
- `docs/refactoring/20260213_diffusion-config-migration-guide.md`
- Any other configuration documentation
- Update references to old `configs/*/default.yaml` paths

#### Task 4.2: Update Examples and Scripts

- Update any example code
- Update migration scripts if affected
- Update CI/CD configurations if needed

#### Task 4.3: Create Migration Guide

- Document changes for users
- Provide before/after examples
- Explain benefits
- Address potential issues

### Phase 5: Cleanup

#### Task 5.1: Remove Deprecation Warnings

- Remove temporary warnings
- Clean up any temporary code
- Final code review

#### Task 5.2: Final Validation

- Run full test suite
- Verify all experiments work
- Check for any remaining hardcoded defaults

## Implementation Tasks (Todo List)

### Phase 0: Directory Restructuring

- [ ] Task 0.1: Move configs/diffusion/default.yaml to src/experiments/diffusion/
- [ ] Task 0.1: Move configs/classifier/default.yaml to src/experiments/classifier/
- [ ] Task 0.1: Verify YAML files are valid after move
- [ ] Task 0.2: Update README with new directory structure
- [ ] Task 0.2: Document configs/ directory purpose
- [ ] Task 0.3: Add backward compatibility if needed

### Phase 1: Preparation

- [ ] Task 1.1: Add YAML loading helper to diffusion config.py
- [ ] Task 1.1: Add YAML loading helper to classifier config.py
- [ ] Task 1.2: Create validation script for default configs
- [ ] Task 1.2: Run validation and fix any YAML/Python mismatches
- [ ] Task 1.3: Add deprecation warnings (if needed)
- [ ] Task 1.3: Update documentation about deprecation

### Phase 2: Core Refactoring

- [ ] Task 2.1: Refactor diffusion get_default_config() to load YAML
- [ ] Task 2.2: Refactor classifier get_default_config() to load YAML
- [ ] Task 2.3: Remove hardcoded defaults from diffusion config.py
- [ ] Task 2.3: Remove hardcoded defaults from classifier config.py
- [ ] Task 2.4: Update src/utils/config.py if needed
- [ ] Task 2.4: Verify all utility functions work correctly

### Phase 3: Testing & Validation

- [ ] Task 3.1: Update diffusion config unit tests
- [ ] Task 3.1: Update classifier config unit tests
- [ ] Task 3.1: Add tests for YAML loading edge cases
- [ ] Task 3.2: Run integration tests for diffusion training
- [ ] Task 3.2: Run integration tests for classifier training
- [ ] Task 3.2: Test config merging logic end-to-end
- [ ] Task 3.3: Performance test YAML loading
- [ ] Task 3.3: Implement caching if needed

### Phase 4: Documentation & Migration

- [ ] Task 4.1: Update README.md
- [ ] Task 4.1: Update configuration documentation
- [ ] Task 4.1: Update API documentation
- [ ] Task 4.2: Review and update example scripts
- [ ] Task 4.2: Update migration scripts
- [ ] Task 4.3: Create user migration guide

### Phase 5: Cleanup

- [ ] Task 5.1: Remove temporary/deprecated code
- [ ] Task 5.1: Final code review
- [ ] Task 5.2: Run full test suite
- [ ] Task 5.2: Verify all experiments work correctly
- [ ] Task 5.2: Update project version

## Testing Strategy

### Unit Tests

**Test Coverage Required:**

1. YAML file loading (valid/invalid/missing)
2. Config validation with YAML-sourced defaults
3. Config merging (YAML defaults + user overrides)
4. Path resolution for YAML files
5. Error handling and messages

### Integration Tests

**Scenarios to Test:**

1. Training workflow with default YAML config
2. Training workflow with user config overriding defaults
3. Generation workflow with YAML defaults
4. CLI argument overrides on top of YAML defaults
5. V1 to V2 migration with YAML defaults

### Regression Tests

**Ensure No Breakage:**

1. Existing user configs still work
2. CLI interface unchanged
3. Config validation catches same errors
4. Output directory structure unchanged
5. All helper functions work correctly

## Migration Path

### Backward Compatibility Strategy

**During Phase 1-2:**

- Existing code continues to work
- No API changes
- `get_default_config()` signature unchanged
- Return type unchanged (still returns dict)

**After Phase 2:**

- All functionality preserved
- Only implementation changes (load YAML instead of hardcoded dict)
- Users see no difference in behavior

### Rollback Plan

**If Issues Arise:**

1. Revert config.py files to hardcoded defaults
2. Keep YAML files for documentation
3. Document issues encountered
4. Plan fixes for next iteration

**Rollback is Easy Because:**

- Git history preserves old implementation
- Changes isolated to config.py files
- No database migrations or data changes
- No API contract changes

## Risk Assessment

### Low Risks ✅

- **YAML syntax errors**: Mitigated by validation in tests
- **Path resolution**: Mitigated by thorough testing across environments
- **Performance**: YAML loading is fast, caching available if needed

### Medium Risks ⚠️

- **Missing YAML files**: Mitigated by clear error messages and fallbacks
- **Config format changes**: Mitigated by validation functions
- **Test suite updates**: Requires careful review but straightforward

### High Risks ❌

- **None identified**: Changes are low-risk and well-isolated

## Timeline Estimate

### Conservative Estimate (Recommended)

- **Phase 0**: 1-2 hours (directory restructuring)
- **Phase 1**: 2-3 hours (preparation and validation)
- **Phase 2**: 3-4 hours (core refactoring)
- **Phase 3**: 3-4 hours (testing and validation)
- **Phase 4**: 2-3 hours (documentation)
- **Phase 5**: 1-2 hours (cleanup)

**Total**: 12-18 hours over 2-3 days

### Aggressive Estimate

- **All Phases**: 7-9 hours in 1 day

**Recommendation**: Use conservative estimate for quality assurance

## Success Criteria

### Functional Requirements ✅

1. All experiments run with YAML-sourced defaults
2. Config merging works correctly
3. Validation catches all errors
4. No behavior changes from user perspective

### Code Quality Requirements ✅

1. No hardcoded defaults in Python config files
2. Clean separation: data (YAML) vs logic (Python)
3. Comprehensive test coverage (>90%)
4. Clear error messages for YAML issues

### Documentation Requirements ✅

1. Updated README and guides
2. Clear inline comments in code
3. Migration guide for developers
4. Updated API documentation

## Next Steps

1. **Review this plan** with the team
2. **Get approval** to proceed
3. **Create feature branch**: `refactor/yaml-single-source-of-truth`
4. **Start Phase 1**: Preparation tasks
5. **Implement incrementally**: Complete each phase before moving on
6. **Test thoroughly**: Run full test suite after each phase
7. **Merge to main**: After all phases complete and validated

## References

- Current implementation: `src/experiments/*/config.py`
- Current YAML location: `configs/*/default.yaml`
- Target YAML location: `src/experiments/*/default.yaml`
- Config utilities: `src/utils/config.py`
- V2 format documentation: `docs/refactoring/20260213_diffusion-config-migration-guide.md`

## Appendix: Key Code Changes

### Before (Current)

```python
def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "experiment": "diffusion",
        "model": {
            "architecture": {
                "image_size": 40,
                # ... many more lines
            }
        }
        # ... 100+ lines of hardcoded config
    }
```

### After (Target)

```python
def get_default_config() -> Dict[str, Any]:
    """Get default configuration by loading default.yaml.

    The default.yaml file is colocated with this module in the same directory,
    following the principle of keeping related files together.

    Returns:
        Dictionary containing default configuration from YAML file

    Raises:
        FileNotFoundError: If default.yaml is not found
        yaml.YAMLError: If YAML is invalid
    """
    # Simple path resolution - default.yaml is in the same directory
    default_yaml = Path(__file__).parent / "default.yaml"

    if not default_yaml.exists():
        raise FileNotFoundError(
            f"Default config not found: {default_yaml}\n"
            f"Expected location: src/experiments/diffusion/default.yaml"
        )

    return load_config(str(default_yaml))
```

## Notes

- This refactoring is **low-risk** with **high value**
- Changes are **isolated** and **reversible**
- **No user-facing changes** in behavior
- Improves **maintainability** significantly
- Aligns with **ML project best practices**

---

**Plan prepared by**: GitHub Copilot  
**Date**: February 14, 2026  
**Review required**: Yes  
**Approval required**: Yes
