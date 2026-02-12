# Test Fixtures

This directory contains test fixtures used across the test suite.

## Directory Structure

```
fixtures/
├── configs/          # Sample configuration files for testing
│   ├── classifier_minimal.yaml
│   ├── diffusion_minimal.yaml
│   └── gan_minimal.yaml
└── mock_data/        # Mock datasets for testing (created by conftest.py fixtures)
```

## Usage

### Configuration Fixtures

The `configs/` directory contains minimal configuration files for each experiment type. These are used in tests to validate configuration loading and provide realistic test scenarios.

- **classifier_minimal.yaml**: Minimal classifier configuration with 2 epochs, batch size 2, CPU device
- **diffusion_minimal.yaml**: Minimal diffusion model configuration for DDPM training
- **gan_minimal.yaml**: Minimal GAN configuration for adversarial training

All test configs are designed for:

- Fast execution (2 epochs, small batch sizes)
- CPU-only testing
- Minimal resource usage
- Deterministic behavior

### Mock Data

Mock data is created dynamically by fixtures in `conftest.py`:

- `mock_dataset_small`: 4 images (2 per class) for unit/component tests
- `mock_dataset_medium`: 20 images (10 per class) for integration tests

These fixtures automatically create temporary directories with proper structure:

```
tmp_data_dir/
├── 0.Normal/
│   ├── img_000.jpg
│   └── img_001.jpg
└── 1.Abnormal/
    ├── img_002.jpg
    └── img_003.jpg
```

## Adding New Fixtures

### Configuration Fixtures

When adding a new experiment type or variant:

1. Create a minimal JSON config with required fields only
2. Set `epochs: 2`, `batch_size: 2`, `device: "cpu"`
3. Use realistic but minimal parameters
4. Document the purpose in this README

### Data Fixtures

When adding new data fixtures:

1. Add fixture function to `conftest.py`
2. Use `tmp_data_dir` or `tmp_path` for temporary data
3. Keep data size minimal (< 100 images for integration tests)
4. Ensure deterministic generation (fixed seeds)

## Best Practices

- **Keep it minimal**: Fixtures should be just large enough to test functionality
- **Fast creation**: Fixture setup should take < 1 second
- **Isolated**: Each test gets fresh fixtures via function scope
- **Documented**: Update this README when adding new fixtures
