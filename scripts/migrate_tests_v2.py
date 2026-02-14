#!/usr/bin/env python3
"""Migration script for updating diffusion tests to V2 config format."""

import re
import sys
from pathlib import Path


def migrate_test_file(file_path: Path) -> None:
    """Migrate a test file to V2 config format."""
    content = file_path.read_text()
    original = content

    # Replace test assertions and expectations for V2 structure
    replacements = [
        # Device at top level -> compute.device
        (
            r'assert "device" in config',
            'assert "compute" in config\n    assert "device" in config["compute"]',
        ),
        (r'assert config\["device"\]', 'assert config["compute"]["device"]'),
        (r'config\["device"\]', 'config["compute"]["device"]'),
        (r"test_device_at_top_level", "test_device_in_compute_section"),
        # Seed at top level -> compute.seed
        (r'assert "seed" in config', 'assert "seed" in config["compute"]'),
        (r'config\["seed"\]', 'config["compute"]["seed"]'),
        (r"test_seed_at_top_level", "test_seed_in_compute_section"),
        # Model structure
        (r'model\["image_size"\]', 'model["architecture"]["image_size"]'),
        (r'model\["in_channels"\]', 'model["architecture"]["in_channels"]'),
        (r'model\["model_channels"\]', 'model["architecture"]["model_channels"]'),
        (
            r'model\["channel_multipliers"\]',
            'model["architecture"]["channel_multipliers"]',
        ),
        (r'model\["use_attention"\]', 'model["architecture"]["use_attention"]'),
        (r'model\["num_timesteps"\]', 'model["diffusion"]["num_timesteps"]'),
        (r'model\["beta_schedule"\]', 'model[" diffusion"]["beta_schedule"]'),
        (r'model\["beta_start"\]', 'model["diffusion"]["beta_start"]'),
        (r'model\["beta_end"\]', 'model["diffusion"]["beta_end"]'),
        (r'model\["num_classes"\]', 'model["conditioning"]["num_classes"]'),
        (
            r'model\["class_dropout_prob"\]',
            'model["conditioning"]["class_dropout_prob"]',
        ),
        # Data structure
        (r'data\["train_path"\]', 'data["paths"]["train"]'),
        (r'data\["val_path"\]', 'data["paths"]["val"]'),
        (r'data\["batch_size"\]', 'data["loading"]["batch_size"]'),
        (r'data\["num_workers"\]', 'data["loading"]["num_workers"]'),
        (r'data\["pin_memory"\]', 'data["loading"]["pin_memory"]'),
        (r'data\["shuffle_train"\]', 'data["loading"]["shuffle_train"]'),
        (r'data\["drop_last"\]', 'data["loading"]["drop_last"]'),
        (r'data\["horizontal_flip"\]', 'data["augmentation"]["horizontal_flip"]'),
        (r'data\["rotation_degrees"\]', 'data["augmentation"]["rotation_degrees"]'),
        (r'data\["color_jitter"\]', 'data["augmentation"]["color_jitter"]["enabled"]'),
        (
            r'data\["color_jitter_strength"\]',
            'data["augmentation"]["color_jitter"]["strength"]',
        ),
        # Output structure
        (r'output\["log_dir"\]', 'output["base_dir"]'),  # Note: this is approximate
        # Training structure
        (r'training\["learning_rate"\]', 'training["optimizer"]["learning_rate"]'),
        (r'training\["optimizer"\]', 'training["optimizer"]["type"]'),
        (r'training\["use_ema"\]', 'training["ema"]["enabled"]'),
        (r'training\["ema_decay"\]', 'training["ema"]["decay"]'),
        (r'training\["use_amp"\]', 'training["performance"]["use_amp"]'),
        (
            r'training\["checkpoint_dir"\]',
            'output["subdirs"]["checkpoints"]',
        ),  # Approximate
        (
            r'training\["save_frequency"\]',
            'training["checkpointing"]["save_frequency"]',
        ),
        (
            r'training\["save_best_only"\]',
            'training["checkpointing"]["save_best_only"]',
        ),
        # Generation structure
        (r'generation\["num_samples"\]', 'generation["sampling"]["num_samples"]'),
        (r'generation\["guidance_scale"\]', 'generation["sampling"]["guidance_scale"]'),
        (r'generation\["use_ema"\]', 'generation["sampling"]["use_ema"]'),
        (r'generation\["save_grid"\]', 'generation["output"]["save_grid"]'),
        (r'generation\["grid_nrow"\]', 'generation["output"]["grid_nrow"]'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    if content != original:
        file_path.write_text(content)
        print(f"Updated: {file_path}")
        return True
    return False


def main():
    # Find all test files related to diffusion
    test_files = [
        "tests/experiments/diffusion/test_config.py",
    ]

    updated = 0
    for test_file in test_files:
        path = Path(test_file)
        if path.exists():
            if migrate_test_file(path):
                updated += 1
        else:
            print(f"Warning: {test_file} not found")

    print(f"\nUpdated {updated} test file(s)")


if __name__ == "__main__":
    main()
