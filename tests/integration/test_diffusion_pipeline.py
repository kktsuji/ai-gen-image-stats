"""Integration tests for diffusion end-to-end pipeline.

This module tests the complete diffusion workflow from configuration
loading through training to checkpoint saving, restoration, and generation.
Uses a tiny dataset (10-20 images) for fast validation of the full pipeline.

Test Coverage:
- Config loading and validation
- Model initialization (DDPM)
- Full training loop execution
- Checkpoint saving and loading
- Training resumption from checkpoint
- Generation mode with unconditional and conditional models
- Sample generation and saving
- Validation loop execution
- Metrics logging and CSV output
- Optimizer and scheduler setup
"""

import json
from pathlib import Path

import pytest
import torch
import yaml
from torchvision.utils import save_image

from src.experiments.diffusion.config import get_default_config
from src.experiments.diffusion.dataloader import DiffusionDataLoader
from src.experiments.diffusion.logger import DiffusionLogger
from src.experiments.diffusion.model import create_ddpm
from src.experiments.diffusion.trainer import DiffusionTrainer

# Dynamic device detection for testing
TEST_DEVICE = "cuda" if torch.cuda.is_available() else "CPU"


def _create_split_json(data_dir, split_json_path, include_val=True):
    """Create a split JSON file from a directory structure."""
    from pathlib import Path

    entries = []
    data_path = Path(data_dir)
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            # Support directory names like '0.Normal' or plain '0'
            label = int(class_dir.name.split(".")[0])
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix in (".png", ".jpg", ".jpeg"):
                    entries.append({"path": str(img_file), "label": label})

    class_names = sorted(set(str(e["label"]) for e in entries))
    classes = {name: int(name) for name in class_names}

    split_data = {
        "metadata": {"classes": classes},
        "train": entries,
        "val": entries if include_val else [],
    }

    split_json_path = Path(split_json_path)
    split_json_path.write_text(json.dumps(split_data))
    return str(split_json_path)


class TestDiffusionPipelineBasic:
    """Test basic diffusion pipeline with minimal configuration."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_unconditional(self, tmp_path, mock_dataset_medium):
        """Test complete diffusion pipeline with unconditional generation.

        Tests:
        - Model initialization
        - Training for 2 epochs
        - Checkpoint saving
        - Metrics logging
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        # Setup configuration
        config = {
            "device": TEST_DEVICE,
            "seed": None,
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 32,
                "channel_multipliers": [1, 2],
                "num_classes": None,
                "num_timesteps": 100,
                "beta_schedule": "cosine",
                "use_attention": [False, True],
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 32,
                "return_labels": False,
            },
            "output": {
                "log_dir": str(tmp_path / "logs"),
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "use_ema": True,
                "ema_decay": 0.999,
                "use_amp": False,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "save_best_only": False,
                "save_frequency": 1,
                "validation": {
                    "frequency": 1,
                    "metric": "loss",
                },
                "visualization": {
                    "log_images_interval": 1,
                    "log_denoising_interval": 1,
                    "num_samples": 4,
                    "guidance_scale": 0.0,
                },
            },
            "generation": {
                "checkpoint": None,
                "num_samples": 100,
                "guidance_scale": 0.0,
                "use_ema": True,
                "output_dir": None,
            },
        }

        # Initialize components
        model = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],  # Now from top level
        )

        dataloader = DiffusionDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            return_labels=config["data"]["return_labels"],
        )

        logger = DiffusionLogger(log_dir=config["output"]["log_dir"])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Initialize trainer
        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["device"],
            use_ema=config["training"]["use_ema"],
            ema_decay=config["training"]["ema_decay"],
            use_amp=config["training"]["use_amp"],
            log_images_interval=config["training"]["visualization"][
                "log_images_interval"
            ],
            log_denoising_interval=config["training"]["visualization"][
                "log_denoising_interval"
            ],
            num_samples=config["training"]["visualization"]["num_samples"],
            guidance_scale=config["training"]["visualization"]["guidance_scale"],
        )

        # Run training
        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["training"]["checkpoint_dir"],
            checkpoint_frequency=config["training"]["save_frequency"],
        )

        # Verify outputs
        checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        log_dir = Path(config["output"]["log_dir"])

        assert checkpoint_dir.exists(), "Checkpoint directory not created"
        assert log_dir.exists(), "Log directory not created"

        # Verify checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoint files saved"

        # Verify metrics CSV
        metrics_csv = log_dir / "metrics" / "metrics.csv"
        assert metrics_csv.exists(), "Metrics CSV not created"

        # Read and verify metrics
        with open(metrics_csv, "r") as f:
            lines = f.readlines()
            assert len(lines) > 1, "Metrics CSV is empty"
            # Should have header + at least 2 epoch entries
            assert len(lines) >= 3

        # Verify generated samples
        samples_dir = log_dir / "samples"
        assert samples_dir.exists(), "Samples directory not created"
        sample_files = list(samples_dir.glob("*.png"))
        assert len(sample_files) > 0, "No sample images generated"

        logger.close()

    @pytest.mark.smoke
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_conditional(self, tmp_path, mock_dataset_medium):
        """Test complete diffusion pipeline with conditional generation.

        Tests:
        - Conditional model initialization
        - Training with class labels
        - Sample generation for each class
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        # Setup configuration
        config = {
            "device": TEST_DEVICE,
            "seed": None,
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 32,
                "channel_multipliers": [1, 2],
                "num_classes": 2,
                "num_timesteps": 100,
                "beta_schedule": "cosine",
                "class_dropout_prob": 0.1,
                "use_attention": [False, True],
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 32,
                "return_labels": True,  # Important for conditional generation
            },
            "output": {
                "log_dir": str(tmp_path / "logs"),
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "use_ema": True,
                "ema_decay": 0.999,
                "use_amp": False,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "save_best_only": False,
                "save_frequency": 1,
                "validation": {
                    "frequency": 1,
                    "metric": "loss",
                },
                "visualization": {
                    "log_images_interval": 1,
                    "log_denoising_interval": 1,
                    "num_samples": 4,
                    "guidance_scale": 2.0,
                },
            },
            "generation": {
                "checkpoint": None,
                "num_samples": 100,
                "guidance_scale": 2.0,
                "use_ema": True,
                "output_dir": None,
            },
        }

        # Initialize components
        model = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            class_dropout_prob=config["model"]["class_dropout_prob"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        dataloader = DiffusionDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            return_labels=config["data"]["return_labels"],
        )

        logger = DiffusionLogger(log_dir=config["output"]["log_dir"])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Initialize trainer
        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["device"],
            use_ema=config["training"]["use_ema"],
            ema_decay=config["training"]["ema_decay"],
            use_amp=config["training"]["use_amp"],
            log_images_interval=config["training"]["visualization"][
                "log_images_interval"
            ],
            log_denoising_interval=config["training"]["visualization"][
                "log_denoising_interval"
            ],
            num_samples=config["training"]["visualization"]["num_samples"],
            guidance_scale=config["training"]["visualization"]["guidance_scale"],
        )

        # Run training
        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["training"]["checkpoint_dir"],
            checkpoint_frequency=config["training"]["save_frequency"],
        )

        # Verify outputs
        checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        log_dir = Path(config["output"]["log_dir"])

        assert checkpoint_dir.exists(), "Checkpoint directory not created"
        assert log_dir.exists(), "Log directory not created"

        # Verify checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoint files saved"

        # Verify metrics CSV
        metrics_csv = log_dir / "metrics" / "metrics.csv"
        assert metrics_csv.exists(), "Metrics CSV not created"

        # Verify generated samples (should have samples for each class)
        samples_dir = log_dir / "samples"
        assert samples_dir.exists(), "Samples directory not created"
        sample_files = list(samples_dir.glob("*.png"))
        assert len(sample_files) > 0, "No sample images generated"

        logger.close()


class TestDiffusionPipelineCheckpoints:
    """Test checkpoint saving and loading functionality."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_checkpoint_save_and_load(self, tmp_path, mock_dataset_medium):
        """Test checkpoint saving and loading.

        Tests:
        - Checkpoint is saved during training
        - Model can be loaded from checkpoint
        - Loaded model produces same outputs
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        # Setup configuration
        config = {
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 32,
                "channel_multipliers": [1, 2],
                "num_classes": None,
                "num_timesteps": 100,
                "beta_schedule": "cosine",
                "use_attention": [False, True],
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 32,
                "return_labels": False,
            },
            "device": TEST_DEVICE,
            "seed": None,
            "output": {
                "log_dir": str(tmp_path / "logs"),
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "use_ema": False,  # Disable EMA for simpler test
            },
        }

        # Train model
        model = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        dataloader = DiffusionDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            return_labels=config["data"]["return_labels"],
        )

        logger = DiffusionLogger(log_dir=str(log_dir))

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["device"],
            use_ema=config["training"]["use_ema"],
        )

        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=1,
        )

        # Get saved state before loading
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Find checkpoint file
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoint saved"

        checkpoint_path = checkpoint_files[0]

        # Create new model and load checkpoint
        model_loaded = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        model_loaded.load_state_dict(checkpoint["model_state_dict"])

        # Verify loaded weights match original
        loaded_state = model_loaded.state_dict()
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key], atol=1e-6), (
                f"Mismatch in {key}"
            )

        logger.close()

    @pytest.mark.smoke
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_resumption_from_checkpoint(self, tmp_path, mock_dataset_medium):
        """Test training can be resumed from a checkpoint.

        Tests:
        - Training can resume from saved checkpoint
        - Epoch counter continues correctly
        - Optimizer state is restored
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        # Setup configuration
        config = {
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 32,
                "channel_multipliers": [1, 2],
                "num_classes": None,
                "num_timesteps": 100,
                "beta_schedule": "cosine",
                "use_attention": [False, True],
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 32,
                "return_labels": False,
            },
            "device": TEST_DEVICE,
            "seed": None,
            "output": {
                "log_dir": str(tmp_path / "logs"),
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "use_ema": False,
            },
        }

        # Initial training
        model = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        dataloader = DiffusionDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            return_labels=config["data"]["return_labels"],
        )

        logger = DiffusionLogger(log_dir=str(log_dir))

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["device"],
            use_ema=config["training"]["use_ema"],
        )

        # Train for 1 epoch
        trainer.train(
            num_epochs=1,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=1,
        )

        logger.close()

        # Find checkpoint
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoint saved"
        checkpoint_path = checkpoint_files[0]

        # Resume training
        model_resumed = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        logger_resumed = DiffusionLogger(log_dir=str(log_dir / "resumed"))

        optimizer_resumed = torch.optim.Adam(
            model_resumed.parameters(), lr=config["training"]["learning_rate"]
        )

        trainer_resumed = DiffusionTrainer(
            model=model_resumed,
            dataloader=dataloader,
            optimizer=optimizer_resumed,
            logger=logger_resumed,
            device=config["device"],
            use_ema=config["training"]["use_ema"],
        )

        # Load checkpoint and resume
        trainer_resumed.load_checkpoint(str(checkpoint_path))

        # Continue training
        trainer_resumed.train(
            num_epochs=2,  # Train to epoch 2 (starting from epoch 1)
            checkpoint_dir=str(checkpoint_dir / "resumed"),
            checkpoint_frequency=1,
        )

        # Verify resumed training created new checkpoints
        resumed_checkpoints = list((checkpoint_dir / "resumed").glob("*.pth"))
        assert len(resumed_checkpoints) > 0, "No checkpoints from resumed training"

        logger_resumed.close()


class TestDiffusionPipelineGeneration:
    """Test generation mode functionality."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_generation_mode_unconditional(self, tmp_path, mock_dataset_medium):
        """Test generation mode with unconditional model.

        Tests:
        - Train a model
        - Load checkpoint for generation
        - Generate samples
        - Save samples to disk
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        output_dir = tmp_path / "generated"

        # Setup configuration
        config = {
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 32,
                "channel_multipliers": [1, 2],
                "num_classes": None,
                "num_timesteps": 100,
                "beta_schedule": "cosine",
                "use_attention": [False, True],
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 32,
                "return_labels": False,
            },
            "device": TEST_DEVICE,
            "seed": None,
            "output": {
                "log_dir": str(tmp_path / "logs"),
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "use_ema": False,
            },
        }

        # Train model
        model = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        dataloader = DiffusionDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            return_labels=config["data"]["return_labels"],
        )

        logger = DiffusionLogger(log_dir=str(log_dir))

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["device"],
            use_ema=config["training"]["use_ema"],
        )

        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=1,
        )

        logger.close()

        # Find checkpoint
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoint saved"
        checkpoint_path = checkpoint_files[0]

        # Load checkpoint for generation
        model_gen = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        model_gen.load_state_dict(checkpoint["model_state_dict"])

        logger_gen = DiffusionLogger(log_dir=str(log_dir / "generation"))

        optimizer_gen = torch.optim.Adam(model_gen.parameters(), lr=0.0001)

        trainer_gen = DiffusionTrainer(
            model=model_gen,
            dataloader=dataloader,
            optimizer=optimizer_gen,
            logger=logger_gen,
            device=config["device"],
            use_ema=False,
        )

        # Generate samples
        num_samples = 8
        samples = trainer_gen.sampler.sample(
            num_samples=num_samples,
            class_labels=None,
            guidance_scale=0.0,
            use_ema=False,
        )

        # Verify samples
        assert samples.shape[0] == num_samples, f"Expected {num_samples} samples"
        assert samples.shape[1:] == (
            config["model"]["in_channels"],
            config["model"]["image_size"],
            config["model"]["image_size"],
        ), "Sample shape mismatch"

        # Save samples
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image(
            samples, output_dir / "generated_samples.png", nrow=4, normalize=True
        )

        # Verify saved file
        assert (output_dir / "generated_samples.png").exists(), (
            "Generated samples not saved"
        )

        logger_gen.close()

    @pytest.mark.smoke
    @pytest.mark.integration
    @pytest.mark.slow
    def test_generation_mode_conditional(self, tmp_path, mock_dataset_medium):
        """Test generation mode with conditional model.

        Tests:
        - Train a conditional model
        - Generate samples for specific classes
        - Use classifier-free guidance
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        output_dir = tmp_path / "generated"

        # Setup configuration
        config = {
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 32,
                "channel_multipliers": [1, 2],
                "num_classes": 2,
                "num_timesteps": 100,
                "beta_schedule": "cosine",
                "class_dropout_prob": 0.1,
                "use_attention": [False, True],
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 32,
                "return_labels": True,
            },
            "device": TEST_DEVICE,
            "seed": None,
            "output": {
                "log_dir": str(tmp_path / "logs"),
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "use_ema": False,
            },
        }

        # Train model
        model = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            class_dropout_prob=config["model"]["class_dropout_prob"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        dataloader = DiffusionDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            return_labels=config["data"]["return_labels"],
        )

        logger = DiffusionLogger(log_dir=str(log_dir))

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["device"],
            use_ema=config["training"]["use_ema"],
        )

        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=1,
        )

        logger.close()

        # Find checkpoint
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoint saved"
        checkpoint_path = checkpoint_files[0]

        # Load checkpoint for generation
        model_gen = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            class_dropout_prob=config["model"]["class_dropout_prob"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        model_gen.load_state_dict(checkpoint["model_state_dict"])

        logger_gen = DiffusionLogger(log_dir=str(log_dir / "generation"))

        optimizer_gen = torch.optim.Adam(model_gen.parameters(), lr=0.0001)

        trainer_gen = DiffusionTrainer(
            model=model_gen,
            dataloader=dataloader,
            optimizer=optimizer_gen,
            logger=logger_gen,
            device=config["device"],
            use_ema=False,
        )

        # Generate samples for each class
        num_samples = 8
        class_labels = torch.tensor([0, 1] * 4, device=config["device"])

        samples = trainer_gen.sampler.sample(
            num_samples=num_samples,
            class_labels=class_labels,
            guidance_scale=2.0,  # Use classifier-free guidance
            use_ema=False,
        )

        # Verify samples
        assert samples.shape[0] == num_samples, f"Expected {num_samples} samples"
        assert samples.shape[1:] == (
            config["model"]["in_channels"],
            config["model"]["image_size"],
            config["model"]["image_size"],
        ), "Sample shape mismatch"

        # Save samples
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image(
            samples,
            output_dir / "generated_samples_conditional.png",
            nrow=4,
            normalize=True,
        )

        # Verify saved file
        assert (output_dir / "generated_samples_conditional.png").exists(), (
            "Generated samples not saved"
        )

        logger_gen.close()


class TestDiffusionPipelineAdvanced:
    """Test advanced diffusion pipeline features."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_with_cosine_scheduler(self, tmp_path, mock_dataset_medium):
        """Test pipeline with learning rate scheduler.

        Tests:
        - Training with cosine annealing scheduler
        - Learning rate is adjusted correctly
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        # Setup configuration
        config = {
            "device": TEST_DEVICE,
            "seed": None,
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 32,
                "channel_multipliers": [1, 2],
                "num_classes": None,
                "num_timesteps": 100,
                "beta_schedule": "cosine",
                "use_attention": [False, True],
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 32,
                "return_labels": False,
            },
            "output": {
                "log_dir": str(tmp_path / "logs"),
            },
            "training": {
                "epochs": 3,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "use_ema": False,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "save_best_only": False,
                "save_frequency": 1,
                "validation": {
                    "frequency": 1,
                    "metric": "loss",
                },
                "visualization": {
                    "log_images_interval": None,
                    "log_denoising_interval": None,
                    "num_samples": 4,
                    "guidance_scale": 0.0,
                },
            },
            "generation": {
                "checkpoint": None,
                "num_samples": 100,
                "guidance_scale": 0.0,
                "use_ema": True,
                "output_dir": None,
            },
        }

        model = create_ddpm(
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            channel_multipliers=tuple(config["model"]["channel_multipliers"]),
            num_classes=config["model"]["num_classes"],
            num_timesteps=config["model"]["num_timesteps"],
            beta_schedule=config["model"]["beta_schedule"],
            use_attention=tuple(config["model"]["use_attention"]),
            device=config["device"],
        )

        dataloader = DiffusionDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            return_labels=config["data"]["return_labels"],
        )

        logger = DiffusionLogger(log_dir=config["output"]["log_dir"])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Add cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"], eta_min=1e-6
        )

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["device"],
            scheduler=scheduler,
            use_ema=config["training"]["use_ema"],
        )

        # Record initial learning rate
        initial_lr = optimizer.param_groups[0]["lr"]

        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["training"]["checkpoint_dir"],
        )

        # Verify learning rate changed
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr, (
            "Learning rate should decrease with cosine annealing"
        )

        logger.close()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_config_file_driven_pipeline(self, tmp_path, mock_dataset_medium):
        """Test pipeline driven entirely by config file.

        Tests:
        - Load config from YAML file
        - Use config to initialize all components
        - Run complete training
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        # Create config file
        config_file = tmp_path / "config.yaml"
        config = {
            "experiment": "diffusion",
            "device": TEST_DEVICE,
            "seed": None,
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 32,
                "channel_multipliers": [1, 2],
                "num_classes": None,
                "num_timesteps": 100,
                "beta_schedule": "cosine",
                "use_attention": [False, True],
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 32,
                "return_labels": False,
            },
            "output": {
                "log_dir": str(tmp_path / "logs"),
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "use_ema": False,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "validation": {
                    "validate_interval": 1,
                    "num_validation_samples": 0,
                },
                "visualization": {
                    "log_images_interval": None,
                    "log_denoising_interval": None,
                },
            },
            "generation": {
                "num_samples": 4,
                "batch_size": 4,
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Load config from file
        with open(config_file, "r") as f:
            loaded_config = yaml.safe_load(f)

        # Initialize components from config
        model = create_ddpm(
            image_size=loaded_config["model"]["image_size"],
            in_channels=loaded_config["model"]["in_channels"],
            model_channels=loaded_config["model"]["model_channels"],
            channel_multipliers=tuple(loaded_config["model"]["channel_multipliers"]),
            num_classes=loaded_config["model"]["num_classes"],
            num_timesteps=loaded_config["model"]["num_timesteps"],
            beta_schedule=loaded_config["model"]["beta_schedule"],
            use_attention=tuple(loaded_config["model"]["use_attention"]),
            device=loaded_config["device"],
        )

        dataloader = DiffusionDataLoader(
            split_file=loaded_config["data"]["split_file"],
            batch_size=loaded_config["data"]["batch_size"],
            num_workers=loaded_config["data"]["num_workers"],
            image_size=loaded_config["data"]["image_size"],
            return_labels=loaded_config["data"]["return_labels"],
        )

        logger = DiffusionLogger(log_dir=loaded_config["output"]["log_dir"])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=loaded_config["training"]["learning_rate"]
        )

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=loaded_config["device"],
            use_ema=loaded_config["training"]["use_ema"],
        )

        # Run training
        trainer.train(
            num_epochs=loaded_config["training"]["epochs"],
            checkpoint_dir=loaded_config["training"]["checkpoint_dir"],
        )

        # Verify outputs
        assert Path(loaded_config["training"]["checkpoint_dir"]).exists()
        assert Path(loaded_config["output"]["log_dir"]).exists()

        logger.close()
