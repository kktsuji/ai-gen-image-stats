"""Integration of the ADM wrapper with the shared DiffusionTrainer.

Confirms the duck-typed contract (compute_loss / sample / EMA over trainable
params) holds so the existing trainer drives ADM transfer training unchanged.
"""

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.diffusion.trainer import DiffusionTrainer
from src.experiments.diffusion_pretrained.model import create_adm_ddpm

from .conftest import TINY_ADM_ARCH


class _CaptureLogger:
    def __init__(self):
        self.logged_metrics = []

    def log_metrics(self, metrics, step, epoch=None):
        self.logged_metrics.append(metrics)

    def log_images(self, *args, **kwargs):
        pass

    def log_hyperparams(self, hyperparams):
        pass


def _tiny_adm():
    return create_adm_ddpm(
        image_size=16,
        num_classes=2,
        class_dropout_prob=0.1,
        num_timesteps=10,
        sample_timestep_respacing="5",
        arch=copy.deepcopy(TINY_ADM_ARCH),
        device="cpu",
    )


def _loader(num_samples=8, batch_size=4):
    images = torch.randn(num_samples, 3, 16, 16)
    labels = torch.randint(0, 2, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


@pytest.mark.component
def test_train_epoch_runs_and_reports_loss():
    model = _tiny_adm()
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
    trainer = DiffusionTrainer(
        model=model,
        train_loader=_loader(),
        optimizer=optimizer,
        logger=_CaptureLogger(),
        device="cpu",
        show_progress=False,
        use_ema=True,
        use_amp=False,
    )
    metrics = trainer.train_epoch()
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))


@pytest.mark.component
def test_ema_tracks_only_trainable_params():
    model = _tiny_adm()
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
    trainer = DiffusionTrainer(
        model=model,
        train_loader=_loader(),
        optimizer=optimizer,
        logger=_CaptureLogger(),
        device="cpu",
        show_progress=False,
        use_ema=True,
        use_amp=False,
    )
    # EMA shadow must match the trainable parameter set exactly (so apply_shadow
    # in generation mode does not hit a missing-key assertion).
    assert trainer.ema is not None
    assert set(trainer.ema.shadow.keys()) == {
        n for n, p in model.named_parameters() if p.requires_grad
    }
    assert len(trainer.ema.shadow) < sum(1 for _ in model.parameters())
