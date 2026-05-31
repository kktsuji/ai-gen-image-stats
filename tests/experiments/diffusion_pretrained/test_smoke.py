"""Smoke test: load the real public ADM ImageNet-64 checkpoint and sample.

Downloads ~300MB and instantiates a ~296M-param model, so it is gated behind
the ``smoke`` marker (manual/weekly), not run in unit/component/CI tiers.
"""

import pytest
import torch

from src.experiments.diffusion_pretrained.model import create_adm_ddpm


@pytest.mark.smoke
def test_load_real_adm_checkpoint_and_sample(tmp_path):
    pretrained = {
        "source": "adm_imagenet64",
        "checkpoint_url": (
            "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
            "64x64_diffusion.pt"
        ),
        "cache_path": str(tmp_path / "64x64_diffusion.pt"),
    }
    model = create_adm_ddpm(
        image_size=40,
        num_classes=2,
        num_timesteps=1000,
        noise_schedule="cosine",
        sample_timestep_respacing="10",  # few steps to keep the smoke test fast
        pretrained=pretrained,
        device="cpu",
    )
    model.freeze_backbone()
    model.eval()
    with torch.no_grad():
        samples = model.sample(
            batch_size=1, class_labels=torch.tensor([1]), guidance_scale=2.0
        )
    assert samples.shape == (1, 3, 40, 40)
    assert torch.isfinite(samples).all()
