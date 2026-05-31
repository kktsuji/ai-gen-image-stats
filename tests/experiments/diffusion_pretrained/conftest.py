"""Shared fixtures/helpers for diffusion_pretrained tests."""

import copy
import pathlib
from typing import Any, Dict

import pytest
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_EXAMPLE_CONFIG = _REPO_ROOT / "configs" / "examples" / "diffusion-pretrained.yaml"

# Tiny ADM U-Net flags so tests build a fast CPU model instead of the 296M
# ImageNet-64 backbone. model_channels must stay a multiple of 32 (GroupNorm32).
TINY_ADM_ARCH: Dict[str, Any] = {
    "model_channels": 32,
    "num_res_blocks": 1,
    "channel_mult": (1, 2),
    "attention_resolutions": (2,),
    "num_head_channels": 8,
    "num_heads": 2,
}


@pytest.fixture
def example_config() -> Dict[str, Any]:
    """Load a fresh copy of the example pretrained-transfer config."""
    with open(_EXAMPLE_CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture
def tiny_adm_model():
    """A tiny ADM-backed model (no pretrained download) for fast CPU tests."""
    from src.experiments.diffusion_pretrained.model import create_adm_ddpm

    return create_adm_ddpm(
        image_size=16,
        num_classes=2,
        class_dropout_prob=0.1,
        num_timesteps=10,
        noise_schedule="cosine",
        sample_timestep_respacing="5",
        pretrained=None,
        arch=copy.deepcopy(TINY_ADM_ARCH),
        device="cpu",
    )
