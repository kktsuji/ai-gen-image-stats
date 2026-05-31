"""Tests for the AdmDdpmModel wrapper and build_adm_model."""

import copy
from pathlib import Path

import pytest
import torch

from src.experiments.diffusion_pretrained import model as adm_model
from src.experiments.diffusion_pretrained.model import build_adm_model, create_adm_ddpm
from src.utils.checkpoint import load_checkpoint, save_checkpoint

from .conftest import TINY_ADM_ARCH

# ----- unit: architecture / class head ---------------------------------------


@pytest.mark.unit
def test_label_embedding_sized_for_cfg(tiny_adm_model):
    # num_classes semantic classes + 1 unconditional (CFG) token.
    assert tiny_adm_model.num_classes == 2
    assert tiny_adm_model.label_dim == 3
    assert tiny_adm_model.uncond_index == 2
    assert tuple(tiny_adm_model.unet.label_emb.weight.shape)[0] == 3


@pytest.mark.unit
def test_freeze_backbone_only_head_and_output_trainable(tiny_adm_model):
    tiny_adm_model.freeze_backbone()
    trainable_names = {
        name for name, p in tiny_adm_model.unet.named_parameters() if p.requires_grad
    }
    assert trainable_names, "expected some trainable params"
    # Every trainable tensor must belong to the class head or the output block.
    assert all(
        name.startswith("label_emb") or name.startswith("out.")
        for name in trainable_names
    )


@pytest.mark.unit
def test_set_trainable_layers_staged_unfreeze(tiny_adm_model):
    tiny_adm_model.freeze_backbone()
    frozen_count = sum(1 for _ in tiny_adm_model.get_trainable_parameters())
    tiny_adm_model.set_trainable_layers(["label_emb*", "out.*", "output_blocks.*"])
    staged_count = sum(1 for _ in tiny_adm_model.get_trainable_parameters())
    assert staged_count > frozen_count


@pytest.mark.unit
def test_get_trainable_parameters_matches_requires_grad(tiny_adm_model):
    tiny_adm_model.set_trainable_layers(["label_emb*"])
    expected = [p for p in tiny_adm_model.parameters() if p.requires_grad]
    got = list(tiny_adm_model.get_trainable_parameters())
    assert len(got) == len(expected)


# ----- component: forward / loss / sampling ----------------------------------


@pytest.mark.component
def test_forward_returns_eps_and_noise(tiny_adm_model):
    x = torch.randn(2, 3, 16, 16)
    y = torch.tensor([0, 1])
    eps, noise = tiny_adm_model(x, class_labels=y)
    assert eps.shape == (2, 3, 16, 16)
    assert noise.shape == (2, 3, 16, 16)


@pytest.mark.component
def test_compute_loss_is_scalar_and_backprops(tiny_adm_model):
    tiny_adm_model.freeze_backbone()
    tiny_adm_model.train()
    x = torch.randn(2, 3, 16, 16)
    y = torch.tensor([0, 1])
    loss = tiny_adm_model.compute_loss(x, class_labels=y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    # Gradients only on trainable (head/output) params.
    grads = [
        p.grad for p in tiny_adm_model.get_trainable_parameters() if p.grad is not None
    ]
    assert grads, "expected gradients on trainable params"


@pytest.mark.component
def test_sample_shapes_and_range(tiny_adm_model):
    tiny_adm_model.eval()
    y = torch.tensor([0, 1])
    samples = tiny_adm_model.sample(batch_size=2, class_labels=y, guidance_scale=0.0)
    assert samples.shape == (2, 3, 16, 16)
    assert samples.min() >= -1.0 - 1e-4 and samples.max() <= 1.0 + 1e-4


@pytest.mark.component
def test_sample_with_cfg(tiny_adm_model):
    tiny_adm_model.eval()
    y = torch.tensor([0, 1])
    samples = tiny_adm_model.sample(batch_size=2, class_labels=y, guidance_scale=3.0)
    assert samples.shape == (2, 3, 16, 16)


@pytest.mark.component
def test_sample_return_intermediates(tiny_adm_model):
    tiny_adm_model.eval()
    y = torch.tensor([0, 1])
    seq = tiny_adm_model.sample(
        batch_size=2, class_labels=y, guidance_scale=3.0, return_intermediates=True
    )
    # (T, N, C, H, W); final frame is the sample. T == respaced steps (5).
    assert seq.ndim == 5
    assert seq.shape[1:] == (2, 3, 16, 16)


# ----- component: build_adm_model freeze policy ------------------------------


@pytest.mark.component
def test_build_adm_model_freeze_backbone(example_config, monkeypatch):
    monkeypatch.setattr(
        adm_model,
        "IMAGENET64_ADM_FLAGS",
        {**adm_model.IMAGENET64_ADM_FLAGS, **TINY_ADM_ARCH},
    )
    model = build_adm_model(example_config, "cpu", load_pretrained=False)
    trainable = {name for name, p in model.unet.named_parameters() if p.requires_grad}
    assert all(
        name.startswith("label_emb") or name.startswith("out.") for name in trainable
    )


@pytest.mark.component
def test_build_adm_model_full_finetune(example_config, monkeypatch):
    monkeypatch.setattr(
        adm_model,
        "IMAGENET64_ADM_FLAGS",
        {**adm_model.IMAGENET64_ADM_FLAGS, **TINY_ADM_ARCH},
    )
    cfg = copy.deepcopy(example_config)
    cfg["model"]["initialization"]["freeze_backbone"] = False
    cfg["model"]["initialization"]["trainable_layers"] = None
    model = build_adm_model(cfg, "cpu", load_pretrained=False)
    assert all(p.requires_grad for p in model.parameters())


# ----- component: checkpoint roundtrip ---------------------------------------


@pytest.mark.component
def test_checkpoint_save_load_roundtrip(tmp_path: Path):
    arch = copy.deepcopy(TINY_ADM_ARCH)
    m1 = create_adm_ddpm(
        image_size=16,
        num_classes=2,
        num_timesteps=10,
        sample_timestep_respacing="5",
        arch=arch,
        device="cpu",
    )
    ckpt = tmp_path / "model.pth"
    save_checkpoint(
        path=ckpt,
        model=m1,
        optimizer=None,  # type: ignore[arg-type]
        epoch=1,
        global_step=10,
        save_optimizer=False,
    )

    m2 = create_adm_ddpm(
        image_size=16,
        num_classes=2,
        num_timesteps=10,
        sample_timestep_respacing="5",
        arch=copy.deepcopy(TINY_ADM_ARCH),
        device="cpu",
    )
    load_checkpoint(path=ckpt, model=m2, optimizer=None, strict=True)

    ref = dict(m1.named_parameters())
    for name, p in m2.named_parameters():
        assert torch.allclose(p, ref[name]), f"param mismatch: {name}"
