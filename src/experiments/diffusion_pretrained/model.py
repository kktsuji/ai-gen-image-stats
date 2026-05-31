"""Pretrained-transfer diffusion model (ADM / guided-diffusion backbone).

This module adapts OpenAI's class-conditional ImageNet-64 ADM checkpoint for
transfer learning onto small, domain-specific datasets (e.g. 40x40 grayscale
medical images stored as RGB). The heavy lifting -- the U-Net architecture and
the (learned-variance) Gaussian-diffusion process -- comes from the vendored
``guided_diffusion`` package; this wrapper only:

- builds the ADM U-Net with the public ImageNet-64 hyper-parameters,
- loads the public checkpoint (with a local cache, like the classifier slice),
- re-initializes the class-embedding "head" for the target label space
  (``num_classes + 1`` so the last index is the classifier-free-guidance
  unconditional token, matching the project-wide DDPM convention),
- exposes the duck-typed interface the existing ``DiffusionTrainer`` and
  ``sampler`` expect (``compute_loss``, ``forward``, ``sample``,
  ``num_classes``/``image_size``/``in_channels`` attributes), and
- mirrors the classifier transfer pattern (``set_trainable_layers`` /
  ``get_trainable_parameters``) for freezing and staged unfreezing.

The backbone is built at the checkpoint's native ``image_size`` flags (so the
pretrained weight shapes match) but runs at the target resolution (40x40); the
ADM U-Net is fully convolutional with flattened-HW attention, so it is
resolution independent.
"""

import fnmatch
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .guided_diffusion import gaussian_diffusion as gd
from .guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .guided_diffusion.unet import UNetModel

_logger = logging.getLogger(__name__)

# Public class-conditional ImageNet-64 ADM checkpoint and its training flags.
# Source: https://github.com/openai/guided-diffusion (README "MODEL_FLAGS").
ADM_IMAGENET64_CHECKPOINT_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt"
)

# UNetModel kwargs reproducing the public ImageNet-64 model. ``num_classes`` here
# is the checkpoint's label space (1000); it is replaced after loading. Note
# ``attention_resolutions`` is expressed as downsample factors (image_size // res
# for res in {32, 16, 8} at image_size 64 -> {2, 4, 8}). ``use_fp16`` is forced
# off: fine-tuning runs in fp32 (the trainer owns AMP).
IMAGENET64_ADM_FLAGS: Dict[str, Any] = {
    "image_size": 64,
    "in_channels": 3,
    "model_channels": 192,
    "out_channels": 6,  # learn_sigma -> epsilon (3) + variance interpolation (3)
    "num_res_blocks": 3,
    "attention_resolutions": (2, 4, 8),
    "dropout": 0.1,
    "channel_mult": (1, 2, 3, 4),
    "num_classes": 1000,
    "use_checkpoint": False,
    "use_fp16": False,
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "resblock_updown": True,
    "use_new_attention_order": True,
}


def _create_gaussian_diffusion(
    *,
    steps: int,
    learn_sigma: bool,
    noise_schedule: str,
    timestep_respacing: str,
) -> SpacedDiffusion:
    """Build a (learned-variance) Gaussian diffusion process.

    Mirrors ``guided_diffusion.script_util.create_gaussian_diffusion`` for the
    flags used by the ImageNet-64 model (epsilon mean, learned-range variance,
    rescaled hybrid MSE+VLB loss, no timestep input rescaling). ``RESCALED_MSE``
    (not plain ``MSE``) is required for learned-range models so the VLB term is
    scaled by ``num_timesteps / 1000`` and does not swamp the epsilon MSE.
    """
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    respacing: Sequence[int] | str = (
        timestep_respacing if timestep_respacing else [steps]
    )
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=(
            gd.ModelVarType.LEARNED_RANGE
            if learn_sigma
            else gd.ModelVarType.FIXED_LARGE
        ),
        loss_type=gd.LossType.RESCALED_MSE,
        rescale_timesteps=False,
    )


def _resolve_checkpoint(pretrained: Dict[str, Any]) -> Path:
    """Resolve a local checkpoint path, downloading + caching if necessary.

    Mirrors the classifier backbone cache pattern: prefer an explicit local
    ``checkpoint_path``; otherwise use ``cache_path`` and download from
    ``checkpoint_url`` on a cache miss.
    """
    explicit = pretrained.get("checkpoint_path")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint_path '{explicit}' not found")
        return path

    cache_path = Path(pretrained["cache_path"])
    if cache_path.exists():
        _logger.info("Using cached ADM checkpoint at %s", cache_path)
        return cache_path

    # checkpoint_url is required by config validation when no checkpoint_path is
    # given, so it is always present here (no implicit default fallback).
    url = pretrained["checkpoint_url"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _logger.info("Downloading ADM checkpoint from %s -> %s", url, cache_path)
    torch.hub.download_url_to_file(url, str(cache_path))
    return cache_path


class AdmDdpmModel(nn.Module):
    """ADM-backed conditional diffusion model with a re-initialized class head.

    Args:
        image_size: Target (and runtime) square image size, e.g. 40.
        num_classes: Number of semantic classes. The model is built with
            ``num_classes + 1`` label embeddings; the extra index is the
            classifier-free-guidance unconditional token.
        class_dropout_prob: Probability of replacing a label with the
            unconditional token during training (enables CFG at sampling).
        arch: Optional overrides for the U-Net flags (used by tests to build a
            tiny model). Defaults to the ImageNet-64 ADM flags.
        num_timesteps: Diffusion steps used for training (1000 for ADM).
        noise_schedule: Beta schedule name ("cosine" for ADM ImageNet-64).
        sample_timestep_respacing: Respacing spec for sampling (e.g. "250" for
            250-step sampling, "" for the full chain).
        pretrained: Optional dict ``{checkpoint_path | checkpoint_url,
            cache_path}``. When ``None`` the backbone is randomly initialized
            (from-scratch / tests).
        device: Device string.
    """

    def __init__(
        self,
        *,
        image_size: int = 40,
        num_classes: int = 2,
        class_dropout_prob: float = 0.1,
        arch: Optional[Dict[str, Any]] = None,
        num_timesteps: int = 1000,
        noise_schedule: str = "cosine",
        sample_timestep_respacing: str = "250",
        pretrained: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.image_size = image_size
        # Semantic classes (0..num_classes-1); the unconditional token is the
        # index ``num_classes`` (matches the project-wide DDPM convention).
        self.num_classes = num_classes
        self.label_dim = num_classes + 1
        self.uncond_index = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.num_timesteps = num_timesteps

        flags: Dict[str, Any] = dict(IMAGENET64_ADM_FLAGS)
        if arch:
            flags.update(arch)
        # Tie the channel count to the actually-built U-Net (single source of
        # truth) rather than hardcoding 3.
        self.in_channels = flags["in_channels"]
        self.time_embed_dim = flags["model_channels"] * 4

        # Build with the checkpoint's label space so weights load cleanly; when
        # not loading pretrained weights, build directly with our label space.
        build_flags = {k: v for k, v in flags.items() if k != "num_classes"}
        build_num_classes = flags["num_classes"] if pretrained else self.label_dim
        unet = UNetModel(num_classes=build_num_classes, **build_flags)

        if pretrained:
            self._load_pretrained_backbone(unet, pretrained)
            # Replace the class-embedding "head" for the target label space.
            unet.label_emb = nn.Embedding(self.label_dim, self.time_embed_dim)
            unet.num_classes = self.label_dim

        self.unet = unet

        self.train_diffusion = _create_gaussian_diffusion(
            steps=num_timesteps,
            learn_sigma=True,
            noise_schedule=noise_schedule,
            timestep_respacing="",
        )
        self.sample_diffusion = _create_gaussian_diffusion(
            steps=num_timesteps,
            learn_sigma=True,
            noise_schedule=noise_schedule,
            timestep_respacing=sample_timestep_respacing,
        )

        self.to(device)

    # ----- pretrained loading -------------------------------------------------

    @staticmethod
    def _load_pretrained_backbone(unet: UNetModel, pretrained: Dict[str, Any]) -> None:
        path = _resolve_checkpoint(pretrained)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = unet.load_state_dict(state_dict, strict=False)
        # The ADM checkpoint is a plain state dict that matches the architecture
        # exactly. Guard against a wrong/wrapped checkpoint (e.g. a nested
        # {"state_dict": ...} or mismatched flags) silently leaving the backbone
        # at random init: almost every parameter must be filled.
        total = len(unet.state_dict())
        loaded = total - len(missing)
        if loaded < 0.5 * total:
            raise RuntimeError(
                f"ADM checkpoint at {path} matched only {loaded}/{total} model "
                f"parameters ({len(missing)} missing, {len(unexpected)} unexpected); "
                f"it does not match the ADM ImageNet-64 architecture. Refusing to "
                f"fine-tune a randomly-initialized backbone."
            )
        if missing or unexpected:
            _logger.warning(
                "ADM checkpoint load: %d missing, %d unexpected keys",
                len(missing),
                len(unexpected),
            )

    # ----- training -----------------------------------------------------------

    def _labels_for_training(
        self,
        class_labels: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if class_labels is None:
            return torch.full(
                (batch_size,), self.uncond_index, device=device, dtype=torch.long
            )
        labels = class_labels.to(device).clone()
        if self.training and self.class_dropout_prob > 0.0:
            drop = torch.rand(batch_size, device=device) < self.class_dropout_prob
            labels[drop] = self.uncond_index
        return labels

    def compute_loss(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        criterion: Optional[nn.Module] = None,  # accepted for trainer parity; unused
    ) -> torch.Tensor:
        """Hybrid (MSE + VLB) diffusion loss via the ADM diffusion process."""
        del criterion  # ADM owns its loss (learned-variance hybrid objective)
        batch_size = x.shape[0]
        device = x.device
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        labels = self._labels_for_training(class_labels, batch_size, device)
        terms = self.train_diffusion.training_losses(
            self.unet, x, t, model_kwargs={"y": labels}
        )
        return terms["loss"].mean()

    def forward(
        self, x: torch.Tensor, class_labels: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(predicted_epsilon, target_noise)`` for class-weighted losses.

        Provided for drop-in parity with the custom DDPM (the trainer uses this
        path only when class-weighting is enabled). Note this exposes only the
        epsilon prediction; the recommended balancing for this slice is
        data-level sampling rather than a weighted epsilon MSE.
        """
        batch_size = x.shape[0]
        device = x.device
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x)
        x_t = self.train_diffusion.q_sample(x, t, noise=noise)
        labels = self._labels_for_training(class_labels, batch_size, device)
        model_out = self.unet(x_t, t, y=labels)
        predicted_noise = model_out[:, : self.in_channels]
        return predicted_noise, noise

    # ----- sampling -----------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        return_intermediates: bool = False,
        show_progress: bool = False,
        progress_desc: str = "Denoising",
        **_kwargs: Any,
    ) -> torch.Tensor:
        """Generate samples in [-1, 1] via the (respaced) ADM diffusion process.

        With ``guidance_scale > 0`` classifier-free guidance is applied by
        combining conditional and unconditional epsilon predictions
        (``uncond + scale * (cond - uncond)``), matching the project convention.

        Note: ``return_intermediates=True`` keeps every denoising frame
        (one per respaced timestep) in memory before stacking, so peak memory
        scales with the number of sampling steps × batch size. Prefer a small
        batch / coarse ``sample_timestep_respacing`` when collecting frames for
        the full chain.
        """
        del progress_desc  # ADM progress bar is a simple flag
        device = next(self.parameters()).device
        shape = (batch_size, self.in_channels, self.image_size, self.image_size)

        if class_labels is not None:
            y_cond = class_labels.to(device).long()
        else:
            y_cond = torch.full(
                (batch_size,), self.uncond_index, device=device, dtype=torch.long
            )

        use_cfg = guidance_scale > 0.0
        if use_cfg:
            y_uncond = torch.full(
                (batch_size,), self.uncond_index, device=device, dtype=torch.long
            )

            def model_fn(x: torch.Tensor, ts: torch.Tensor, **__: Any) -> torch.Tensor:
                # Batch the conditional and unconditional predictions into a
                # single forward over a doubled batch (one U-Net call per step
                # instead of two).
                combined_x = torch.cat([x, x], dim=0)
                combined_ts = torch.cat([ts, ts], dim=0)
                combined_y = torch.cat([y_cond, y_uncond], dim=0)
                combined = self.unet(combined_x, combined_ts, y=combined_y)
                cond, uncond = torch.chunk(combined, 2, dim=0)
                cond_eps, cond_var = torch.split(cond, self.in_channels, dim=1)
                uncond_eps, _ = torch.split(uncond, self.in_channels, dim=1)
                eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                return torch.cat([eps, cond_var], dim=1)

            model_kwargs: Dict[str, Any] = {}
            callable_model: Any = model_fn
        else:
            model_kwargs = {"y": y_cond}
            callable_model = self.unet

        if return_intermediates:
            frames: List[torch.Tensor] = []
            for out in self.sample_diffusion.p_sample_loop_progressive(
                callable_model,
                shape,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                progress=show_progress,
            ):
                frames.append(out["sample"])
            return torch.stack(frames, dim=0)

        return self.sample_diffusion.p_sample_loop(
            callable_model,
            shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            device=device,
            progress=show_progress,
        )

    # ----- transfer-learning controls (mirrors the classifier slice) ----------

    def set_trainable_layers(self, layer_patterns: Sequence[str]) -> None:
        """Freeze the whole backbone, then unfreeze params matching any pattern.

        Patterns are ``fnmatch`` globs over ``unet`` parameter names, e.g.
        ``"label_emb*"``, ``"out.*"``, ``"output_blocks.*"``,
        ``"middle_block.*"``, ``"input_blocks.*"``, ``"time_embed.*"``.
        """
        for param in self.unet.parameters():
            param.requires_grad = False

        matched = 0
        for name, param in self.unet.named_parameters():
            if any(fnmatch.fnmatch(name, pattern) for pattern in layer_patterns):
                param.requires_grad = True
                matched += 1

        if matched == 0:
            raise ValueError(
                "set_trainable_layers matched no parameters for patterns: "
                f"{list(layer_patterns)}. Check the fnmatch patterns against the "
                "unet parameter names (e.g. 'label_emb*', 'out.*', "
                "'output_blocks.*', 'middle_block.*', 'input_blocks.*', "
                "'time_embed.*')."
            )
        _logger.info(
            "Trainable parameters: %d tensors match %s",
            matched,
            list(layer_patterns),
        )

    def freeze_backbone(self) -> None:
        """Freeze everything except the re-initialized class head and output."""
        self.set_trainable_layers(["label_emb*", "out.*"])

    def get_trainable_parameters(self):
        """Return an iterator over parameters with ``requires_grad=True``."""
        return (param for param in self.parameters() if param.requires_grad)


def create_adm_ddpm(
    *,
    image_size: int = 40,
    num_classes: int = 2,
    class_dropout_prob: float = 0.1,
    num_timesteps: int = 1000,
    noise_schedule: str = "cosine",
    sample_timestep_respacing: str = "250",
    pretrained: Optional[Dict[str, Any]] = None,
    arch: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> AdmDdpmModel:
    """Factory for :class:`AdmDdpmModel` (parallels ``create_ddpm``)."""
    return AdmDdpmModel(
        image_size=image_size,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        num_timesteps=num_timesteps,
        noise_schedule=noise_schedule,
        sample_timestep_respacing=sample_timestep_respacing,
        pretrained=pretrained,
        arch=arch,
        device=device,
    )


def build_adm_model(
    config: Dict[str, Any], device: str, load_pretrained: bool = True
) -> AdmDdpmModel:
    """Build an :class:`AdmDdpmModel` from a validated config and apply the
    freeze/unfreeze policy.

    Args:
        config: Full (validated) experiment config.
        device: Device string.
        load_pretrained: If True, download/load the public ADM checkpoint as
            initialization (train mode). If False (generate mode / resume),
            build the architecture only -- weights are restored from a
            fine-tuned checkpoint by the caller. The freeze policy is applied in
            both cases so that ``requires_grad`` flags (and therefore the EMA
            shadow keys) match between training and generation.
    """
    model_cfg = config["model"]
    arch = model_cfg["architecture"]
    diff = model_cfg["diffusion"]
    cond = model_cfg["conditioning"]
    init = model_cfg["initialization"]

    model = create_adm_ddpm(
        image_size=arch["image_size"],
        num_classes=cond["num_classes"],
        class_dropout_prob=cond["class_dropout_prob"],
        num_timesteps=diff["num_timesteps"],
        noise_schedule=diff["noise_schedule"],
        sample_timestep_respacing=diff["sample_timestep_respacing"],
        pretrained=model_cfg["pretrained"] if load_pretrained else None,
        device=device,
    )

    trainable_layers = init.get("trainable_layers")
    if trainable_layers:
        model.set_trainable_layers(trainable_layers)
    elif init.get("freeze_backbone", True):
        model.freeze_backbone()
    # Otherwise (freeze_backbone=False, no patterns): full fine-tuning.

    return model
