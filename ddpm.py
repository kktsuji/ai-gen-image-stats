"""DDPM (Denoising Diffusion Probabilistic Models) Implementation

A simple DDPM implementation for 40x40x3 images without text encoder, VAE, or latent space.
Uses U-Net architecture for the denoising network.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    """Exponential Moving Average of model parameters.

    Keeps a moving average of model weights during training to improve sampling quality.
    The EMA weights are typically more stable and produce cleaner outputs with fewer artifacts.

    Args:
        model: The model to track with EMA
        decay: Decay rate for the moving average (default: 0.9999)
        device: Device to store EMA weights on
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = "cpu"):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)

    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to the model (for inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original parameters (after inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].clone()
        self.backup = {}

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding and optional class conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Add class conditioning if num_classes is provided
        self.class_mlp = (
            nn.Linear(time_emb_dim, out_channels) if num_classes is not None else None
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.dropout = nn.Dropout(dropout)

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        class_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time embedding
        emb = self.time_mlp(F.silu(time_emb))

        # Add class embedding if provided
        if class_emb is not None and self.class_mlp is not None:
            emb = emb + self.class_mlp(F.silu(class_emb))

        h = h + emb[:, :, None, None]
        h = F.silu(h)

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)

        # Residual connection
        return F.silu(h + self.residual_conv(x))


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        # Normalize and compute Q, K, V
        h = self.norm(x)
        qkv = self.qkv(h)

        # Reshape for multi-head attention
        qkv = qkv.reshape(
            batch, 3, self.num_heads, channels // self.num_heads, height * width
        )
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, heads, hw, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        h = torch.matmul(attn, v)

        # Reshape back
        h = h.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        h = self.proj(h)

        # Residual
        return x + h


class DownBlock(nn.Module):
    """Downsampling block with residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_classes: Optional[int] = None,
        num_layers: int = 2,
        downsample: bool = True,
        use_attention: bool = False,
    ):
        super().__init__()
        self.downsample = downsample

        self.resblocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    time_emb_dim,
                    num_classes,
                )
                for i in range(num_layers)
            ]
        )

        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = None

        if downsample:
            self.downsample_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.downsample_conv = None

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        class_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for resblock in self.resblocks:
            h = resblock(h, time_emb, class_emb)

        if self.attention is not None:
            h = self.attention(h)

        if self.downsample_conv is not None:
            skip = h
            h = self.downsample_conv(h)
            return h, skip
        else:
            return h, h


class UpBlock(nn.Module):
    """Upsampling block with residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_classes: Optional[int] = None,
        num_layers: int = 2,
        upsample: bool = True,
        use_attention: bool = False,
    ):
        super().__init__()
        self.upsample = upsample

        self.resblocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    time_emb_dim,
                    num_classes,
                )
                for i in range(num_layers)
            ]
        )

        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = None

        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            self.upsample_conv = None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
        class_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Concatenate skip connection
        h = torch.cat([x, skip], dim=1)

        for resblock in self.resblocks:
            h = resblock(h, time_emb, class_emb)

        if self.attention is not None:
            h = self.attention(h)

        if self.upsample_conv is not None:
            h = self.upsample_conv(h)

        return h


class UNet(nn.Module):
    """U-Net architecture for DDPM with optional class conditioning."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
        use_attention: Tuple[bool, ...] = (False, False, True),
    ):
        super().__init__()

        self.num_classes = num_classes

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Class embedding (for classifier-free guidance)
        if num_classes is not None:
            self.class_emb = nn.Embedding(
                num_classes + 1, time_emb_dim
            )  # +1 for unconditional class
        else:
            self.class_emb = None

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([])
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            self.downs.append(
                DownBlock(
                    now_channels,
                    out_ch,
                    time_emb_dim,
                    num_classes,
                    num_layers=num_res_blocks,
                    downsample=(i != len(channel_multipliers) - 1),
                    use_attention=use_attention[i],
                )
            )

            now_channels = out_ch
            channels.append(now_channels)

        # Middle
        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels, now_channels, time_emb_dim, num_classes, dropout
                ),
                AttentionBlock(now_channels),
                ResidualBlock(
                    now_channels, now_channels, time_emb_dim, num_classes, dropout
                ),
            ]
        )

        # Upsampling
        self.ups = nn.ModuleList([])

        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult

            self.ups.append(
                UpBlock(
                    now_channels + channels.pop(),
                    out_ch,
                    time_emb_dim,
                    num_classes,
                    num_layers=num_res_blocks,
                    upsample=(i != len(channel_multipliers) - 1),
                    use_attention=use_attention[len(channel_multipliers) - 1 - i],
                )
            )

            now_channels = out_ch

        # Final convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_mlp(timestep)

        # Class embedding
        class_emb = None
        if self.class_emb is not None and class_labels is not None:
            class_emb = self.class_emb(class_labels)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling with skip connections
        skips = []
        for down in self.downs:
            h, skip = down(h, time_emb, class_emb)
            skips.append(skip)

        # Middle
        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb, class_emb)
            else:
                h = layer(h)

        # Upsampling
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, time_emb, class_emb)

        # Final conv
        return self.conv_out(h)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model with Classifier-Free Guidance.

    Args:
        image_size: Size of the input images (assumes square images)
        in_channels: Number of input channels
        model_channels: Base number of channels in the U-Net
        num_classes: Number of classes for conditional generation (None for unconditional)
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Type of noise schedule ('linear', 'cosine', 'quadratic', 'sigmoid')
        beta_start: Starting beta value for noise schedule (used for linear, quadratic, sigmoid)
        beta_end: Ending beta value for noise schedule (used for linear, quadratic, sigmoid)
        class_dropout_prob: Probability of dropping class labels during training (for classifier-free guidance)
    """

    def __init__(
        self,
        image_size: int = 40,
        in_channels: int = 3,
        model_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        num_classes: Optional[int] = None,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        class_dropout_prob: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.beta_schedule = beta_schedule
        self.device = device

        # U-Net model
        self.model = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=model_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            num_classes=num_classes,
        )

        # EMA for improved sampling quality
        self.ema = None  # Will be initialized externally if needed

        # Precompute noise schedule
        self.register_buffer(
            "betas",
            self._get_beta_schedule(beta_schedule, num_timesteps, beta_start, beta_end),
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )

        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1)
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod),
        )

    def _get_beta_schedule(
        self, schedule: str, timesteps: int, beta_start: float, beta_end: float
    ) -> torch.Tensor:
        """Get beta schedule based on the specified type.

        Args:
            schedule: Type of schedule ('linear', 'cosine', 'quadratic', 'sigmoid')
            timesteps: Number of timesteps
            beta_start: Starting beta value (for linear, quadratic, sigmoid)
            beta_end: Ending beta value (for linear, quadratic, sigmoid)

        Returns:
            Beta schedule tensor
        """
        if schedule == "linear":
            return self._linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            return self._cosine_beta_schedule(timesteps)
        elif schedule == "quadratic":
            return self._quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "sigmoid":
            return self._sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(
                f"Unknown beta schedule: {schedule}. "
                f"Choose from: 'linear', 'cosine', 'quadratic', 'sigmoid'"
            )

    def _linear_beta_schedule(
        self, timesteps: int, beta_start: float, beta_end: float
    ) -> torch.Tensor:
        """Linear beta schedule."""
        return torch.linspace(beta_start, beta_end, timesteps)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672

        Args:
            timesteps: Number of timesteps
            s: Small offset to prevent beta from being too small near t=0

        Returns:
            Beta schedule tensor
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _quadratic_beta_schedule(
        self, timesteps: int, beta_start: float, beta_end: float
    ) -> torch.Tensor:
        """Quadratic beta schedule for smoother transitions.

        Args:
            timesteps: Number of timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value

        Returns:
            Beta schedule tensor
        """
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    def _sigmoid_beta_schedule(
        self, timesteps: int, beta_start: float, beta_end: float
    ) -> torch.Tensor:
        """Sigmoid beta schedule for smoother transitions.

        Args:
            timesteps: Number of timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value

        Returns:
            Beta schedule tensor
        """
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def _extract(
        self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple
    ) -> torch.Tensor:
        """Extract coefficients at specified timesteps and reshape to match x_shape."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def dynamic_threshold(
        self, x_start: torch.Tensor, percentile: float = 0.995
    ) -> torch.Tensor:
        """Apply dynamic thresholding to reduce extreme pixel values.

        This technique from Imagen helps suppress artifacts by clipping values
        based on a percentile threshold rather than fixed values.

        Args:
            x_start: Predicted x_0
            percentile: Percentile for dynamic threshold (default: 0.995, i.e., 99.5%)

        Returns:
            Thresholded x_start
        """
        # Compute absolute values
        x_flat = x_start.flatten(start_dim=1)
        abs_flat = torch.abs(x_flat)

        # Compute percentile threshold per sample in batch
        s = torch.quantile(abs_flat, percentile, dim=1, keepdim=True)

        # Ensure threshold is at least 1.0
        s = torch.clamp(s, min=1.0)

        # Clip and rescale
        # For older PyTorch versions, use torch.max/min instead of clamp with tensor args
        s = s.view(-1, 1, 1, 1)  # Reshape for broadcasting
        x_start = torch.max(torch.min(x_start, s), -s) / s

        return x_start

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance of diffusion posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)

        return posterior_mean, posterior_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        use_dynamic_threshold: bool = True,
        dynamic_threshold_percentile: float = 0.995,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute predicted mean and variance for reverse process with classifier-free guidance.

        Args:
            x_t: Noisy input
            t: Timestep
            class_labels: Class labels for conditional generation
            guidance_scale: Classifier-free guidance scale (0.0 = no guidance, higher = stronger conditioning)
            use_dynamic_threshold: Whether to apply dynamic thresholding (default: True)
            dynamic_threshold_percentile: Percentile for dynamic thresholding (default: 0.995)
        """
        if (
            self.num_classes is not None
            and guidance_scale > 0.0
            and class_labels is not None
        ):
            # Classifier-free guidance: interpolate between conditional and unconditional predictions
            # Conditional prediction
            predicted_noise_cond = self.model(x_t, t, class_labels)

            # Unconditional prediction (use class index num_classes as the unconditional token)
            batch_size = x_t.shape[0]
            unconditional_labels = torch.full(
                (batch_size,), self.num_classes, device=x_t.device, dtype=torch.long
            )
            predicted_noise_uncond = self.model(x_t, t, unconditional_labels)

            # Apply guidance: noise = uncond + guidance_scale * (cond - uncond)
            predicted_noise = predicted_noise_uncond + guidance_scale * (
                predicted_noise_cond - predicted_noise_uncond
            )
        else:
            # Standard prediction without guidance
            predicted_noise = self.model(x_t, t, class_labels)

        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        x_start = torch.clamp(x_start, -1.0, 1.0)

        # Apply dynamic thresholding if enabled
        if use_dynamic_threshold:
            x_start = self.dynamic_threshold(x_start, dynamic_threshold_percentile)

        # Compute posterior
        model_mean, model_variance = self.q_posterior(x_start, x_t, t)

        return model_mean, model_variance

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        use_dynamic_threshold: bool = True,
        dynamic_threshold_percentile: float = 0.995,
    ) -> torch.Tensor:
        """Sample from reverse process: p(x_{t-1} | x_t) with classifier-free guidance."""
        model_mean, model_variance = self.p_mean_variance(
            x_t,
            t,
            class_labels,
            guidance_scale,
            use_dynamic_threshold,
            dynamic_threshold_percentile,
        )

        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        return_intermediates: bool = False,
        use_dynamic_threshold: bool = True,
        dynamic_threshold_percentile: float = 0.995,
    ) -> torch.Tensor:
        """Generate samples from the model with optional classifier-free guidance.

        Args:
            batch_size: Number of samples to generate
            class_labels: Class labels for conditional generation (shape: [batch_size])
            guidance_scale: Classifier-free guidance scale (0.0 = no guidance, 3.0-7.0 typical)
            return_intermediates: If True, return all intermediate timesteps
            use_dynamic_threshold: Whether to apply dynamic thresholding (default: True)
            dynamic_threshold_percentile: Percentile for dynamic thresholding (default: 0.995)

        Returns:
            Generated samples, shape (batch_size, in_channels, image_size, image_size)
            If return_intermediates is True, returns (num_timesteps, batch_size, in_channels, image_size, image_size)
        """
        shape = (batch_size, self.in_channels, self.image_size, self.image_size)
        device = next(self.parameters()).device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        intermediates = [x] if return_intermediates else None

        # Iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            if (i + 1) % 10 == 0:
                print(
                    f"  - Sampling timestep {i+1:04d}/{self.num_timesteps:04d}",
                    end="\r",
                )
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(
                x,
                t,
                class_labels,
                guidance_scale,
                use_dynamic_threshold,
                dynamic_threshold_percentile,
            )

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return torch.stack(intermediates)

        return x

    @torch.no_grad()
    def sample_from_image(
        self,
        x_0: torch.Tensor,
        t_0: int = 300,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        return_intermediates: bool = False,
        use_dynamic_threshold: bool = True,
        dynamic_threshold_percentile: float = 0.995,
    ) -> torch.Tensor:
        """SDEdit: Generate samples by starting from a real image (image-to-image diffusion).

        This is particularly useful for generating abnormalities on real normal backgrounds,
        which helps preserve real image statistics and reduces domain gap.

        Args:
            x_0: Starting images (e.g., real Normal images), shape (batch_size, in_channels, image_size, image_size)
                 Should be normalized to [-1, 1] range.
            t_0: Starting timestep for denoising (e.g., 200-400). Higher = more change, lower = more preservation.
            class_labels: Target class labels for conditional generation (shape: [batch_size])
            guidance_scale: Classifier-free guidance scale (0.0 = no guidance, 3.0-7.0 typical)
            return_intermediates: If True, return all intermediate timesteps
            use_dynamic_threshold: Whether to apply dynamic thresholding (default: True)
            dynamic_threshold_percentile: Percentile for dynamic thresholding (default: 0.995)

        Returns:
            Generated samples, shape (batch_size, in_channels, image_size, image_size)
            If return_intermediates is True, returns (num_denoising_steps, batch_size, in_channels, image_size, image_size)
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Add noise to x_0 to reach timestep t_0
        t = torch.full((batch_size,), t_0, device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        print(f"  - Starting SDEdit from timestep {t_0}/{self.num_timesteps}")

        intermediates = [x_t] if return_intermediates else None

        # Denoise from t_0 down to 0
        x = x_t
        for i in reversed(range(t_0)):
            if (i + 1) % 10 == 0:
                print(
                    f"  - Sampling timestep {i+1:04d}/{t_0:04d}",
                    end="\r",
                )
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(
                x,
                t,
                class_labels,
                guidance_scale,
                use_dynamic_threshold,
                dynamic_threshold_percentile,
            )

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return torch.stack(intermediates)

        return x

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training with optional class conditioning.

        Args:
            x: Input images, shape (batch_size, in_channels, image_size, image_size)
            t: Timesteps, shape (batch_size,). If None, random timesteps are sampled.
            class_labels: Class labels, shape (batch_size,). For classifier-free guidance training.

        Returns:
            Predicted noise and target noise
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(
                0, self.num_timesteps, (batch_size,), device=device
            ).long()

        # Apply class dropout for classifier-free guidance training
        if self.num_classes is not None and class_labels is not None and self.training:
            # Randomly drop class labels (replace with unconditional token)
            mask = torch.rand(batch_size, device=device) < self.class_dropout_prob
            class_labels = class_labels.clone()
            class_labels[mask] = (
                self.num_classes
            )  # Use num_classes as unconditional token

        # Sample noise
        noise = torch.randn_like(x)

        # Forward diffusion
        x_t = self.q_sample(x, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t, class_labels)

        return predicted_noise, noise

    def training_step(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        criterion: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Single training step with optional class conditioning.

        Args:
            x: Input images
            class_labels: Class labels for conditional generation
            criterion: Loss function (default: MSE)

        Returns:
            Loss value
        """
        if criterion is None:
            criterion = nn.MSELoss()

        predicted_noise, noise = self.forward(x, class_labels=class_labels)
        loss = criterion(predicted_noise, noise)

        return loss


def create_ddpm(
    image_size: int = 40,
    in_channels: int = 3,
    model_channels: int = 64,
    num_classes: Optional[int] = None,
    num_timesteps: int = 1000,
    beta_schedule: str = "cosine",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    class_dropout_prob: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DDPM:
    """Factory function to create a DDPM model.

    Args:
        image_size: Size of input images (assumes square)
        in_channels: Number of input channels
        model_channels: Base number of channels in U-Net
        num_classes: Number of classes for conditional generation (None for unconditional)
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Type of noise schedule ('linear', 'cosine', 'quadratic', 'sigmoid')
        beta_start: Starting beta value for noise schedule
        beta_end: Ending beta value for noise schedule
        class_dropout_prob: Probability of dropping class labels during training (for classifier-free guidance)
        device: Device to place the model on

    Returns:
        DDPM model
    """
    model = DDPM(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=model_channels,
        num_classes=num_classes,
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        class_dropout_prob=class_dropout_prob,
        device=device,
    )
    return model.to(device)


# Example usage for SDEdit (image-to-image generation):
"""
# Load a trained DDPM model
model = create_ddpm(
    image_size=40,
    in_channels=3,
    model_channels=64,
    num_classes=2,
    num_timesteps=1000,
    beta_schedule="cosine",
    device="cuda"
)
model.load_state_dict(torch.load("./out/ddpm/ddpm_model_ema.pth"))
model.eval()

# Load real Normal images (normalized to [-1, 1])
from torchvision import transforms, datasets
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
normal_dataset = datasets.ImageFolder("./data/stats/0.Normal", transform=transform)
normal_images = torch.stack([normal_dataset[i][0] for i in range(16)]).cuda()

# Generate Abnormal images from Normal backgrounds using SDEdit
# t_0 controls how much to change: 200-300 = subtle, 400-600 = more change
abnormal_class_labels = torch.ones(16, dtype=torch.long, device="cuda")  # Class 1 = Abnormal
generated_abnormals = model.sample_from_image(
    x_0=normal_images,
    t_0=300,  # Start denoising from timestep 300
    class_labels=abnormal_class_labels,
    guidance_scale=2.0,
    use_dynamic_threshold=True,
    dynamic_threshold_percentile=0.995
)

# Save generated images
from torchvision.utils import save_image
for i, img in enumerate(generated_abnormals):
    img_normalized = (img + 1.0) / 2.0
    save_image(img_normalized, f"./out/sdedit_abnormal_{i}.png")
"""
