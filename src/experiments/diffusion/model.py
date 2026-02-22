"""Diffusion Model for Image Generation

This module provides a DDPM (Denoising Diffusion Probabilistic Model) implementation
for image generation. The model supports:
- Unconditional and conditional generation
- Multiple noise schedules (linear, cosine, quadratic, sigmoid)
- Classifier-free guidance for conditional generation
- SDEdit for image-to-image generation
- EMA (Exponential Moving Average) for improved sample quality
- Dynamic thresholding to reduce artifacts

The implementation is based on the DDPM paper with enhancements for
better sample quality and training stability.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.base.model import BaseModel


class EMA:
    """Exponential Moving Average of model parameters.

    Keeps a moving average of model weights during training to improve sampling quality.
    The EMA weights are typically more stable and produce cleaner outputs with fewer artifacts.

    Args:
        model: The model to track with EMA
        decay: Decay rate for the moving average (default: 0.9999)
        device: Device to store EMA weights on

    Example:
        >>> model = UNet()
        >>> ema = EMA(model, decay=0.9999)
        >>> # During training
        >>> loss.backward()
        >>> optimizer.step()
        >>> ema.update()
        >>> # For inference
        >>> ema.apply_shadow()
        >>> samples = model.sample()
        >>> ema.restore()
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
        # Move shadow parameters to the correct device
        # Strip torch.compile prefix if present
        shadow = state_dict["shadow"]
        self.shadow = {
            name.removeprefix("_orig_mod."): tensor.to(self.device)
            for name, tensor in shadow.items()
        }


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding.

    Encodes timesteps using sinusoidal functions of different frequencies,
    similar to positional encodings in Transformers.

    Args:
        dim: Embedding dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings for given timesteps.

        Args:
            time: Timesteps, shape (batch_size,)

        Returns:
            Embeddings, shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding and optional class conditioning.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embeddings
        num_classes: Number of classes for conditional generation (None for unconditional)
        dropout: Dropout probability
    """

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
        """Forward pass with time and optional class conditioning.

        Args:
            x: Input tensor, shape (batch_size, in_channels, H, W)
            time_emb: Time embeddings, shape (batch_size, time_emb_dim)
            class_emb: Class embeddings, shape (batch_size, time_emb_dim) (optional)

        Returns:
            Output tensor, shape (batch_size, out_channels, H, W)
        """
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
    """Self-attention block for capturing long-range dependencies.

    Args:
        channels: Number of channels
        num_heads: Number of attention heads
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor, shape (batch_size, channels, H, W)

        Returns:
            Output tensor, shape (batch_size, channels, H, W)
        """
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
    """Downsampling block with residual blocks and optional attention.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embeddings
        num_classes: Number of classes for conditional generation
        num_layers: Number of residual blocks
        downsample: Whether to downsample spatially
        use_attention: Whether to use self-attention
    """

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
        """Forward pass through downsampling block.

        Args:
            x: Input tensor
            time_emb: Time embeddings
            class_emb: Class embeddings (optional)

        Returns:
            Tuple of (downsampled_output, skip_connection)
        """
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
    """Upsampling block with residual blocks and optional attention.

    Args:
        in_channels: Number of input channels (includes skip connection)
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embeddings
        num_classes: Number of classes for conditional generation
        num_layers: Number of residual blocks
        upsample: Whether to upsample spatially
        use_attention: Whether to use self-attention
    """

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
            # Use bilinear interpolation + Conv2d instead of ConvTranspose2d
            # This eliminates checkerboard artifacts and produces smoother boundaries
            self.upsample_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
        """Forward pass through upsampling block.

        Args:
            x: Input tensor
            skip: Skip connection from encoder
            time_emb: Time embeddings
            class_emb: Class embeddings (optional)

        Returns:
            Upsampled output tensor
        """
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
    """U-Net architecture for DDPM with optional class conditioning.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        channel_multipliers: Channel multipliers for each resolution level
        num_res_blocks: Number of residual blocks per level
        time_emb_dim: Dimension of time embeddings
        num_classes: Number of classes for conditional generation
        dropout: Dropout probability
        use_attention: Which levels to use attention at
    """

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
        """Forward pass through U-Net.

        Args:
            x: Noisy input, shape (batch_size, in_channels, H, W)
            timestep: Timesteps, shape (batch_size,)
            class_labels: Class labels, shape (batch_size,) (optional)

        Returns:
            Predicted noise, shape (batch_size, out_channels, H, W)
        """
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


class DDPMModel(BaseModel):
    """Denoising Diffusion Probabilistic Model with Classifier-Free Guidance.

    This model implements DDPM for image generation with support for:
    - Unconditional and conditional generation
    - Multiple noise schedules (linear, cosine, quadratic, sigmoid)
    - Classifier-free guidance for improved conditional generation
    - SDEdit for image-to-image generation
    - Dynamic thresholding to reduce artifacts

    Args:
        image_size: Size of the input images (assumes square images)
        in_channels: Number of input channels
        model_channels: Base number of channels in the U-Net
        channel_multipliers: Channel multipliers for each U-Net stage
        num_res_blocks: Number of residual blocks per stage
        num_classes: Number of classes for conditional generation (None for unconditional)
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Type of noise schedule ('linear', 'cosine', 'quadratic', 'sigmoid')
        beta_start: Starting beta value for noise schedule
        beta_end: Ending beta value for noise schedule
        class_dropout_prob: Probability of dropping class labels during training
        use_attention: Which stages to use attention at
        device: Device to place the model on

    Example:
        >>> # Unconditional generation
        >>> model = DDPMModel(image_size=40, num_classes=None)
        >>> samples = model.sample(batch_size=16)
        >>>
        >>> # Conditional generation with classifier-free guidance
        >>> model = DDPMModel(image_size=40, num_classes=2)
        >>> class_labels = torch.tensor([0, 1, 0, 1])
        >>> samples = model.sample(batch_size=4, class_labels=class_labels, guidance_scale=2.0)
        >>>
        >>> # Training
        >>> images = torch.randn(16, 3, 40, 40)
        >>> loss = model.compute_loss(images, class_labels=class_labels)
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
        use_attention: Tuple[bool, ...] = (False, False, True),
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
            use_attention=use_attention,
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
            beta_start: Starting beta value
            beta_end: Ending beta value

        Returns:
            Beta schedule tensor

        Raises:
            ValueError: If schedule type is unknown
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
        """Extract coefficients at specified timesteps and reshape to match x_shape.

        Args:
            a: Coefficient tensor
            t: Timesteps
            x_shape: Shape to match

        Returns:
            Extracted and reshaped coefficients
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0).

        Args:
            x_start: Clean images
            t: Timesteps
            noise: Noise to add (sampled if None)

        Returns:
            Noisy images at timestep t
        """
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
        """Predict x_0 from x_t and predicted noise.

        Args:
            x_t: Noisy images
            t: Timesteps
            noise: Predicted noise

        Returns:
            Predicted clean images
        """
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
            percentile: Percentile for dynamic threshold (default: 0.995)

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
        s = s.view(-1, 1, 1, 1)  # Reshape for broadcasting
        x_start = torch.max(torch.min(x_start, s), -s) / s

        return x_start

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance of diffusion posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Predicted clean images
            x_t: Noisy images
            t: Timesteps

        Returns:
            Tuple of (posterior_mean, posterior_variance)
        """
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
        """Compute predicted mean and variance for reverse process.

        Args:
            x_t: Noisy input
            t: Timestep
            class_labels: Class labels for conditional generation
            guidance_scale: Classifier-free guidance scale
            use_dynamic_threshold: Whether to apply dynamic thresholding
            dynamic_threshold_percentile: Percentile for dynamic thresholding

        Returns:
            Tuple of (model_mean, model_variance)
        """
        if (
            self.num_classes is not None
            and guidance_scale > 0.0
            and class_labels is not None
        ):
            # Classifier-free guidance
            predicted_noise_cond = self.model(x_t, t, class_labels)

            # Unconditional prediction
            batch_size = x_t.shape[0]
            unconditional_labels = torch.full(
                (batch_size,), self.num_classes, device=x_t.device, dtype=torch.long
            )
            predicted_noise_uncond = self.model(x_t, t, unconditional_labels)

            # Apply guidance
            predicted_noise = predicted_noise_uncond + guidance_scale * (
                predicted_noise_cond - predicted_noise_uncond
            )
        else:
            # Standard prediction without guidance
            predicted_noise = self.model(x_t, t, class_labels)

        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)

        # Apply dynamic thresholding or fixed clamp
        if use_dynamic_threshold:
            x_start = self.dynamic_threshold(x_start, dynamic_threshold_percentile)
        else:
            x_start = torch.clamp(x_start, -1.0, 1.0)

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
        """Sample from reverse process: p(x_{t-1} | x_t).

        Args:
            x_t: Noisy input
            t: Timestep
            class_labels: Class labels
            guidance_scale: Guidance scale
            use_dynamic_threshold: Apply dynamic thresholding
            dynamic_threshold_percentile: Percentile for thresholding

        Returns:
            Denoised sample at t-1
        """
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
        show_progress: bool = False,
        progress_desc: str = "Denoising",
    ) -> torch.Tensor:
        """Generate samples from the model.

        Args:
            batch_size: Number of samples to generate
            class_labels: Class labels for conditional generation
            guidance_scale: Classifier-free guidance scale
            return_intermediates: Return all intermediate timesteps
            use_dynamic_threshold: Apply dynamic thresholding
            dynamic_threshold_percentile: Percentile for thresholding
            show_progress: Show tqdm progress bar for denoising steps
            progress_desc: Description label for the progress bar

        Returns:
            Generated samples, shape (batch_size, in_channels, image_size, image_size)
            If return_intermediates is True, shape (num_timesteps, batch_size, ...)
        """
        shape = (batch_size, self.in_channels, self.image_size, self.image_size)
        device = next(self.parameters()).device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        intermediates = [x] if return_intermediates else None

        # Iteratively denoise
        timesteps = reversed(range(self.num_timesteps))
        if show_progress:
            timesteps = tqdm(
                timesteps,
                total=self.num_timesteps,
                desc=progress_desc,
                leave=False,
            )

        for i in timesteps:
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
        """SDEdit: Generate samples by starting from a real image.

        Args:
            x_0: Starting images (normalized to [-1, 1])
            t_0: Starting timestep for denoising
            class_labels: Target class labels
            guidance_scale: Guidance scale
            return_intermediates: Return all intermediate timesteps
            use_dynamic_threshold: Apply dynamic thresholding
            dynamic_threshold_percentile: Percentile for thresholding

        Returns:
            Generated samples
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Add noise to x_0 to reach timestep t_0
        t = torch.full((batch_size,), t_0, device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        intermediates = [x_t] if return_intermediates else None

        # Denoise from t_0 down to 0
        x = x_t
        for i in reversed(range(t_0)):
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training with optional class conditioning.

        Args:
            x: Input images, shape (batch_size, in_channels, image_size, image_size)
            t: Timesteps, shape (batch_size,). If None, random timesteps are sampled.
            class_labels: Class labels, shape (batch_size,)

        Returns:
            Tuple of (predicted_noise, target_noise)
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
            # Randomly drop class labels
            mask = torch.rand(batch_size, device=device) < self.class_dropout_prob
            class_labels = class_labels.clone()
            class_labels[mask] = self.num_classes  # Unconditional token

        # Sample noise
        noise = torch.randn_like(x)

        # Forward diffusion
        x_t = self.q_sample(x, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t, class_labels)

        return predicted_noise, noise

    def compute_loss(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        criterion: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute the loss for training.

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
    channel_multipliers: Tuple[int, ...] = (1, 2, 4),
    num_classes: Optional[int] = None,
    num_timesteps: int = 1000,
    beta_schedule: str = "cosine",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    class_dropout_prob: float = 0.1,
    use_attention: Tuple[bool, ...] = (False, False, True),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DDPMModel:
    """Factory function to create a DDPM model.

    Args:
        image_size: Size of input images
        in_channels: Number of input channels
        model_channels: Base number of channels in U-Net
        channel_multipliers: Channel multipliers for each stage
        num_classes: Number of classes for conditional generation
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Type of noise schedule
        beta_start: Starting beta value
        beta_end: Ending beta value
        class_dropout_prob: Probability of dropping class labels
        use_attention: Which stages to use attention at
        device: Device to place the model on

    Returns:
        DDPM model

    Example:
        >>> model = create_ddpm(image_size=40, num_classes=2, device="cuda")
        >>> samples = model.sample(batch_size=16)
    """
    model = DDPMModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=model_channels,
        channel_multipliers=channel_multipliers,
        num_classes=num_classes,
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        class_dropout_prob=class_dropout_prob,
        use_attention=use_attention,
        device=device,
    )
    return model.to(device)
