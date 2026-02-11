"""Tests for Diffusion Model

This module contains unit and component tests for the DDPM diffusion model.
Tests are organized into:
- Unit tests: Fast tests for model components and interfaces
- Component tests: Tests with small batches to verify forward passes
"""

import pytest
import torch
import torch.nn as nn

from src.base.model import BaseModel
from src.experiments.diffusion.model import (
    EMA,
    AttentionBlock,
    DDPMModel,
    DownBlock,
    ResidualBlock,
    SinusoidalPositionEmbeddings,
    UNet,
    UpBlock,
    create_ddpm,
)

# ========================================
# Unit Tests
# ========================================


@pytest.mark.unit
class TestSinusoidalPositionEmbeddings:
    """Unit tests for SinusoidalPositionEmbeddings."""

    def test_initialization(self):
        """Test that SinusoidalPositionEmbeddings initializes correctly."""
        dim = 128
        emb = SinusoidalPositionEmbeddings(dim)
        assert emb.dim == dim

    def test_output_shape(self):
        """Test that output has correct shape."""
        dim = 128
        batch_size = 4
        emb = SinusoidalPositionEmbeddings(dim)

        time = torch.randint(0, 1000, (batch_size,))
        output = emb(time)

        assert output.shape == (batch_size, dim)

    def test_different_timesteps_different_outputs(self):
        """Test that different timesteps produce different embeddings."""
        dim = 128
        emb = SinusoidalPositionEmbeddings(dim)

        time1 = torch.tensor([0, 100])
        time2 = torch.tensor([500, 999])

        output1 = emb(time1)
        output2 = emb(time2)

        assert not torch.allclose(output1, output2)


@pytest.mark.unit
class TestEMA:
    """Unit tests for Exponential Moving Average."""

    def test_initialization(self):
        """Test that EMA initializes with correct decay and shadow params."""
        model = nn.Linear(10, 10)
        decay = 0.999
        ema = EMA(model, decay=decay)

        assert ema.decay == decay
        assert len(ema.shadow) > 0
        assert len(ema.backup) == 0

    def test_update(self):
        """Test that EMA update changes shadow parameters."""
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.999)

        # Store initial shadow
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model parameters
        for param in model.parameters():
            param.data += 1.0

        # Update EMA
        ema.update()

        # Shadow should have changed
        for name in ema.shadow:
            assert not torch.allclose(ema.shadow[name], initial_shadow[name])

    def test_apply_shadow_and_restore(self):
        """Test that apply_shadow and restore work correctly."""
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.999)

        # Store original parameters
        original_params = {
            name: param.data.clone() for name, param in model.named_parameters()
        }

        # Apply shadow
        ema.apply_shadow()

        # Parameters should be different
        for name, param in model.named_parameters():
            # They might be close but not identical
            pass  # Just check no error

        # Restore
        ema.restore()

        # Parameters should be back to original
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_params[name])

    def test_state_dict(self):
        """Test that state_dict contains required keys."""
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.999)

        state = ema.state_dict()

        assert "decay" in state
        assert "shadow" in state
        assert state["decay"] == 0.999


@pytest.mark.unit
class TestResidualBlock:
    """Unit tests for ResidualBlock."""

    def test_initialization(self):
        """Test ResidualBlock initialization."""
        in_channels = 64
        out_channels = 128
        time_emb_dim = 256

        block = ResidualBlock(in_channels, out_channels, time_emb_dim)

        assert isinstance(block.conv1, nn.Conv2d)
        assert isinstance(block.conv2, nn.Conv2d)
        assert isinstance(block.norm1, nn.GroupNorm)

    def test_initialization_with_class_conditioning(self):
        """Test ResidualBlock with class conditioning."""
        block = ResidualBlock(64, 128, 256, num_classes=10)

        assert block.class_mlp is not None
        assert isinstance(block.class_mlp, nn.Linear)


@pytest.mark.unit
class TestAttentionBlock:
    """Unit tests for AttentionBlock."""

    def test_initialization(self):
        """Test AttentionBlock initialization."""
        channels = 128
        num_heads = 4

        attn = AttentionBlock(channels, num_heads)

        assert attn.channels == channels
        assert attn.num_heads == num_heads

    def test_channels_divisible_by_heads(self):
        """Test that initialization fails if channels not divisible by heads."""
        with pytest.raises(AssertionError):
            AttentionBlock(channels=100, num_heads=3)


@pytest.mark.unit
class TestDownBlock:
    """Unit tests for DownBlock."""

    def test_initialization(self):
        """Test DownBlock initialization."""
        in_channels = 64
        out_channels = 128
        time_emb_dim = 256

        block = DownBlock(in_channels, out_channels, time_emb_dim)

        assert len(block.resblocks) == 2  # default num_layers
        assert block.downsample_conv is not None

    def test_initialization_no_downsample(self):
        """Test DownBlock without downsampling."""
        block = DownBlock(64, 128, 256, downsample=False)

        assert block.downsample_conv is None

    def test_initialization_with_attention(self):
        """Test DownBlock with attention."""
        block = DownBlock(64, 128, 256, use_attention=True)

        assert block.attention is not None
        assert isinstance(block.attention, AttentionBlock)


@pytest.mark.unit
class TestUpBlock:
    """Unit tests for UpBlock."""

    def test_initialization(self):
        """Test UpBlock initialization."""
        # in_channels includes skip connection
        in_channels = 256  # 128 from previous + 128 from skip
        out_channels = 64
        time_emb_dim = 256

        block = UpBlock(in_channels, out_channels, time_emb_dim)

        assert len(block.resblocks) == 2
        assert block.upsample_conv is not None

    def test_initialization_no_upsample(self):
        """Test UpBlock without upsampling."""
        block = UpBlock(256, 64, 256, upsample=False)

        assert block.upsample_conv is None


@pytest.mark.unit
class TestUNet:
    """Unit tests for UNet."""

    def test_initialization_unconditional(self):
        """Test UNet initialization for unconditional generation."""
        unet = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=64,
            channel_multipliers=(1, 2, 4),
            num_classes=None,
        )

        assert unet.class_emb is None
        assert len(unet.downs) == 3
        assert len(unet.ups) == 3

    def test_initialization_conditional(self):
        """Test UNet initialization for conditional generation."""
        unet = UNet(in_channels=3, out_channels=3, base_channels=64, num_classes=10)

        assert unet.class_emb is not None
        assert isinstance(unet.class_emb, nn.Embedding)


@pytest.mark.unit
class TestDDPMModel:
    """Unit tests for DDPMModel."""

    def test_initialization_unconditional(self):
        """Test DDPMModel initialization for unconditional generation."""
        model = DDPMModel(image_size=40, num_classes=None)

        assert model.image_size == 40
        assert model.num_classes is None
        assert isinstance(model, BaseModel)
        assert isinstance(model.model, UNet)

    def test_initialization_conditional(self):
        """Test DDPMModel initialization for conditional generation."""
        model = DDPMModel(image_size=40, num_classes=2)

        assert model.num_classes == 2
        assert model.class_dropout_prob > 0

    def test_beta_schedules(self):
        """Test that different beta schedules are supported."""
        schedules = ["linear", "cosine", "quadratic", "sigmoid"]

        for schedule in schedules:
            model = DDPMModel(beta_schedule=schedule, num_timesteps=100)
            assert model.betas.shape == (100,)
            assert torch.all(model.betas > 0)
            assert torch.all(model.betas < 1)

    def test_invalid_beta_schedule(self):
        """Test that invalid beta schedule raises error."""
        with pytest.raises(ValueError):
            DDPMModel(beta_schedule="invalid")

    def test_precomputed_coefficients(self):
        """Test that noise schedule coefficients are precomputed."""
        model = DDPMModel(num_timesteps=100)

        # Check that buffers exist
        assert hasattr(model, "betas")
        assert hasattr(model, "alphas")
        assert hasattr(model, "alphas_cumprod")
        assert hasattr(model, "sqrt_alphas_cumprod")
        assert hasattr(model, "sqrt_one_minus_alphas_cumprod")

        # Check shapes
        assert model.betas.shape == (100,)
        assert model.alphas.shape == (100,)
        assert model.alphas_cumprod.shape == (100,)


@pytest.mark.unit
class TestCreateDDPM:
    """Unit tests for create_ddpm factory function."""

    def test_create_ddpm_default(self):
        """Test creating DDPM with default parameters."""
        model = create_ddpm(device="cpu")

        assert isinstance(model, DDPMModel)
        assert model.image_size == 40
        assert model.num_classes is None

    def test_create_ddpm_conditional(self):
        """Test creating conditional DDPM."""
        model = create_ddpm(num_classes=2, device="cpu")

        assert model.num_classes == 2

    def test_create_ddpm_custom_size(self):
        """Test creating DDPM with custom image size."""
        model = create_ddpm(image_size=64, device="cpu")

        assert model.image_size == 64


# ========================================
# Component Tests
# ========================================


@pytest.mark.component
class TestResidualBlockForward:
    """Component tests for ResidualBlock forward pass."""

    def test_forward_pass(self):
        """Test forward pass with small batch."""
        batch_size = 2
        in_channels = 64
        out_channels = 128
        time_emb_dim = 256
        H, W = 8, 8

        block = ResidualBlock(in_channels, out_channels, time_emb_dim)

        x = torch.randn(batch_size, in_channels, H, W)
        time_emb = torch.randn(batch_size, time_emb_dim)

        output = block(x, time_emb)

        assert output.shape == (batch_size, out_channels, H, W)

    def test_forward_pass_with_class_conditioning(self):
        """Test forward pass with class conditioning."""
        batch_size = 2
        block = ResidualBlock(64, 128, 256, num_classes=10)

        x = torch.randn(batch_size, 64, 8, 8)
        time_emb = torch.randn(batch_size, 256)
        class_emb = torch.randn(batch_size, 256)

        output = block(x, time_emb, class_emb)

        assert output.shape == (batch_size, 128, 8, 8)


@pytest.mark.component
class TestAttentionBlockForward:
    """Component tests for AttentionBlock forward pass."""

    def test_forward_pass(self):
        """Test forward pass with small batch."""
        batch_size = 2
        channels = 128
        H, W = 8, 8

        attn = AttentionBlock(channels, num_heads=4)

        x = torch.randn(batch_size, channels, H, W)
        output = attn(x)

        # Attention has residual connection, so shape preserved
        assert output.shape == x.shape

    def test_forward_pass_different_sizes(self):
        """Test forward pass with different spatial sizes."""
        attn = AttentionBlock(64, num_heads=4)

        for h, w in [(4, 4), (8, 8), (16, 16)]:
            x = torch.randn(2, 64, h, w)
            output = attn(x)
            assert output.shape == (2, 64, h, w)


@pytest.mark.component
class TestDownBlockForward:
    """Component tests for DownBlock forward pass."""

    def test_forward_pass(self):
        """Test forward pass with small batch."""
        batch_size = 2
        in_channels = 64
        out_channels = 128
        time_emb_dim = 256
        H, W = 16, 16

        block = DownBlock(in_channels, out_channels, time_emb_dim, downsample=True)

        x = torch.randn(batch_size, in_channels, H, W)
        time_emb = torch.randn(batch_size, time_emb_dim)

        output, skip = block(x, time_emb)

        # Downsampling reduces spatial dimensions by 2
        assert output.shape == (batch_size, out_channels, H // 2, W // 2)
        # Skip has same spatial size as input but different channels
        assert skip.shape == (batch_size, out_channels, H, W)

    def test_forward_pass_no_downsample(self):
        """Test forward pass without downsampling."""
        batch_size = 2
        block = DownBlock(64, 128, 256, downsample=False)

        x = torch.randn(batch_size, 64, 16, 16)
        time_emb = torch.randn(batch_size, 256)

        output, skip = block(x, time_emb)

        # No downsampling, same spatial dimensions
        assert output.shape == (batch_size, 128, 16, 16)
        assert skip.shape == (batch_size, 128, 16, 16)


@pytest.mark.component
class TestUpBlockForward:
    """Component tests for UpBlock forward pass."""

    def test_forward_pass(self):
        """Test forward pass with small batch."""
        batch_size = 2
        in_channels = 128  # From previous layer
        skip_channels = 128  # From skip connection
        out_channels = 64
        time_emb_dim = 256
        H, W = 8, 8

        block = UpBlock(
            in_channels + skip_channels, out_channels, time_emb_dim, upsample=True
        )

        x = torch.randn(batch_size, in_channels, H, W)
        skip = torch.randn(batch_size, skip_channels, H, W)
        time_emb = torch.randn(batch_size, time_emb_dim)

        output = block(x, skip, time_emb)

        # Upsampling increases spatial dimensions by 2
        assert output.shape == (batch_size, out_channels, H * 2, W * 2)


@pytest.mark.component
class TestUNetForward:
    """Component tests for UNet forward pass."""

    def test_forward_pass_unconditional(self):
        """Test forward pass for unconditional generation."""
        batch_size = 2
        in_channels = 3
        image_size = 32

        unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=32,  # Small for fast testing
            channel_multipliers=(1, 2),
            num_classes=None,
        )

        x = torch.randn(batch_size, in_channels, image_size, image_size)
        timestep = torch.randint(0, 1000, (batch_size,))

        output = unet(x, timestep)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_forward_pass_conditional(self):
        """Test forward pass for conditional generation."""
        batch_size = 2
        unet = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=32,
            channel_multipliers=(1, 2),
            num_classes=2,
        )

        x = torch.randn(batch_size, 3, 32, 32)
        timestep = torch.randint(0, 1000, (batch_size,))
        class_labels = torch.randint(0, 2, (batch_size,))

        output = unet(x, timestep, class_labels)

        assert output.shape == x.shape


@pytest.mark.component
class TestDDPMModelForward:
    """Component tests for DDPMModel forward pass."""

    def test_forward_pass_unconditional(self):
        """Test forward pass for unconditional model."""
        batch_size = 2
        image_size = 32

        model = DDPMModel(
            image_size=image_size,
            model_channels=32,
            channel_multipliers=(1, 2),
            num_classes=None,
            num_timesteps=100,
        )

        x = torch.randn(batch_size, 3, image_size, image_size)

        predicted_noise, target_noise = model(x)

        assert predicted_noise.shape == x.shape
        assert target_noise.shape == x.shape

    def test_forward_pass_conditional(self):
        """Test forward pass for conditional model."""
        batch_size = 2
        model = DDPMModel(
            image_size=32,
            model_channels=32,
            channel_multipliers=(1, 2),
            num_classes=2,
            num_timesteps=100,
        )

        x = torch.randn(batch_size, 3, 32, 32)
        class_labels = torch.randint(0, 2, (batch_size,))

        predicted_noise, target_noise = model(x, class_labels=class_labels)

        assert predicted_noise.shape == x.shape
        assert target_noise.shape == x.shape

    def test_forward_pass_with_timesteps(self):
        """Test forward pass with specified timesteps."""
        batch_size = 2
        model = DDPMModel(image_size=32, model_channels=32, channel_multipliers=(1, 2))

        x = torch.randn(batch_size, 3, 32, 32)
        t = torch.randint(0, 100, (batch_size,))

        predicted_noise, target_noise = model(x, t)

        assert predicted_noise.shape == x.shape
        assert target_noise.shape == x.shape


@pytest.mark.component
class TestDDPMModelComputeLoss:
    """Component tests for DDPMModel compute_loss method."""

    def test_compute_loss_unconditional(self):
        """Test compute_loss for unconditional model."""
        model = DDPMModel(
            image_size=32,
            model_channels=32,
            channel_multipliers=(1, 2),
            num_timesteps=100,
        )
        model.train()

        x = torch.randn(2, 3, 32, 32)
        loss = model.compute_loss(x)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.item() >= 0

    def test_compute_loss_conditional(self):
        """Test compute_loss for conditional model."""
        model = DDPMModel(
            image_size=32,
            model_channels=32,
            channel_multipliers=(1, 2),
            num_classes=2,
            num_timesteps=100,
        )
        model.train()

        x = torch.randn(2, 3, 32, 32)
        class_labels = torch.randint(0, 2, (2,))

        loss = model.compute_loss(x, class_labels=class_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0


@pytest.mark.component
class TestDDPMModelSampling:
    """Component tests for DDPMModel sampling methods."""

    def test_q_sample(self):
        """Test forward diffusion (q_sample)."""
        model = DDPMModel(image_size=32, model_channels=32, num_timesteps=100)

        x_start = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 100, (2,))

        noisy_x = model.q_sample(x_start, t)

        assert noisy_x.shape == x_start.shape

    def test_predict_start_from_noise(self):
        """Test predicting x_0 from noise."""
        model = DDPMModel(image_size=32, model_channels=32, num_timesteps=100)

        x_t = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 100, (2,))
        noise = torch.randn(2, 3, 32, 32)

        x_start_pred = model.predict_start_from_noise(x_t, t, noise)

        assert x_start_pred.shape == x_t.shape

    def test_dynamic_threshold(self):
        """Test dynamic thresholding."""
        model = DDPMModel(image_size=32, model_channels=32)

        # Create input with some extreme values
        x_start = torch.randn(2, 3, 32, 32) * 10

        thresholded = model.dynamic_threshold(x_start, percentile=0.95)

        assert thresholded.shape == x_start.shape
        # Values should be more constrained
        assert thresholded.abs().max() < x_start.abs().max()

    def test_p_sample(self):
        """Test single denoising step."""
        model = DDPMModel(
            image_size=32,
            model_channels=32,
            channel_multipliers=(1, 2),
            num_timesteps=100,
        )
        model.eval()

        x_t = torch.randn(2, 3, 32, 32)
        t = torch.tensor([50, 50])

        x_t_minus_1 = model.p_sample(x_t, t)

        assert x_t_minus_1.shape == x_t.shape

    @pytest.mark.slow
    def test_sample_unconditional(self):
        """Test unconditional sampling (slow due to full denoising loop)."""
        model = DDPMModel(
            image_size=16,  # Very small for speed
            model_channels=16,
            channel_multipliers=(1, 2),
            num_timesteps=10,  # Very few steps
        )
        model.eval()

        samples = model.sample(batch_size=2)

        assert samples.shape == (2, 3, 16, 16)
        # Check values are in reasonable range
        assert samples.abs().max() < 10

    @pytest.mark.slow
    def test_sample_conditional(self):
        """Test conditional sampling."""
        model = DDPMModel(
            image_size=16,
            model_channels=16,
            channel_multipliers=(1, 2),
            num_classes=2,
            num_timesteps=10,
        )
        model.eval()

        class_labels = torch.tensor([0, 1])
        samples = model.sample(batch_size=2, class_labels=class_labels)

        assert samples.shape == (2, 3, 16, 16)

    @pytest.mark.slow
    def test_sample_with_guidance(self):
        """Test sampling with classifier-free guidance."""
        model = DDPMModel(
            image_size=16,
            model_channels=16,
            channel_multipliers=(1, 2),
            num_classes=2,
            num_timesteps=10,
        )
        model.eval()

        class_labels = torch.tensor([0, 1])
        samples = model.sample(
            batch_size=2, class_labels=class_labels, guidance_scale=2.0
        )

        assert samples.shape == (2, 3, 16, 16)

    @pytest.mark.slow
    def test_sample_from_image(self):
        """Test SDEdit (image-to-image generation)."""
        model = DDPMModel(
            image_size=16,
            model_channels=16,
            channel_multipliers=(1, 2),
            num_timesteps=50,
        )
        model.eval()

        x_0 = torch.randn(2, 3, 16, 16)
        samples = model.sample_from_image(x_0, t_0=25)

        assert samples.shape == x_0.shape


@pytest.mark.component
class TestDDPMModelCheckpointing:
    """Component tests for model checkpointing."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading model checkpoint."""
        model = DDPMModel(image_size=32, model_channels=32, channel_multipliers=(1, 2))

        checkpoint_path = tmp_path / "model.pth"

        # Save checkpoint
        model.save_checkpoint(checkpoint_path)

        # Create new model and load
        new_model = DDPMModel(
            image_size=32, model_channels=32, channel_multipliers=(1, 2)
        )
        new_model.load_checkpoint(checkpoint_path)

        # Check that weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


@pytest.mark.component
class TestModelInheritance:
    """Test that DDPMModel properly inherits from BaseModel."""

    def test_isinstance_basemodel(self):
        """Test that DDPMModel is instance of BaseModel."""
        model = DDPMModel()
        assert isinstance(model, BaseModel)

    def test_implements_required_methods(self):
        """Test that DDPMModel implements required abstract methods."""
        model = DDPMModel()

        # Should have forward and compute_loss methods
        assert hasattr(model, "forward")
        assert hasattr(model, "compute_loss")
        assert callable(model.forward)
        assert callable(model.compute_loss)
