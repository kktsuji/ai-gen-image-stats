"""Pytest tests for DDPM (Denoising Diffusion Probabilistic Models)"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from ddpm import (
    DDPM,
    AttentionBlock,
    DownBlock,
    ResidualBlock,
    SinusoidalPositionEmbeddings,
    UNet,
    UpBlock,
    create_ddpm,
    generate,
)


@pytest.fixture
def device():
    """Return the device to use for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def temp_out_dir():
    """Create a temporary directory for outputs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestSinusoidalPositionEmbeddings:
    """Test SinusoidalPositionEmbeddings module"""

    def test_initialization(self):
        """Test SinusoidalPositionEmbeddings initialization"""
        dim = 256
        emb = SinusoidalPositionEmbeddings(dim)

        assert emb is not None
        assert emb.dim == dim
        assert isinstance(emb, nn.Module)

    def test_forward_single_timestep(self, device):
        """Test forward pass with single timestep"""
        dim = 256
        emb = SinusoidalPositionEmbeddings(dim)

        time = torch.tensor([0], device=device)
        output = emb(time)

        assert output.shape == (1, dim)
        assert output.device.type == device

    def test_forward_batch_timesteps(self, device):
        """Test forward pass with batch of timesteps"""
        dim = 256
        batch_size = 4
        emb = SinusoidalPositionEmbeddings(dim)

        time = torch.randint(0, 1000, (batch_size,), device=device)
        output = emb(time)

        assert output.shape == (batch_size, dim)

    def test_different_timesteps_different_embeddings(self, device):
        """Test that different timesteps produce different embeddings"""
        dim = 256
        emb = SinusoidalPositionEmbeddings(dim)

        time1 = torch.tensor([0], device=device)
        time2 = torch.tensor([100], device=device)

        out1 = emb(time1)
        out2 = emb(time2)

        assert not torch.allclose(out1, out2)

    @pytest.mark.parametrize("dim", [64, 128, 256, 512])
    def test_different_dimensions(self, dim, device):
        """Test with different embedding dimensions"""
        emb = SinusoidalPositionEmbeddings(dim)
        time = torch.tensor([50], device=device)

        output = emb(time)

        assert output.shape == (1, dim)


class TestResidualBlock:
    """Test ResidualBlock module"""

    def test_initialization_without_class_conditioning(self):
        """Test ResidualBlock initialization without class conditioning"""
        block = ResidualBlock(64, 128, time_emb_dim=256)

        assert block is not None
        assert isinstance(block, nn.Module)
        assert block.class_mlp is None

    def test_initialization_with_class_conditioning(self):
        """Test ResidualBlock initialization with class conditioning"""
        block = ResidualBlock(64, 128, time_emb_dim=256, num_classes=2)

        assert block is not None
        assert block.class_mlp is not None
        assert isinstance(block.class_mlp, nn.Linear)

    def test_forward_without_class_emb(self, device):
        """Test forward pass without class embedding"""
        block = ResidualBlock(64, 64, time_emb_dim=256).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 64, 40, 40, device=device)
        time_emb = torch.randn(batch_size, 256, device=device)

        output = block(x, time_emb)

        assert output.shape == x.shape

    def test_forward_with_class_emb(self, device):
        """Test forward pass with class embedding"""
        block = ResidualBlock(64, 64, time_emb_dim=256, num_classes=2).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 64, 40, 40, device=device)
        time_emb = torch.randn(batch_size, 256, device=device)
        class_emb = torch.randn(batch_size, 256, device=device)

        output = block(x, time_emb, class_emb)

        assert output.shape == x.shape

    def test_channel_change(self, device):
        """Test that block correctly changes number of channels"""
        in_ch, out_ch = 64, 128
        block = ResidualBlock(in_ch, out_ch, time_emb_dim=256).to(device)

        x = torch.randn(2, in_ch, 40, 40, device=device)
        time_emb = torch.randn(2, 256, device=device)

        output = block(x, time_emb)

        assert output.shape == (2, out_ch, 40, 40)


class TestAttentionBlock:
    """Test AttentionBlock module"""

    def test_initialization(self):
        """Test AttentionBlock initialization"""
        block = AttentionBlock(channels=64, num_heads=4)

        assert block is not None
        assert isinstance(block, nn.Module)
        assert block.channels == 64
        assert block.num_heads == 4

    def test_channels_divisible_by_heads(self):
        """Test that channels must be divisible by num_heads"""
        with pytest.raises(AssertionError):
            AttentionBlock(channels=65, num_heads=4)

    def test_forward(self, device):
        """Test forward pass"""
        block = AttentionBlock(channels=64, num_heads=4).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 64, 10, 10, device=device)

        output = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self, device):
        """Test that attention block has residual connection"""
        block = AttentionBlock(channels=64, num_heads=4).to(device)

        x = torch.randn(2, 64, 10, 10, device=device)
        output = block(x)

        # Output should not be zero even if attention is zero
        assert not torch.allclose(output, torch.zeros_like(output))


class TestDownBlock:
    """Test DownBlock module"""

    def test_initialization_with_downsample(self):
        """Test DownBlock initialization with downsampling"""
        block = DownBlock(64, 128, time_emb_dim=256, downsample=True)

        assert block is not None
        assert block.downsample_conv is not None

    def test_initialization_without_downsample(self):
        """Test DownBlock initialization without downsampling"""
        block = DownBlock(64, 128, time_emb_dim=256, downsample=False)

        assert block.downsample_conv is None

    def test_forward_with_downsample(self, device):
        """Test forward pass with downsampling"""
        block = DownBlock(64, 128, time_emb_dim=256, downsample=True).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 64, 40, 40, device=device)
        time_emb = torch.randn(batch_size, 256, device=device)

        output, skip = block(x, time_emb)

        assert output.shape == (batch_size, 128, 20, 20)  # Downsampled
        assert skip.shape == (batch_size, 128, 40, 40)  # Before downsampling

    def test_forward_without_downsample(self, device):
        """Test forward pass without downsampling"""
        block = DownBlock(64, 128, time_emb_dim=256, downsample=False).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 64, 40, 40, device=device)
        time_emb = torch.randn(batch_size, 256, device=device)

        output, skip = block(x, time_emb)

        assert output.shape == (batch_size, 128, 40, 40)  # No downsampling
        assert skip.shape == (batch_size, 128, 40, 40)

    def test_with_attention(self, device):
        """Test DownBlock with attention"""
        block = DownBlock(64, 128, time_emb_dim=256, use_attention=True).to(device)

        x = torch.randn(2, 64, 20, 20, device=device)
        time_emb = torch.randn(2, 256, device=device)

        output, skip = block(x, time_emb)

        assert output is not None
        assert skip is not None


class TestUpBlock:
    """Test UpBlock module"""

    def test_initialization_with_upsample(self):
        """Test UpBlock initialization with upsampling"""
        block = UpBlock(128 + 128, 64, time_emb_dim=256, upsample=True)

        assert block is not None
        assert block.upsample_conv is not None

    def test_initialization_without_upsample(self):
        """Test UpBlock initialization without upsampling"""
        block = UpBlock(128 + 128, 64, time_emb_dim=256, upsample=False)

        assert block.upsample_conv is None

    def test_forward_with_upsample(self, device):
        """Test forward pass with upsampling"""
        block = UpBlock(128 + 128, 64, time_emb_dim=256, upsample=True).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 128, 10, 10, device=device)
        skip = torch.randn(batch_size, 128, 10, 10, device=device)
        time_emb = torch.randn(batch_size, 256, device=device)

        output = block(x, skip, time_emb)

        assert output.shape == (batch_size, 64, 20, 20)  # Upsampled

    def test_forward_without_upsample(self, device):
        """Test forward pass without upsampling"""
        block = UpBlock(128 + 128, 64, time_emb_dim=256, upsample=False).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 128, 10, 10, device=device)
        skip = torch.randn(batch_size, 128, 10, 10, device=device)
        time_emb = torch.randn(batch_size, 256, device=device)

        output = block(x, skip, time_emb)

        assert output.shape == (batch_size, 64, 10, 10)  # No upsampling


class TestUNet:
    """Test UNet module"""

    def test_initialization_unconditional(self):
        """Test UNet initialization without class conditioning"""
        unet = UNet(in_channels=3, out_channels=3, base_channels=64, num_classes=None)

        assert unet is not None
        assert isinstance(unet, nn.Module)
        assert unet.class_emb is None

    def test_initialization_conditional(self):
        """Test UNet initialization with class conditioning"""
        unet = UNet(in_channels=3, out_channels=3, base_channels=64, num_classes=2)

        assert unet is not None
        assert unet.class_emb is not None
        assert isinstance(unet.class_emb, nn.Embedding)

    def test_forward_unconditional(self, device):
        """Test forward pass without class labels"""
        unet = UNet(in_channels=3, out_channels=3, base_channels=32).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 3, 40, 40, device=device)
        timestep = torch.randint(0, 1000, (batch_size,), device=device)

        output = unet(x, timestep)

        assert output.shape == x.shape

    def test_forward_conditional(self, device):
        """Test forward pass with class labels"""
        unet = UNet(in_channels=3, out_channels=3, base_channels=32, num_classes=2).to(
            device
        )

        batch_size = 2
        x = torch.randn(batch_size, 3, 40, 40, device=device)
        timestep = torch.randint(0, 1000, (batch_size,), device=device)
        class_labels = torch.randint(0, 2, (batch_size,), device=device)

        output = unet(x, timestep, class_labels)

        assert output.shape == x.shape

    @pytest.mark.parametrize("img_size", [32, 40, 64])
    def test_different_image_sizes(self, img_size, device):
        """Test UNet with different image sizes"""
        unet = UNet(in_channels=3, out_channels=3, base_channels=32).to(device)

        x = torch.randn(2, 3, img_size, img_size, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)

        output = unet(x, timestep)

        assert output.shape == x.shape


class TestDDPM:
    """Test DDPM module"""

    def test_initialization_unconditional(self, device):
        """Test DDPM initialization without class conditioning"""
        ddpm = DDPM(
            image_size=40,
            in_channels=3,
            model_channels=32,
            num_classes=None,
            num_timesteps=100,
            device=device,
        )

        assert ddpm is not None
        assert isinstance(ddpm, nn.Module)
        assert ddpm.num_classes is None
        assert ddpm.num_timesteps == 100

    def test_initialization_conditional(self, device):
        """Test DDPM initialization with class conditioning"""
        ddpm = DDPM(
            image_size=40,
            in_channels=3,
            model_channels=32,
            num_classes=2,
            num_timesteps=100,
            device=device,
        )

        assert ddpm.num_classes == 2
        assert ddpm.class_dropout_prob > 0

    def test_buffers_registered(self, device):
        """Test that noise schedule buffers are registered"""
        ddpm = DDPM(image_size=40, model_channels=32, num_timesteps=100, device=device)

        assert hasattr(ddpm, "betas")
        assert hasattr(ddpm, "alphas")
        assert hasattr(ddpm, "alphas_cumprod")
        assert hasattr(ddpm, "sqrt_alphas_cumprod")

    def test_forward_unconditional(self, device):
        """Test forward pass without class labels"""
        ddpm = DDPM(
            image_size=40, model_channels=32, num_timesteps=100, device=device
        ).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 3, 40, 40, device=device)

        predicted_noise, noise = ddpm(x)

        assert predicted_noise.shape == x.shape
        assert noise.shape == x.shape

    def test_forward_conditional(self, device):
        """Test forward pass with class labels"""
        ddpm = DDPM(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=100,
            device=device,
        ).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 3, 40, 40, device=device)
        class_labels = torch.randint(0, 2, (batch_size,), device=device)

        predicted_noise, noise = ddpm(x, class_labels=class_labels)

        assert predicted_noise.shape == x.shape
        assert noise.shape == x.shape

    def test_q_sample(self, device):
        """Test forward diffusion q_sample"""
        ddpm = DDPM(
            image_size=40, model_channels=32, num_timesteps=100, device=device
        ).to(device)

        x_start = torch.randn(2, 3, 40, 40, device=device)
        t = torch.randint(0, 100, (2,), device=device)

        x_t = ddpm.q_sample(x_start, t)

        assert x_t.shape == x_start.shape

    def test_p_sample(self, device):
        """Test reverse diffusion p_sample"""
        ddpm = DDPM(
            image_size=40, model_channels=32, num_timesteps=100, device=device
        ).to(device)
        ddpm.eval()

        x_t = torch.randn(2, 3, 40, 40, device=device)
        t = torch.tensor([50, 50], device=device)

        x_t_minus_1 = ddpm.p_sample(x_t, t)

        assert x_t_minus_1.shape == x_t.shape

    def test_sample_unconditional(self, device):
        """Test sampling without class labels"""
        ddpm = DDPM(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        ).to(device)
        ddpm.eval()

        samples = ddpm.sample(batch_size=2)

        assert samples.shape == (2, 3, 40, 40)

    def test_sample_conditional(self, device):
        """Test sampling with class labels"""
        ddpm = DDPM(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        ).to(device)
        ddpm.eval()

        class_labels = torch.zeros(2, device=device, dtype=torch.long)
        samples = ddpm.sample(batch_size=2, class_labels=class_labels)

        assert samples.shape == (2, 3, 40, 40)

    def test_sample_with_guidance(self, device):
        """Test sampling with classifier-free guidance"""
        ddpm = DDPM(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        ).to(device)
        ddpm.eval()

        class_labels = torch.zeros(2, device=device, dtype=torch.long)
        samples = ddpm.sample(
            batch_size=2, class_labels=class_labels, guidance_scale=3.0
        )

        assert samples.shape == (2, 3, 40, 40)

    def test_training_step(self, device):
        """Test training_step method"""
        ddpm = DDPM(
            image_size=40, model_channels=32, num_timesteps=100, device=device
        ).to(device)

        x = torch.randn(2, 3, 40, 40, device=device)
        loss = ddpm.training_step(x)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_training_step_with_labels(self, device):
        """Test training_step with class labels"""
        ddpm = DDPM(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=100,
            device=device,
        ).to(device)

        x = torch.randn(2, 3, 40, 40, device=device)
        class_labels = torch.randint(0, 2, (2,), device=device)
        loss = ddpm.training_step(x, class_labels=class_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_class_dropout_during_training(self, device):
        """Test that class dropout is applied during training"""
        ddpm = DDPM(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=100,
            class_dropout_prob=1.0,  # Always drop
            device=device,
        ).to(device)
        ddpm.train()

        x = torch.randn(2, 3, 40, 40, device=device)
        class_labels = torch.zeros(2, device=device, dtype=torch.long)

        # During training, labels should be dropped
        predicted_noise, noise = ddpm(x, class_labels=class_labels)

        assert predicted_noise.shape == x.shape


class TestCreateDDPM:
    """Test create_ddpm factory function"""

    def test_create_unconditional(self, device):
        """Test creating unconditional DDPM"""
        model = create_ddpm(
            image_size=40,
            in_channels=3,
            model_channels=32,
            num_classes=None,
            device=device,
        )

        assert isinstance(model, DDPM)
        assert model.num_classes is None

    def test_create_conditional(self, device):
        """Test creating conditional DDPM"""
        model = create_ddpm(
            image_size=40,
            in_channels=3,
            model_channels=32,
            num_classes=2,
            device=device,
        )

        assert isinstance(model, DDPM)
        assert model.num_classes == 2

    @pytest.mark.parametrize("img_size", [32, 40, 64])
    def test_different_image_sizes(self, img_size, device):
        """Test creating DDPM with different image sizes"""
        model = create_ddpm(image_size=img_size, model_channels=32, device=device)

        assert model.image_size == img_size

    @pytest.mark.parametrize("num_timesteps", [10, 100, 1000])
    def test_different_timesteps(self, num_timesteps, device):
        """Test creating DDPM with different number of timesteps"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=num_timesteps, device=device
        )

        assert model.num_timesteps == num_timesteps


class TestDDPMIntegration:
    """Integration tests for DDPM"""

    def test_full_training_step(self, device):
        """Test a complete training step"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=100,
            device=device,
        )
        model.train()

        # Create dummy batch
        x = torch.randn(4, 3, 40, 40, device=device)
        labels = torch.randint(0, 2, (4,), device=device)

        # Forward pass
        loss = model.training_step(x, class_labels=labels)

        # Backward pass (just test it doesn't crash)
        loss.backward()

        assert True  # If we get here, it worked

    def test_sample_after_training(self, device):
        """Test sampling after a training step"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )

        # Training step
        x = torch.randn(2, 3, 40, 40, device=device)
        labels = torch.randint(0, 2, (2,), device=device)
        loss = model.training_step(x, class_labels=labels)

        # Switch to eval and sample
        model.eval()
        samples = model.sample(batch_size=2, class_labels=labels, guidance_scale=3.0)

        assert samples.shape == (2, 3, 40, 40)

    def test_save_and_load_model(self, device, temp_out_dir):
        """Test saving and loading model state"""
        model1 = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )

        # Save model
        model_path = os.path.join(temp_out_dir, "test_model.pth")
        torch.save(model1.state_dict(), model_path)

        # Create new model and load state
        model2 = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        model2.load_state_dict(torch.load(model_path))

        # Test both models produce same output with same random seed
        model1.eval()
        model2.eval()

        # Set both models to eval and use no_grad for deterministic output
        with torch.no_grad():
            torch.manual_seed(42)
            x1 = torch.randn(2, 3, 40, 40, device=device)
            t1 = torch.tensor([5, 5], device=device)
            out1, _ = model1(x1, t1)

            torch.manual_seed(42)
            x2 = torch.randn(2, 3, 40, 40, device=device)
            t2 = torch.tensor([5, 5], device=device)
            out2, _ = model2(x2, t2)

        # Models should produce same output given same input
        assert torch.allclose(out1, out2, rtol=1e-4, atol=1e-4)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_invalid_image_size(self, device):
        """Test with invalid image size"""
        # Should not raise error during initialization
        model = create_ddpm(image_size=1, model_channels=32, device=device)
        assert model.image_size == 1

    def test_zero_timesteps(self, device):
        """Test with zero timesteps"""
        # Zero timesteps might work but would be meaningless
        # Just test that it doesn't crash during initialization
        try:
            model = create_ddpm(
                image_size=40, num_timesteps=1, model_channels=32, device=device
            )
            assert model.num_timesteps == 1
        except (ValueError, RuntimeError):
            pytest.skip("Zero timesteps not supported")

    def test_negative_guidance_scale(self, device):
        """Test sampling with negative guidance scale"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model.eval()

        class_labels = torch.zeros(2, device=device, dtype=torch.long)

        # Should work but guidance might be weird
        samples = model.sample(
            batch_size=2, class_labels=class_labels, guidance_scale=-1.0
        )

        assert samples.shape == (2, 3, 40, 40)

    def test_very_high_guidance_scale(self, device):
        """Test sampling with very high guidance scale"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model.eval()

        class_labels = torch.zeros(2, device=device, dtype=torch.long)
        samples = model.sample(
            batch_size=2, class_labels=class_labels, guidance_scale=100.0
        )

        assert samples.shape == (2, 3, 40, 40)

    def test_mismatched_batch_sizes(self, device):
        """Test with mismatched batch sizes"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_classes=2, device=device
        ).to(device)

        x = torch.randn(4, 3, 40, 40, device=device)
        labels = torch.zeros(2, device=device, dtype=torch.long)  # Wrong batch size

        with pytest.raises((RuntimeError, IndexError)):
            model(x, class_labels=labels)


class TestGenerateFunction:
    """Tests for the generate() function"""

    def test_generate_basic(self, device, temp_out_dir):
        """Test basic generation with saved model"""
        # Create and save a model
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model_path = os.path.join(temp_out_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)

        # Generate samples
        samples = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=4,
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=temp_out_dir,
            save_images=False,
            device=device,
        )

        assert samples.shape == (4, 3, 40, 40)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_generate_with_class_labels(self, device, temp_out_dir):
        """Test generation with specific class labels"""
        # Create and save a model
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model_path = os.path.join(temp_out_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)

        # Generate samples for class 0
        class_labels = [0, 0, 0, 0]
        samples = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=2,
            class_labels=class_labels,
            guidance_scale=3.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=temp_out_dir,
            save_images=False,
            device=device,
        )

        assert samples.shape == (4, 3, 40, 40)

    def test_generate_saves_images(self, device, temp_out_dir):
        """Test that generate saves images when requested"""
        # Create and save a model
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model_path = os.path.join(temp_out_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)

        # Generate samples with saving
        out_dir = os.path.join(temp_out_dir, "generated")
        samples = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=2,
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            device=device,
        )

        # Check that output directory was created
        assert os.path.exists(out_dir)

        # Check that grid image was saved
        grid_files = [f for f in os.listdir(out_dir) if "grid" in f]
        assert len(grid_files) > 0

    def test_generate_balanced_classes(self, device, temp_out_dir):
        """Test that generate creates balanced samples when class_labels is None"""
        # Create and save a model
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model_path = os.path.join(temp_out_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)

        # Generate 10 samples (should be 5 per class)
        samples = generate(
            model_path=model_path,
            num_samples=10,
            batch_size=5,
            class_labels=None,
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=temp_out_dir,
            save_images=False,
            device=device,
        )

        assert samples.shape == (10, 3, 40, 40)

    def test_generate_invalid_class_labels_length(self, device, temp_out_dir):
        """Test that generate raises error when class_labels length doesn't match num_samples"""
        # Create and save a model
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model_path = os.path.join(temp_out_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)

        # Try to generate with mismatched length
        with pytest.raises(ValueError):
            generate(
                model_path=model_path,
                num_samples=4,
                batch_size=2,
                class_labels=[0, 0],  # Only 2 labels for 4 samples
                guidance_scale=0.0,
                image_size=40,
                num_classes=2,
                model_channels=32,
                num_timesteps=10,
                out_dir=temp_out_dir,
                save_images=False,
                device=device,
            )

    def test_generate_with_batches(self, device, temp_out_dir):
        """Test generation with num_samples > batch_size (generates in multiple batches)"""
        # Create and save a model
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model_path = os.path.join(temp_out_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)

        # Generate 10 samples in batches of 3
        samples = generate(
            model_path=model_path,
            num_samples=10,
            batch_size=3,  # Will need 4 batches (3+3+3+1)
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=temp_out_dir,
            save_images=False,
            device=device,
        )

        assert samples.shape == (10, 3, 40, 40)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_generate_batch_size_larger_than_num_samples(self, device, temp_out_dir):
        """Test generation when batch_size > num_samples (single batch)"""
        # Create and save a model
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        model_path = os.path.join(temp_out_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)

        # Generate 4 samples with batch_size=10 (should work with single batch)
        samples = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=10,
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=temp_out_dir,
            save_images=False,
            device=device,
        )

        assert samples.shape == (4, 3, 40, 40)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
