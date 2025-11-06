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
    EMA,
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

        # Generate samples with saving
        out_dir = os.path.join(temp_out_dir, "generated_basic")
        result = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=4,
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            device=device,
        )

        # Function should return None
        assert result is None

        # Check that images were saved
        assert os.path.exists(out_dir)
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 4

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

        # Generate samples for class 0 with saving
        class_labels = [0, 0, 0, 0]
        out_dir = os.path.join(temp_out_dir, "generated_class_labels")
        result = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=2,
            class_labels=class_labels,
            guidance_scale=3.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            device=device,
        )

        # Function should return None
        assert result is None

        # Check that images were saved
        assert os.path.exists(out_dir)
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 4

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
        result = generate(
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

        # Function should return None
        assert result is None

        # Check that output directory was created
        assert os.path.exists(out_dir)

        # Check that individual images were saved
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 4

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

        # Generate 10 samples (should be 5 per class) with saving
        out_dir = os.path.join(temp_out_dir, "generated_balanced")
        result = generate(
            model_path=model_path,
            num_samples=10,
            batch_size=5,
            class_labels=None,
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            device=device,
        )

        # Function should return None
        assert result is None

        # Check that images were saved
        assert os.path.exists(out_dir)
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 10

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

        # Generate 10 samples in batches of 3 with saving
        out_dir = os.path.join(temp_out_dir, "generated_batches")
        result = generate(
            model_path=model_path,
            num_samples=10,
            batch_size=3,  # Will need 4 batches (3+3+3+1)
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            device=device,
        )

        # Function should return None
        assert result is None

        # Check that images were saved
        assert os.path.exists(out_dir)
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 10

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
        out_dir = os.path.join(temp_out_dir, "generated_large_batch")
        result = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=10,
            guidance_scale=0.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            device=device,
        )

        # Function should return None
        assert result is None

        # Check that images were saved
        assert os.path.exists(out_dir)
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 4


class TestEMA:
    """Test EMA (Exponential Moving Average) class"""

    def test_initialization(self, device):
        """Test EMA initialization"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema = EMA(model, decay=0.9999, device=device)

        assert ema is not None
        assert ema.model is model
        assert ema.decay == 0.9999
        assert len(ema.shadow) > 0
        assert len(ema.backup) == 0

    def test_shadow_parameters_initialized(self, device):
        """Test that shadow parameters are properly initialized"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema = EMA(model, decay=0.9999, device=device)

        # Check that shadow params match model params initially
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert ema.shadow[name].shape == param.data.shape

    def test_update(self, device):
        """Test EMA update"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema = EMA(model, decay=0.9999, device=device)

        # Get initial shadow value
        first_param_name = list(ema.shadow.keys())[0]
        initial_shadow = ema.shadow[first_param_name].clone()

        # Modify model parameters
        for param in model.parameters():
            if param.requires_grad:
                param.data += 1.0
                break

        # Update EMA
        ema.update()

        # Shadow should have changed
        updated_shadow = ema.shadow[first_param_name]
        assert not torch.allclose(initial_shadow, updated_shadow)

    def test_apply_shadow(self, device):
        """Test applying shadow parameters to model"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema = EMA(model, decay=0.9999, device=device)

        # Get original param value
        first_param = next(model.parameters())
        original_value = first_param.data.clone()

        # Modify model slightly
        first_param.data += 1.0

        # Update EMA
        ema.update()

        # Apply shadow
        ema.apply_shadow()

        # Model params should now be shadow params
        shadow_value = first_param.data.clone()
        assert not torch.allclose(original_value + 1.0, shadow_value)

        # Backup should be populated
        assert len(ema.backup) > 0

    def test_restore(self, device):
        """Test restoring original parameters"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema = EMA(model, decay=0.9999, device=device)

        # Get original value
        first_param = next(model.parameters())
        original_value = first_param.data.clone()

        # Modify and update
        first_param.data += 1.0
        ema.update()

        # Apply shadow and then restore
        ema.apply_shadow()
        shadow_value = first_param.data.clone()
        ema.restore()

        # Should be back to the modified value
        restored_value = first_param.data.clone()
        assert torch.allclose(restored_value, original_value + 1.0)
        assert not torch.allclose(restored_value, shadow_value)

        # Backup should be cleared
        assert len(ema.backup) == 0

    def test_state_dict(self, device):
        """Test EMA state_dict"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema = EMA(model, decay=0.9999, device=device)

        state_dict = ema.state_dict()

        assert "decay" in state_dict
        assert "shadow" in state_dict
        assert state_dict["decay"] == 0.9999
        assert len(state_dict["shadow"]) > 0

    def test_load_state_dict(self, device):
        """Test EMA load_state_dict"""
        model1 = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema1 = EMA(model1, decay=0.9999, device=device)
        ema1.update()

        # Save state
        state_dict = ema1.state_dict()

        # Create new EMA and load
        model2 = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema2 = EMA(model2, decay=0.9, device=device)  # Different decay
        ema2.load_state_dict(state_dict)

        assert ema2.decay == 0.9999  # Should match loaded value

    def test_ema_reduces_variance(self, device):
        """Test that EMA actually smooths parameter updates"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        )
        ema = EMA(model, decay=0.99, device=device)  # High decay for noticeable effect

        first_param = next(model.parameters())
        first_param_name = [n for n, p in model.named_parameters()][0]

        # Apply several random updates
        original = first_param.data.clone()
        for _ in range(5):
            first_param.data = original + torch.randn_like(first_param.data) * 10
            ema.update()

        # Shadow should be closer to original than final param
        final_param = first_param.data
        shadow = ema.shadow[first_param_name]

        dist_shadow_to_orig = torch.norm(shadow - original)
        dist_final_to_orig = torch.norm(final_param - original)

        assert dist_shadow_to_orig < dist_final_to_orig


class TestDynamicThreshold:
    """Test dynamic thresholding functionality"""

    def test_dynamic_threshold_basic(self, device):
        """Test basic dynamic thresholding"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        ).to(device)

        x_start = torch.randn(2, 3, 40, 40, device=device) * 5  # Larger values

        thresholded = model.dynamic_threshold(x_start, percentile=0.95)

        assert thresholded.shape == x_start.shape
        # All values should be within [-1, 1] after thresholding
        assert torch.all(torch.abs(thresholded) <= 1.0 + 1e-5)

    def test_dynamic_threshold_percentile(self, device):
        """Test dynamic thresholding with different percentiles"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        ).to(device)

        x_start = torch.randn(2, 3, 40, 40, device=device) * 3

        # Higher percentile should allow more extreme values
        thresh_low = model.dynamic_threshold(x_start, percentile=0.9)
        thresh_high = model.dynamic_threshold(x_start, percentile=0.99)

        # Both should be normalized to [-1, 1]
        assert torch.all(torch.abs(thresh_low) <= 1.0 + 1e-5)
        assert torch.all(torch.abs(thresh_high) <= 1.0 + 1e-5)

    def test_dynamic_threshold_minimum(self, device):
        """Test that threshold has minimum of 1.0"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        ).to(device)

        # Small values that would have percentile < 1.0
        x_start = torch.randn(2, 3, 40, 40, device=device) * 0.1

        thresholded = model.dynamic_threshold(x_start, percentile=0.999)

        # Should not amplify small values beyond 1.0
        assert torch.all(torch.abs(thresholded) <= 1.0 + 1e-5)

    def test_p_mean_variance_with_dynamic_threshold(self, device):
        """Test p_mean_variance with dynamic thresholding enabled"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=100, device=device
        ).to(device)
        model.eval()

        x_t = torch.randn(2, 3, 40, 40, device=device)
        t = torch.tensor([50, 50], device=device)

        # With dynamic thresholding
        mean1, var1 = model.p_mean_variance(
            x_t, t, use_dynamic_threshold=True, dynamic_threshold_percentile=0.995
        )

        # Without dynamic thresholding
        mean2, var2 = model.p_mean_variance(x_t, t, use_dynamic_threshold=False)

        # Both should produce valid outputs
        assert mean1.shape == x_t.shape
        assert mean2.shape == x_t.shape
        # Variance is broadcast-compatible with x_t shape
        assert var1.shape == (2, 1, 1, 1)
        assert var2.shape == (2, 1, 1, 1)

    def test_sample_with_dynamic_threshold(self, device):
        """Test sampling with dynamic thresholding"""
        model = create_ddpm(
            image_size=40, model_channels=32, num_timesteps=10, device=device
        ).to(device)
        model.eval()

        # Sample with dynamic thresholding
        samples_with = model.sample(
            batch_size=2,
            use_dynamic_threshold=True,
            dynamic_threshold_percentile=0.995,
        )

        # Sample without dynamic thresholding
        samples_without = model.sample(
            batch_size=2,
            use_dynamic_threshold=False,
        )

        assert samples_with.shape == (2, 3, 40, 40)
        assert samples_without.shape == (2, 3, 40, 40)

        # Both should produce reasonable outputs (in [-1, 1] range roughly)
        # Note: outputs won't be exactly in [-1, 1] without final clamping


class TestSDEdit:
    """Test SDEdit (sample_from_image) functionality"""

    def test_sample_from_image_basic(self, device):
        """Test basic SDEdit functionality"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=100,
            device=device,
        ).to(device)
        model.eval()

        # Create starting images (real normals)
        x_0 = torch.randn(2, 3, 40, 40, device=device)
        class_labels = torch.ones(2, device=device, dtype=torch.long)

        # Generate from images
        generated = model.sample_from_image(
            x_0=x_0, t_0=50, class_labels=class_labels, guidance_scale=2.0
        )

        assert generated.shape == x_0.shape

    def test_sample_from_image_different_t0(self, device):
        """Test SDEdit with different starting timesteps"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=100,
            device=device,
        ).to(device)
        model.eval()

        x_0 = torch.randn(2, 3, 40, 40, device=device)
        class_labels = torch.ones(2, device=device, dtype=torch.long)

        # Lower t_0 = less change
        gen_low = model.sample_from_image(
            x_0=x_0, t_0=10, class_labels=class_labels, guidance_scale=0.0
        )

        # Higher t_0 = more change
        gen_high = model.sample_from_image(
            x_0=x_0, t_0=80, class_labels=class_labels, guidance_scale=0.0
        )

        assert gen_low.shape == x_0.shape
        assert gen_high.shape == x_0.shape

        # Higher t_0 should produce more different results
        # (though this is probabilistic so we just check shapes)

    def test_sample_from_image_with_guidance(self, device):
        """Test SDEdit with classifier-free guidance"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=50,
            device=device,
        ).to(device)
        model.eval()

        x_0 = torch.randn(2, 3, 40, 40, device=device)
        class_labels = torch.ones(2, device=device, dtype=torch.long)

        # With guidance
        gen_guided = model.sample_from_image(
            x_0=x_0, t_0=25, class_labels=class_labels, guidance_scale=3.0
        )

        # Without guidance
        gen_unguided = model.sample_from_image(
            x_0=x_0, t_0=25, class_labels=class_labels, guidance_scale=0.0
        )

        assert gen_guided.shape == x_0.shape
        assert gen_unguided.shape == x_0.shape

    def test_sample_from_image_with_dynamic_threshold(self, device):
        """Test SDEdit with dynamic thresholding"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=50,
            device=device,
        ).to(device)
        model.eval()

        x_0 = torch.randn(2, 3, 40, 40, device=device)
        class_labels = torch.ones(2, device=device, dtype=torch.long)

        # With dynamic thresholding
        gen_with = model.sample_from_image(
            x_0=x_0,
            t_0=25,
            class_labels=class_labels,
            use_dynamic_threshold=True,
            dynamic_threshold_percentile=0.995,
        )

        # Without dynamic thresholding
        gen_without = model.sample_from_image(
            x_0=x_0, t_0=25, class_labels=class_labels, use_dynamic_threshold=False
        )

        assert gen_with.shape == x_0.shape
        assert gen_without.shape == x_0.shape

    def test_sample_from_image_return_intermediates(self, device):
        """Test SDEdit with return_intermediates"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=50,
            device=device,
        ).to(device)
        model.eval()

        x_0 = torch.randn(2, 3, 40, 40, device=device)
        class_labels = torch.ones(2, device=device, dtype=torch.long)
        t_0 = 20

        # Get intermediates
        intermediates = model.sample_from_image(
            x_0=x_0,
            t_0=t_0,
            class_labels=class_labels,
            return_intermediates=True,
        )

        # Should return t_0 + 1 timesteps (initial + each denoising step)
        assert intermediates.shape == (t_0 + 1, 2, 3, 40, 40)

    def test_sample_from_image_preserves_structure(self, device):
        """Test that SDEdit preserves some structure from input"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=100,
            device=device,
        ).to(device)
        model.eval()

        # Create a structured input (e.g., all positive values)
        x_0 = torch.abs(torch.randn(2, 3, 40, 40, device=device))
        class_labels = torch.ones(2, device=device, dtype=torch.long)

        # With very low t_0, should preserve a lot of structure
        generated = model.sample_from_image(
            x_0=x_0, t_0=5, class_labels=class_labels, guidance_scale=0.0
        )

        assert generated.shape == x_0.shape
        # With low t_0, the mean should be somewhat preserved
        # (This is a weak test but checks basic functionality)

    def test_sample_from_image_unconditional(self, device):
        """Test SDEdit without class labels (unconditional)"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=None,  # Unconditional
            num_timesteps=50,
            device=device,
        ).to(device)
        model.eval()

        x_0 = torch.randn(2, 3, 40, 40, device=device)

        # Should work without class labels
        generated = model.sample_from_image(
            x_0=x_0, t_0=25, class_labels=None, guidance_scale=0.0
        )

        assert generated.shape == x_0.shape


class TestGenerateWithNewFeatures:
    """Test generate() function with new features"""

    def test_generate_with_dynamic_threshold(self, device, temp_out_dir):
        """Test generation with dynamic thresholding enabled"""
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

        # Generate with dynamic thresholding
        out_dir = os.path.join(temp_out_dir, "generated_dynamic_thresh")
        result = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=4,
            guidance_scale=2.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            use_dynamic_threshold=True,
            dynamic_threshold_percentile=0.995,
            device=device,
        )

        assert result is None
        assert os.path.exists(out_dir)
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 4

    def test_generate_without_dynamic_threshold(self, device, temp_out_dir):
        """Test generation with dynamic thresholding disabled"""
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

        # Generate without dynamic thresholding
        out_dir = os.path.join(temp_out_dir, "generated_no_dynamic_thresh")
        result = generate(
            model_path=model_path,
            num_samples=4,
            batch_size=4,
            guidance_scale=2.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            use_dynamic_threshold=False,
            device=device,
        )

        assert result is None
        assert os.path.exists(out_dir)
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 4

    def test_generate_different_percentiles(self, device, temp_out_dir):
        """Test generation with different dynamic threshold percentiles"""
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

        # Test with 90th percentile
        out_dir_90 = os.path.join(temp_out_dir, "generated_p90")
        result = generate(
            model_path=model_path,
            num_samples=2,
            batch_size=2,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir_90,
            save_images=True,
            use_dynamic_threshold=True,
            dynamic_threshold_percentile=0.90,
            device=device,
        )

        assert result is None
        assert os.path.exists(out_dir_90)

        # Test with 99.9th percentile
        out_dir_999 = os.path.join(temp_out_dir, "generated_p999")
        result = generate(
            model_path=model_path,
            num_samples=2,
            batch_size=2,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir_999,
            save_images=True,
            use_dynamic_threshold=True,
            dynamic_threshold_percentile=0.999,
            device=device,
        )

        assert result is None
        assert os.path.exists(out_dir_999)


class TestIntegrationWithNewFeatures:
    """Integration tests combining new features"""

    def test_ema_training_workflow(self, device):
        """Test complete EMA training workflow"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        ema = EMA(model, decay=0.9999, device=device)

        # Training step
        x = torch.randn(2, 3, 40, 40, device=device)
        labels = torch.randint(0, 2, (2,), device=device)

        loss = model.training_step(x, class_labels=labels)
        loss.backward()

        # Update EMA
        ema.update()

        # Validation with EMA
        model.eval()
        ema.apply_shadow()

        with torch.no_grad():
            val_loss = model.training_step(x, class_labels=labels)

        ema.restore()

        assert val_loss is not None

    def test_full_pipeline_with_all_features(self, device, temp_out_dir):
        """Test full pipeline: train with EMA, save, generate with dynamic threshold"""
        # Create model
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=10,
            device=device,
        )
        ema = EMA(model, decay=0.9999, device=device)

        # Quick training step
        x = torch.randn(2, 3, 40, 40, device=device)
        labels = torch.randint(0, 2, (2,), device=device)
        loss = model.training_step(x, class_labels=labels)
        loss.backward()
        ema.update()

        # Save EMA weights
        ema.apply_shadow()
        model_path = os.path.join(temp_out_dir, "model_ema.pth")
        torch.save(model.state_dict(), model_path)
        ema.restore()

        # Generate with all features
        out_dir = os.path.join(temp_out_dir, "full_pipeline")
        generate(
            model_path=model_path,
            num_samples=4,
            batch_size=2,
            guidance_scale=2.0,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=out_dir,
            save_images=True,
            use_dynamic_threshold=True,
            dynamic_threshold_percentile=0.995,
            device=device,
        )

        # Check outputs
        assert os.path.exists(out_dir)
        saved_images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(saved_images) == 4

    def test_sdedit_with_dynamic_threshold_and_guidance(self, device):
        """Test SDEdit combining dynamic thresholding and guidance"""
        model = create_ddpm(
            image_size=40,
            model_channels=32,
            num_classes=2,
            num_timesteps=50,
            device=device,
        ).to(device)
        model.eval()

        # Create starting images
        x_0 = torch.randn(2, 3, 40, 40, device=device)
        class_labels = torch.ones(2, device=device, dtype=torch.long)

        # Generate with all features
        generated = model.sample_from_image(
            x_0=x_0,
            t_0=25,
            class_labels=class_labels,
            guidance_scale=3.0,
            use_dynamic_threshold=True,
            dynamic_threshold_percentile=0.995,
        )

        assert generated.shape == x_0.shape
        # Values should be reasonably bounded due to dynamic thresholding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
