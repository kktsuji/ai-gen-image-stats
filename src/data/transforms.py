"""
Data transformation utilities for image preprocessing and augmentation.

This module provides common transforms and normalization utilities
for training generative models and classifiers.
"""

from typing import List, Optional, Tuple

import torch
from torchvision import transforms

# Common normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_normalization_transform(
    dataset: str = "imagenet",
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Normalize:
    """
    Get normalization transform for a specific dataset.

    Args:
        dataset: Dataset name ('imagenet', 'cifar10', or 'custom')
        mean: Custom mean values (required if dataset='custom')
        std: Custom std values (required if dataset='custom')

    Returns:
        torchvision.transforms.Normalize transform

    Raises:
        ValueError: If dataset is 'custom' but mean/std not provided
    """
    if dataset.lower() == "imagenet":
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    elif dataset.lower() == "cifar10":
        return transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    elif dataset.lower() == "custom":
        if mean is None or std is None:
            raise ValueError("Custom normalization requires mean and std parameters")
        return transforms.Normalize(mean=mean, std=std)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_denormalization_transform(
    dataset: str = "imagenet",
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Normalize:
    """
    Get denormalization transform to reverse normalization.

    Args:
        dataset: Dataset name ('imagenet', 'cifar10', or 'custom')
        mean: Custom mean values (required if dataset='custom')
        std: Custom std values (required if dataset='custom')

    Returns:
        torchvision.transforms.Normalize transform that reverses normalization
    """
    if dataset.lower() == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif dataset.lower() == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset.lower() == "custom":
        if mean is None or std is None:
            raise ValueError("Custom denormalization requires mean and std parameters")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Denormalization: x_original = x_normalized * std + mean
    # Using Normalize(mean, std): output = (input - mean) / std
    # To reverse: use Normalize(-mean/std, 1/std)
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


def get_base_transforms(
    image_size: int = 256,
    crop_size: int = 224,
    resize_mode: str = "resize_crop",
) -> transforms.Compose:
    """
    Get basic transforms for inference/validation.

    Args:
        image_size: Size to resize the image to
        crop_size: Size to center crop (if using resize_crop mode)
        resize_mode: 'resize' (just resize) or 'resize_crop' (resize then center crop)

    Returns:
        Composed transforms
    """
    transform_list = []

    if resize_mode == "resize":
        transform_list.append(transforms.Resize((crop_size, crop_size)))
    elif resize_mode == "resize_crop":
        transform_list.extend(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(crop_size),
            ]
        )
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode}")

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def get_train_transforms(
    image_size: int = 256,
    crop_size: int = 224,
    horizontal_flip: bool = True,
    color_jitter: bool = False,
    rotation_degrees: int = 0,
    normalize: Optional[str] = None,
) -> transforms.Compose:
    """
    Get augmentation transforms for training.

    Args:
        image_size: Size to resize the image to
        crop_size: Size for random crop
        horizontal_flip: Whether to apply random horizontal flip
        color_jitter: Whether to apply color jittering
        rotation_degrees: Max rotation degrees (0 = no rotation)
        normalize: Dataset name for normalization ('imagenet', 'cifar10', or None)

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.RandomCrop(crop_size),
    ]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if rotation_degrees > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )
        )

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(get_normalization_transform(normalize))

    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = 256,
    crop_size: int = 224,
    normalize: Optional[str] = None,
) -> transforms.Compose:
    """
    Get transforms for validation/testing.

    Args:
        image_size: Size to resize the image to
        crop_size: Size to center crop
        normalize: Dataset name for normalization ('imagenet', 'cifar10', or None)

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(get_normalization_transform(normalize))

    return transforms.Compose(transform_list)


def get_diffusion_transforms(
    image_size: int = 64,
    horizontal_flip: bool = True,
) -> transforms.Compose:
    """
    Get transforms for diffusion model training.

    Diffusion models typically use simpler transforms and don't require
    normalization since they model the full data distribution.

    Args:
        image_size: Size to resize images to (typically 64, 128, or 256)
        horizontal_flip: Whether to apply random horizontal flip

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def get_gan_transforms(
    image_size: int = 64,
    horizontal_flip: bool = True,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> transforms.Compose:
    """
    Get transforms for GAN training.

    GANs typically normalize to [-1, 1] range to match tanh activation.

    Args:
        image_size: Size to resize images to
        horizontal_flip: Whether to apply random horizontal flip
        normalize_range: Target range for normalization (default: (-1, 1))

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list.append(transforms.ToTensor())

    # Normalize to target range
    if normalize_range == (-1.0, 1.0):
        # Transform from [0, 1] to [-1, 1]
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

    return transforms.Compose(transform_list)


def denormalize_image(
    image: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    """
    Denormalize a normalized image tensor.

    Args:
        image: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized image tensor
    """
    # Handle both (C, H, W) and (B, C, H, W) shapes
    if image.dim() == 3:
        c, h, w = image.shape
        mean_t = torch.tensor(mean).view(c, 1, 1)
        std_t = torch.tensor(std).view(c, 1, 1)
    elif image.dim() == 4:
        b, c, h, w = image.shape
        mean_t = torch.tensor(mean).view(1, c, 1, 1)
        std_t = torch.tensor(std).view(1, c, 1, 1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")

    return image * std_t + mean_t


def clamp_image(image: torch.Tensor) -> torch.Tensor:
    """
    Clamp image values to [0, 1] range.

    Args:
        image: Image tensor

    Returns:
        Clamped image tensor
    """
    return torch.clamp(image, 0.0, 1.0)
