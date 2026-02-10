"""
Script to create mock images for testing.
Creates a simple ImageFolder-style dataset structure.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

# Base directory for mock data
MOCK_DATA_DIR = Path(__file__).parent

# Create train and val directories with two classes each
datasets = {
    "train": {
        "0.Normal": 5,  # 5 normal images
        "1.Abnormal": 3,  # 3 abnormal images
    },
    "val": {
        "0.Normal": 2,  # 2 normal images
        "1.Abnormal": 2,  # 2 abnormal images
    },
}


def create_simple_image(color, size=(32, 32)):
    """Create a simple colored image."""
    array = np.full((*size, 3), color, dtype=np.uint8)
    return Image.fromarray(array)


def main():
    """Create all mock images."""
    for dataset_name, classes in datasets.items():
        dataset_dir = MOCK_DATA_DIR / dataset_name

        for class_name, num_images in classes.items():
            class_dir = dataset_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Use different colors for different classes
            if "Normal" in class_name:
                color = (100, 150, 200)  # Blueish
            else:
                color = (200, 100, 100)  # Reddish

            # Create images
            for i in range(num_images):
                img = create_simple_image(color)
                img_path = class_dir / f"image_{i:03d}.png"
                img.save(img_path)
                print(f"Created: {img_path}")

    print("\nMock data structure created successfully!")


if __name__ == "__main__":
    main()
