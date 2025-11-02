import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from inception_v3 import InceptionV3FeatureExtractor


def _extract_features(model, path_list, transform, device):
    extracted_features = []
    print(f"\nProcessing {len(path_list)} images...")
    for i, data_path in enumerate(path_list):
        if (i + 1) % 100 == 0:
            print(f"  - Processing image {i + 1}/{len(path_list)}...", end="\r")
        img = Image.open(data_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(img_tensor)

        extracted_features.append(feature.cpu().numpy())
    print()
    return np.vstack(extracted_features)


if __name__ == "__main__":
    TARGET_DATA_PATH = "./data/stats/1.Abnormal"
    SYNTHESIZED_DATA_PATH = "./data/stats/2.Synthesized_Abnormal"
    IMG_SIZE_ORIGINAL = 40
    IMG_SIZE = 299
    LOAD_FLAG = False
    OUT_DIR = "./out/cleansing"
    os.makedirs(OUT_DIR, exist_ok=True)

    if LOAD_FLAG:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        feature_extractor = InceptionV3FeatureExtractor(model_dir="./models/")
        feature_extractor.to(device)
        feature_extractor.eval()
        print("Model initialized successfully.")

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL)
                ),  # make the spacial frequency equal
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        target_path_list = glob(f"{TARGET_DATA_PATH}/*.png")
        target_features = _extract_features(
            feature_extractor, target_path_list, transform, device
        )
        target_path_list = np.array(target_path_list)
        np.save(
            f"{OUT_DIR}/target_features.npy",
            target_features,
        )
        np.save(
            f"{OUT_DIR}/target_paths.npy",
            target_path_list,
        )

        synthesized_path_list = glob(f"{SYNTHESIZED_DATA_PATH}/*.png")
        synthesized_features = _extract_features(
            feature_extractor, synthesized_path_list, transform, device
        )
        synthesized_path_list = np.array(synthesized_path_list)
        np.save(
            f"{OUT_DIR}/synthesized_features.npy",
            synthesized_features,
        )
        np.save(
            f"{OUT_DIR}/synthesized_paths.npy",
            synthesized_path_list,
        )
    else:
        target_features = np.load(f"{OUT_DIR}/target_features.npy", allow_pickle=True)
        target_path_list = np.load(
            f"{OUT_DIR}/target_paths.npy",
            allow_pickle=True,
        )
        print("Loaded target features and path list from disk.")
        synthesized_features = np.load(
            f"{OUT_DIR}/synthesized_features.npy", allow_pickle=True
        )
        synthesized_path_list = np.load(
            f"{OUT_DIR}/synthesized_paths.npy",
            allow_pickle=True,
        )
        print("Loaded synthesized features and path list from disk.")

    print(f"\nExtracted features:")
    print(f"  - Target features shape: {target_features.shape}")
    print(f"  - Synthesized features shape: {synthesized_features.shape}")
