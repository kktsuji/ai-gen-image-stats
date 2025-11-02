import os
import shutil
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
    print("\n  - Completed.")
    return np.vstack(extracted_features)


if __name__ == "__main__":
    TARGET_DATA_PATH = "./data/stats/1.Abnormal"
    SYNTHESIZED_DATA_PATH = "./data/stats/2.Synthesized_Abnormal"
    IMG_SIZE_ORIGINAL = 40
    IMG_SIZE = 299
    LOAD_FLAG = True
    OUT_DIR = "./out/cleansing"
    os.makedirs(OUT_DIR, exist_ok=True)

    if not LOAD_FLAG:
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

    # Z-score normalization using target (real) set statistics
    target_mean = target_features.mean(axis=0)
    target_std = target_features.std(axis=0)
    target_std[target_std == 0] = 1  # Avoid division by zero

    target_features_normalized = (target_features - target_mean) / target_std
    synthesized_features_normalized = (synthesized_features - target_mean) / target_std

    # Compute leave-one-out kNN distances for real samples to determine threshold
    k = 5
    real_knn_distances = []
    print(f"\nComputing leave-one-out kNN distances for real samples (k={k})...")
    for i in range(target_features_normalized.shape[0]):
        if (i + 1) % 100 == 0:
            print(
                f"  - Processing real sample {i + 1}/{target_features_normalized.shape[0]}...",
                end="\r",
            )
        # Exclude the current sample
        other_real = np.delete(target_features_normalized, i, axis=0)
        distances = np.linalg.norm(other_real - target_features_normalized[i], axis=1)
        knn_dist = np.mean(np.sort(distances)[:k])
        real_knn_distances.append(knn_dist)
    print("\n  - Completed.")

    percentile = 80
    threshold = np.percentile(real_knn_distances, percentile)
    print(f"{percentile}th percentile threshold: {threshold:.4f}")

    # Compute kNN distances for each synthesized sample
    print(f"\nComputing kNN distances for synthesized samples...")
    accepted_indices = []
    synthesized_knn_distances = []

    for i in range(synthesized_features_normalized.shape[0]):
        if (i + 1) % 100 == 0:
            print(
                f"  - Processing synthetic sample {i + 1}/{synthesized_features_normalized.shape[0]}...",
                end="\r",
            )
        distances = np.linalg.norm(
            target_features_normalized - synthesized_features_normalized[i], axis=1
        )
        knn_dist = np.mean(np.sort(distances)[:k])
        synthesized_knn_distances.append(knn_dist)

        if knn_dist <= threshold:
            accepted_indices.append(i)
    print("\n  - Completed.")

    print(f"\nResults:")
    print(f"  - Total synthesized samples: {len(synthesized_knn_distances)}")
    print(
        f"  - Accepted samples: {len(accepted_indices)} ({100 * len(accepted_indices) / len(synthesized_knn_distances):.2f}%)"
    )
    print(
        f"  - Rejected samples: {len(synthesized_knn_distances) - len(accepted_indices)}"
    )

    # Save results
    np.save(
        f"{OUT_DIR}/accepted_indices_k{k}_th{percentile}.npy",
        np.array(accepted_indices),
    )
    np.save(
        f"{OUT_DIR}/synthesized_knn_distances_k{k}_th{percentile}.npy",
        np.array(synthesized_knn_distances),
    )

    # Copy accepted images to output directory
    cleansing_data_dir = f"{OUT_DIR}/cleansing_data_k{k}_th{percentile}"
    os.makedirs(cleansing_data_dir, exist_ok=True)

    print(f"\nCopying accepted images to {cleansing_data_dir}...")
    for i, idx in enumerate(accepted_indices):
        if (i + 1) % 100 == 0:
            print(f"  - Copying image {i + 1}/{len(accepted_indices)}...", end="\r")
        src_path = synthesized_path_list[idx]
        dst_path = os.path.join(cleansing_data_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
    print("\n  - Completed.")
    print(
        f"  - Successfully copied {len(accepted_indices)} images to {cleansing_data_dir}"
    )

    # Copy rejected images to output directory
    rejected_indices = [
        i for i in range(len(synthesized_knn_distances)) if i not in accepted_indices
    ]
    rejected_data_dir = f"{OUT_DIR}/rejected_data_k{k}_th{percentile}"
    os.makedirs(rejected_data_dir, exist_ok=True)

    print(f"\nCopying rejected images to {rejected_data_dir}...")
    for i, idx in enumerate(rejected_indices):
        if (i + 1) % 100 == 0:
            print(f"  - Copying image {i + 1}/{len(rejected_indices)}...", end="\r")
        src_path = synthesized_path_list[idx]
        dst_path = os.path.join(rejected_data_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
    print("\n  - Completed.")
    print(
        f"  - Successfully copied {len(rejected_indices)} images to {rejected_data_dir}"
    )
