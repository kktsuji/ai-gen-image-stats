import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import umap.umap_ as umap
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inception_v3 import InceptionV3FeatureExtractor
from resnet import ResNetFeatureExtractor


def _make_dataloader(data_path, transform):
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def _extract_features(feature_extractor, dataloader):
    features_list = []
    class_name_list = []
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(
                    f"  - Processing batch {batch_idx}/{total_batches}",
                    end="\r",
                )
            images, labels = batch
            images = images.to(device)
            feat = feature_extractor(images)
            features_list.append(feat.cpu().numpy())
            class_names = [dataloader.dataset.classes[label] for label in labels]
            class_name_list.extend(class_names)
        print("\n  - Completed.")
    return np.concatenate(features_list, axis=0), class_name_list


def _save_graph(
    data,
    classes,
    unique_classes,
    title: str,
    output_path: str,
    xlim=None,
    ylim=None,
    colors=None,
    alphas=None,
):
    plt.figure(figsize=(10, 8))
    num_classes = len(unique_classes)
    if num_classes <= 10:
        colormap = cm.tab10
    else:
        colormap = cm.tab20
    if colors is None:
        colors = [colormap(i / max(num_classes - 1, 1)) for i in range(num_classes)]
        colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

    for i, class_name in enumerate(unique_classes):
        mask = np.array([c == class_name for c in classes])
        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            color=colors[i],
            label=class_name,
            alpha=alphas[i] if alphas is not None else 0.7,
        )
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.title(title)
    plt.savefig(output_path)


def calculate_wasserstein_distances(feature0, feature1):
    distances = []
    for dim in range(feature0.shape[1]):
        dist = wasserstein_distance(feature0[:, dim], feature1[:, dim])
        distances.append(dist)
    return distances


def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def calculate_precision_recall(real_features, fake_features, k=5):
    """
    Calculate precision and recall metrics for generative models.

    Precision: measures if fake samples fall within the real data manifold
    Recall: measures if real samples are covered by the fake distribution

    Based on "Improved Precision and Recall Metric for Assessing Generative Models"
    (Kynkäänniemi et al., NeurIPS 2019)
    """
    from sklearn.neighbors import NearestNeighbors

    # Compute k-th nearest neighbor distances within each distribution
    real_nbrs = NearestNeighbors(n_neighbors=k + 1).fit(real_features)
    real_distances, _ = real_nbrs.kneighbors(real_features)
    # Take the k-th neighbor (index k, since index 0 is the point itself)
    real_kth_distances = real_distances[:, k]

    fake_nbrs = NearestNeighbors(n_neighbors=k + 1).fit(fake_features)
    fake_distances, _ = fake_nbrs.kneighbors(fake_features)
    fake_kth_distances = fake_distances[:, k]

    # Precision: For each fake sample, find nearest real sample
    fake_to_real_nbrs = NearestNeighbors(n_neighbors=1).fit(real_features)
    fake_to_real_distances, fake_to_real_indices = fake_to_real_nbrs.kneighbors(
        fake_features
    )
    fake_to_real_distances = fake_to_real_distances.flatten()
    fake_to_real_indices = fake_to_real_indices.flatten()

    # Precision: fake is within real manifold if distance to nearest real
    # is <= that real point's k-th nearest neighbor distance
    precision = np.mean(
        fake_to_real_distances <= real_kth_distances[fake_to_real_indices]
    )

    # Recall: For each real sample, find nearest fake sample
    real_to_fake_nbrs = NearestNeighbors(n_neighbors=1).fit(fake_features)
    real_to_fake_distances, real_to_fake_indices = real_to_fake_nbrs.kneighbors(
        real_features
    )
    real_to_fake_distances = real_to_fake_distances.flatten()
    real_to_fake_indices = real_to_fake_indices.flatten()

    # Recall: real is covered by fake manifold if distance to nearest fake
    # is <= that fake point's k-th nearest neighbor distance
    recall = np.mean(real_to_fake_distances <= fake_kth_distances[real_to_fake_indices])

    return precision, recall


def split_features_by_class(features, classes, unique_classes):
    class_feature_dict = {class_name: [] for class_name in unique_classes}
    for feat, class_name in zip(features, classes):
        class_feature_dict[class_name].append(feat)
    for class_name in unique_classes:
        class_feature_dict[class_name] = np.array(class_feature_dict[class_name])
    return class_feature_dict


def under_sample_features(features, num_samples):
    if features.shape[0] <= num_samples:
        raise ValueError(
            "Number of samples to under-sample must be less than the number of features."
        )
    indices = np.random.choice(features.shape[0], num_samples, replace=False)
    return features[indices]


if __name__ == "__main__":
    RESNET50 = "resnet50"
    INCEPTIONV3 = "inceptionv3"
    MODEL_LIST = [RESNET50, INCEPTIONV3]
    MODEL = MODEL_LIST[1]
    IMG_SIZE_ORIGINAL = 40
    IMG_SIZE_RESNET = 224
    IMG_SIZE_INCEPTION = 299
    COLORS = ["blue", "red", "green"]
    ALPHAS = [0.3, 1.0, 0.3]
    LOAD_FLAG = False
    GRAPH_FLAG = True
    STATS_FLAG = True
    REAL_CLASS = "1.Abnormal"
    FAKE_CLASS_LIST = ["0.Normal", "2.Synthesized_Abnormal"]
    FAKE_CLASS = FAKE_CLASS_LIST[0]
    NUM_OF_AVERAGE = 10
    OUT_DIR = "./out/stats"
    os.makedirs(OUT_DIR, exist_ok=True)

    if not LOAD_FLAG:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        feature_extractor = (
            ResNetFeatureExtractor(model_dir="./models/", resnet_variant="resnet50")
            if MODEL == RESNET50
            else InceptionV3FeatureExtractor(model_dir="./models/")
        )
        feature_extractor.to(device)
        feature_extractor.eval()
        img_size = IMG_SIZE_RESNET if MODEL == RESNET50 else IMG_SIZE_INCEPTION
        print(f"Model: {MODEL}, Image Size: {img_size}")

        data_path = "./data/stats"
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL)
                ),  # make the spacial frequency equal
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        loader = _make_dataloader(data_path, transform)
        unique_classes = loader.dataset.classes

        print("\nExtract features...")
        features, classes = _extract_features(feature_extractor, loader)

        print("\nExtracted features:")
        print("  - features.shape:", features.shape)
        print("  - Num of samples:", len(classes))
        print("  - unique classes:", unique_classes)

        print("\nSaving feature extractor...")
        np.save(f"{OUT_DIR}/classes.npy", classes)
        print(f"  - Saved classes to {OUT_DIR}/classes.npy")
        np.save(f"{OUT_DIR}/features.npy", features)
        print(f"  - Saved features to {OUT_DIR}/features.npy")
    else:
        print("\nLoading feature extractor...")
        classes = np.load(f"{OUT_DIR}/classes.npy", allow_pickle=True)
        features = np.load(f"{OUT_DIR}/features.npy", allow_pickle=True)
        print(f"  - Loaded classes from {OUT_DIR}/classes.npy")
        print(f"  - Loaded features from {OUT_DIR}/features.npy")
        unique_classes = np.unique(classes)

        print("\nLoaded features:")
        print("  - features.shape:", features.shape)
        print("  - Num of samples:", len(classes))
        print("  - unique classes:", unique_classes)

    if GRAPH_FLAG:
        print("\nProcessing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(features)
        _save_graph(
            tsne_results,
            classes,
            unique_classes,
            "t-SNE: Data Distribution",
            f"{OUT_DIR}/tsne_plot.png",
            colors=COLORS,
            alphas=ALPHAS,
        )

        print("\nProcessing UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_results = reducer.fit_transform(features)
        _save_graph(
            umap_results,
            classes,
            unique_classes,
            "UMAP: Data Distribution",
            f"{OUT_DIR}/umap_plot.png",
            xlim=(3, 15),
            colors=COLORS,
            alphas=ALPHAS,
        )

    if STATS_FLAG:
        class_feature_dict = split_features_by_class(features, classes, unique_classes)
        num_real_class = class_feature_dict[REAL_CLASS].shape[0]
        real_class = class_feature_dict[REAL_CLASS]
        print(f"\nNumber of each class:")
        for class_name in unique_classes:
            num_samples = class_feature_dict[class_name].shape[0]
            print(f"  - {class_name}: {num_samples} samples")

        fid_list = []
        precision_list = []
        recall_list = []
        wasserstein_list = []

        print(f"\nCalculating statistics over {NUM_OF_AVERAGE} runs...")
        for i in range(NUM_OF_AVERAGE):
            print(f"  - Run {i+1}/{NUM_OF_AVERAGE}", end="\r")
            fake_class = under_sample_features(
                class_feature_dict[FAKE_CLASS], num_real_class
            )
            fid = calculate_fid(real_class, fake_class)
            precision, recall = calculate_precision_recall(real_class, fake_class, k=5)
            wasserstein_distances = calculate_wasserstein_distances(
                real_class, fake_class
            )
            fid_list.append(fid)
            precision_list.append(precision)
            recall_list.append(recall)
            wasserstein_list.append(wasserstein_distances)
        print("\n  - Completed.")

        print(f"\nAverage results {REAL_CLASS} vs. {FAKE_CLASS}:")
        print(f"  - FID Score (lower is better): {np.mean(fid_list)}")
        print(f"  - Precision (higher is better): {np.mean(precision_list)}")
        print(f"  - Recall (higher is better): {np.mean(recall_list)}")
        print(
            f"  - Wasserstein Distance (lower is better): {np.mean(wasserstein_list)}"
        )

        # Save results to text file
        output_file = f"{OUT_DIR}/stats_{REAL_CLASS}_vs_{FAKE_CLASS}.txt"
        with open(output_file, "w") as f:
            f.write(f"Statistics: {REAL_CLASS} vs. {FAKE_CLASS}\n")
            f.write(f"Number of runs: {NUM_OF_AVERAGE}\n")
            f.write(f"Number of samples: {num_real_class}\n\n")
            f.write(
                f"FID Score (lower is better): {np.mean(fid_list):.4f} ± {np.std(fid_list):.4f}\n"
            )
            f.write(
                f"Precision (higher is better): {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}\n"
            )
            f.write(
                f"Recall (higher is better): {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}\n"
            )
            f.write(
                f"Wasserstein Distance (lower is better): {np.mean(wasserstein_list):.4f} ± {np.std(wasserstein_list):.4f}\n"
            )
        print(f"\nResults saved to {output_file}")
