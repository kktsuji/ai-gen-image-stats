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
    LOAD_FLAG = True
    os.makedirs("./out", exist_ok=True)

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

        print("\n\nExtracted features:")
        print("  - features.shape:", features.shape)
        print("  - Num of samples:", len(classes))
        print("  - unique classes:", unique_classes)

        print("\nSaving feature extractor...")
        np.save("./out/classes.npy", classes)
        print("  - Saved classes to ./out/classes.npy")
        np.save("./out/features.npy", features)
        print("  - Saved features to ./out/features.npy")
    else:
        print("\nLoading feature extractor...")
        classes = np.load("./out/classes.npy", allow_pickle=True)
        features = np.load("./out/features.npy", allow_pickle=True)
        print("  - Loaded classes from ./out/classes.npy")
        print("  - Loaded features from ./out/features.npy")

        print("\nLoaded features:")
        print("  - features.shape:", features.shape)
        print("  - Num of samples:", len(classes))
        print("  - unique classes:", unique_classes)

    print("\nProcessing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    _save_graph(
        tsne_results,
        classes,
        unique_classes,
        "t-SNE: Data Distribution",
        "./out/tsne_plot.png",
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
        "./out/umap_plot.png",
        xlim=(3, 15),
        colors=COLORS,
        alphas=ALPHAS,
    )

    # mean_distance_original_generated = np.mean(
    #     calculate_wasserstein_distances(original_features, generated_features)
    # )
    # mean_distance_original_normal = np.mean(
    #     calculate_wasserstein_distances(original_features, original_normal_features)
    # )
    # mean_distance_normal_generated = np.mean(
    #     calculate_wasserstein_distances(original_normal_features, generated_features)
    # )
    # print("Average Wasserstein Distance:")
    # print(f"  - Original vs Generated: {mean_distance_original_generated}")
    # print(f"  - Original vs Original Normal: {mean_distance_original_normal}")
    # print(f"  - Original Normal vs Generated: {mean_distance_normal_generated}")

    # fid_score_original_generated = _calculate_fid(original_features, generated_features)
    # fid_score_original_normal = _calculate_fid(
    #     original_features, original_normal_features
    # )
    # fid_score_normal_generated = _calculate_fid(
    #     original_normal_features, generated_features
    # )
    # print("FID Score:")
    # print(f"  - Original vs Generated: {fid_score_original_generated}")
    # print(f"  - Original vs Original Normal: {fid_score_original_normal}")
    # print(f"  - Original Normal vs Generated: {fid_score_normal_generated}")
    # print(f"  - Original vs Original Normal: {fid_score_original_normal}")
    # print(f"  - Original Normal vs Generated: {fid_score_normal_generated}")
