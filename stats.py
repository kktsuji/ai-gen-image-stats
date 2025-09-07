import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
import os


def _make_dataloader(data_path, transform):
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def _extract_features(feature_extractor, dataloader):
    features_list = []
    class_name_list = []
    with torch.no_grad():
        for batch in dataloader:
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
):
    plt.figure(figsize=(10, 8))
    num_classes = len(unique_classes)
    if num_classes <= 10:
        colormap = cm.tab10
    else:
        colormap = cm.tab20
    colors = [colormap(i / max(num_classes - 1, 1)) for i in range(num_classes)]

    for i, class_name in enumerate(unique_classes):
        mask = np.array([c == class_name for c in classes])
        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            color=colors[i],
            label=class_name,
            alpha=0.6,
        )
    plt.legend()
    plt.title(title)
    plt.savefig(output_path)


def _calculate_wasserstein_distances(feature0, feature1):
    distances = []
    for dim in range(feature0.shape[1]):
        dist = wasserstein_distance(feature0[:, dim], feature1[:, dim])
        distances.append(dist)
    return distances


def _calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./out", exist_ok=True)  # Create output directory
    if not os.path.exists("./models/resnet50.pth"):
        feature_extractor = models.resnet50(pretrained=True)
        torch.save(feature_extractor, "./models/resnet50.pth")
    else:
        feature_extractor = torch.load("./models/resnet50.pth")
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    data_path = "./data/stats"
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = _make_dataloader(data_path, transform)
    unique_classes = loader.dataset.classes

    print("Extract features")
    features, classes = _extract_features(feature_extractor, loader)
    print("Extracted features:")
    print("  - features.shape:", features.shape)
    print("  - len of classes:", len(classes))
    print("  - unique classes:", unique_classes)

    print("Start t-SNE")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    _save_graph(
        tsne_results,
        classes,
        unique_classes,
        "t-SNE: Data Distribution",
        "./out/tsne_plot.png",
    )

    print("Start UMAP")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_results = reducer.fit_transform(features)
    _save_graph(
        umap_results,
        classes,
        unique_classes,
        "UMAP: Data Distribution",
        "./out/umap_plot.png",
    )

    # mean_distance_original_generated = np.mean(
    #     _calculate_wasserstein_distances(original_features, generated_features)
    # )
    # mean_distance_original_normal = np.mean(
    #     _calculate_wasserstein_distances(original_features, original_normal_features)
    # )
    # mean_distance_normal_generated = np.mean(
    #     _calculate_wasserstein_distances(original_normal_features, generated_features)
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
