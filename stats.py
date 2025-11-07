import os
import shutil

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import umap.umap_ as umap
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inception_v3 import InceptionV3FeatureExtractor
from resnet import ResNetFeatureExtractor
from wrn28_cifar10 import WRN28Cifar10FeatureExtractor


def _make_dataloader(data_path, transform):
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=16, shuffle=False)


def _extract_features(feature_extractor, dataloader):
    features_list = []
    class_name_list = []
    image_paths_list = []
    total_batches = len(dataloader)

    # Get all image paths from dataset
    all_paths = [path for path, _ in dataloader.dataset.samples]

    batch_size = dataloader.batch_size
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

            # Get image paths for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + len(images), len(all_paths))
            batch_paths = all_paths[start_idx:end_idx]
            image_paths_list.extend(batch_paths)
        print("\n  - Completed.")
    return np.concatenate(features_list, axis=0), class_name_list, image_paths_list


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


def filter_samples_by_domain_gap(
    features: np.ndarray,
    classes,
    real_class: str = "1.Abnormal",
    synth_class: str = "2.Synthesized_Abnormal",
    threshold: float = 0.5,
    keep_ratio=None,
    random_state: int = 0,
):
    """
    Filter synthetic samples based on domain gap. Keep only synthetic samples
    that are hard to distinguish from real samples.

    Parameters:
    -----------
    features : np.ndarray
        Feature vectors for all samples
    classes : list or array
        Class labels for all samples
    real_class : str
        Label for real samples
    synth_class : str
        Label for synthetic samples
    threshold : float
        Probability threshold (0-1). Synthetic samples with predicted probability
        below this threshold are kept. Lower = stricter filtering.
    keep_ratio : float, optional
        If provided, keep this ratio of synthetic samples with lowest domain gap.
        Overrides threshold parameter.
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    filtered_features : np.ndarray
        Features with filtered synthetic samples
    filtered_classes : list
        Classes with filtered synthetic samples
    kept_synth_indices : np.ndarray
        Original indices of kept synthetic samples
    """
    classes = np.array(classes)
    mask = np.isin(classes, [real_class, synth_class])
    X = features[mask]
    y = (classes[mask] == synth_class).astype(int)  # 1: synthetic, 0: real
    original_indices = np.where(mask)[0]

    if X.shape[0] == 0 or y.sum() == 0:
        print("[Filter] Not enough samples to filter.")
        return features, classes, np.array([])

    # Train domain classifier on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=random_state
    )
    clf.fit(X_scaled, y)

    # Get prediction probabilities
    probs = clf.predict_proba(X_scaled)[:, 1]  # Probability of being synthetic

    # Identify synthetic samples
    synth_mask = y == 1
    synth_indices_in_subset = np.where(synth_mask)[0]
    synth_probs = probs[synth_mask]

    # Filter based on threshold or keep_ratio
    if keep_ratio is not None:
        n_keep = int(len(synth_probs) * keep_ratio)
        threshold_idx = np.argsort(synth_probs)[:n_keep]
        kept_synth_mask = np.zeros(len(synth_probs), dtype=bool)
        kept_synth_mask[threshold_idx] = True
        effective_threshold = np.max(synth_probs[threshold_idx]) if n_keep > 0 else 0.0
    else:
        kept_synth_mask = synth_probs < threshold
        effective_threshold = threshold

    n_total_synth = len(synth_probs)
    n_kept_synth = kept_synth_mask.sum()

    # Map back to original indices
    kept_synth_indices_in_subset = synth_indices_in_subset[kept_synth_mask]
    kept_synth_original_indices = original_indices[kept_synth_indices_in_subset]

    # Keep all real samples and filtered synthetic samples
    real_mask_full = ~np.isin(np.arange(len(features)), original_indices[synth_mask])
    synth_mask_full = np.zeros(len(features), dtype=bool)
    synth_mask_full[kept_synth_original_indices] = True

    final_mask = real_mask_full | synth_mask_full
    filtered_features = features[final_mask]
    filtered_classes = classes[final_mask]

    mode = "keep_ratio" if keep_ratio is not None else "threshold"
    print(
        f"[Filter] Synthetic samples: {n_total_synth} -> {n_kept_synth} "
        f"({n_kept_synth/n_total_synth*100:.1f}% kept, mode={mode}, effective_threshold={effective_threshold:.3f})"
    )

    return filtered_features, list(filtered_classes), kept_synth_original_indices


def save_filtered_images(
    image_paths: list,
    classes,
    kept_synth_indices: np.ndarray,
    synth_class: str,
    output_dir: str,
):
    """
    Save filtered synthetic images to disk.

    Parameters:
    -----------
    image_paths : list
        List of all image paths
    classes : list or array
        Class labels for all samples
    kept_synth_indices : np.ndarray
        Original indices of kept synthetic samples
    synth_class : str
        Label for synthetic samples
    output_dir : str
        Output directory to save filtered images
    """
    os.makedirs(output_dir, exist_ok=True)

    classes = np.array(classes)

    # Get all synthetic sample indices
    synth_mask = classes == synth_class
    synth_indices = np.where(synth_mask)[0]

    print(
        f"\n[Save] Copying {len(kept_synth_indices)} filtered images to {output_dir}..."
    )

    copied_count = 0
    for idx in kept_synth_indices:
        if idx < len(image_paths):
            src_path = image_paths[idx]
            if os.path.exists(src_path):
                # Keep original filename
                filename = os.path.basename(src_path)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                if copied_count % 100 == 0:
                    print(
                        f"  - Copied {copied_count}/{len(kept_synth_indices)} images...",
                        end="\r",
                    )

    print(f"\n[Save] Successfully copied {copied_count} images to {output_dir}")
    return copied_count


def compute_domain_classifier_auc(
    features: np.ndarray,
    classes,
    real_class: str = "1.Abnormal",
    synth_class: str = "2.Synthesized_Abnormal",
    out_dir: str = "./out/stats",
    balance_mode: str = "none",  # "none" or "downsample"
    repeats: int = 1,
    test_size: float = 0.3,
    random_state: int = 0,
    filtered: bool = False,
):
    """
    Train a simple logistic regression to separate real vs synthetic within the Abnormal class.
    Returns ROC AUC and PR AUC on a held-out split and saves a short report.
    """
    classes = np.array(classes)
    mask = np.isin(classes, [real_class, synth_class])
    X = features[mask]
    y = (classes[mask] == synth_class).astype(int)  # 1: synthetic, 0: real

    if X.shape[0] == 0 or y.sum() == 0:
        print(
            "[Domain] Not enough synthetic abnormal samples to run domain classifier."
        )
        return None, None

    rng = np.random.default_rng(random_state)
    roc_list = []
    pr_list = []

    def run_once(_X, _y):
        scaler = StandardScaler()
        Xn = scaler.fit_transform(_X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xn, _y, test_size=test_size, random_state=random_state, stratify=_y
        )
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]
        roc = roc_auc_score(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        pr_auc = auc(rc, pr)
        return roc, pr_auc

    if balance_mode == "downsample":
        real_idx = np.where(y == 0)[0]
        synth_idx = np.where(y == 1)[0]
        n = min(len(real_idx), len(synth_idx))
        if n < 10:
            print(
                "[Domain] Too few samples per class after downsampling; falling back to none."
            )
            balance_mode = "none"
        else:
            for _ in range(max(1, repeats)):
                sel_real = rng.choice(real_idx, n, replace=False)
                sel_synth = rng.choice(synth_idx, n, replace=False)
                sel = np.concatenate([sel_real, sel_synth])
                roc, pr_auc = run_once(X[sel], y[sel])
                roc_list.append(roc)
                pr_list.append(pr_auc)

    if balance_mode == "none":
        roc, pr_auc = run_once(X, y)
        roc_list.append(roc)
        pr_list.append(pr_auc)

    roc_mean, roc_std = float(np.mean(roc_list)), float(np.std(roc_list))
    pr_mean, pr_std = float(np.mean(pr_list)), float(np.std(pr_list))

    # Save a brief report
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "domain_gap_abnormal_real_vs_synth.txt")
    if filtered:
        out_path = os.path.join(
            out_dir, "domain_gap_abnormal_real_vs_synth_filtered.txt"
        )
    with open(out_path, "w") as f:
        f.write("Domain classifier (Abnormal: real vs synthetic)\n")
        f.write(
            f"Samples (original): total={len(y)}, real={int((y==0).sum())}, synth={int((y==1).sum())}\n"
        )
        f.write(f"Balance mode: {balance_mode}, repeats={repeats}\n")
        f.write(f"ROC AUC: {roc_mean:.4f} ± {roc_std:.4f}\n")
        f.write(f"PR AUC: {pr_mean:.4f} ± {pr_std:.4f}\n")
    print(
        f"[Domain] ROC AUC={roc_mean:.3f}±{roc_std:.3f}, PR AUC={pr_mean:.3f}±{pr_std:.3f} — saved to {out_path}"
    )
    return roc_mean, pr_mean


if __name__ == "__main__":
    RESNET50 = "resnet50"
    INCEPTIONV3 = "inceptionv3"
    WRN28_CIFAR10 = "wrn28_cifar10"
    MODEL_LIST = [RESNET50, INCEPTIONV3, WRN28_CIFAR10]
    MODEL = MODEL_LIST[1]
    IMG_SIZE_ORIGINAL = 40
    IMG_SIZE_RESNET = 224
    IMG_SIZE_INCEPTION = 299
    IMG_SIZE_WRN28 = 40
    COLORS = ["blue", "red", "green"]
    ALPHAS = [0.3, 1.0, 0.3]
    FINETUNED_FLAG = False
    FINETUNED_MODEL_PATH = "./out/train/inception_v3_trained.pth"
    LOAD_FLAG = False
    GRAPH_FLAG = False
    STATS_FLAG = False
    DOMAIN_GAP_FLAG = True
    REAL_CLASS = "1.Abnormal"
    FAKE_CLASS_LIST = ["0.Normal", "2.Synthesized_Abnormal"]
    FAKE_CLASS = FAKE_CLASS_LIST[0]
    NUM_OF_AVERAGE = 10
    SYNTH_CLASS = "2.Synthesized_Abnormal"
    OUT_DIR = "./out/stats"
    os.makedirs(OUT_DIR, exist_ok=True)

    if not LOAD_FLAG:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if MODEL == RESNET50:
            feature_extractor = ResNetFeatureExtractor(
                model_dir="./models/", resnet_variant="resnet50"
            )
        elif MODEL == INCEPTIONV3:
            feature_extractor = InceptionV3FeatureExtractor(model_dir="./models/")
        elif MODEL == WRN28_CIFAR10:
            feature_extractor = WRN28Cifar10FeatureExtractor(model_dir="./models/")
        else:
            raise ValueError(f"Unknown model: {MODEL}")

        if FINETUNED_FLAG:
            print("\nLoading fine-tuned model...")
            if os.path.exists(FINETUNED_MODEL_PATH):
                print(f"  - Loading fine-tuned weights from {FINETUNED_MODEL_PATH}")
                feature_extractor.load_state_dict(
                    torch.load(FINETUNED_MODEL_PATH, map_location=device), strict=False
                )
            else:
                raise FileNotFoundError(
                    f"Fine-tuned model not found at {FINETUNED_MODEL_PATH}"
                )

        feature_extractor.to(device)
        feature_extractor.eval()

        if MODEL == RESNET50:
            img_size = IMG_SIZE_RESNET
            # ImageNet
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif MODEL == INCEPTIONV3:
            img_size = IMG_SIZE_INCEPTION
            # ImageNet
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif MODEL == WRN28_CIFAR10:
            img_size = IMG_SIZE_WRN28
            # CIFAR-10
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]

        else:
            raise ValueError(f"Unknown model: {MODEL}")

        print(f"Model: {MODEL}, Image Size: {img_size}")

        data_path = "./data/stats"
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL)
                ),  # make the spacial frequency equal
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        loader = _make_dataloader(data_path, transform)
        unique_classes = loader.dataset.classes

        print("\nExtract features...")
        features, classes, image_paths = _extract_features(feature_extractor, loader)

        print("\nExtracted features:")
        print("  - features.shape:", features.shape)
        print("  - Num of samples:", len(classes))
        print("  - unique classes:", unique_classes)

        print("\nSaving feature extractor...")
        np.save(f"{OUT_DIR}/classes.npy", classes)
        print(f"  - Saved classes to {OUT_DIR}/classes.npy")
        np.save(f"{OUT_DIR}/features.npy", features)
        print(f"  - Saved features to {OUT_DIR}/features.npy")
        np.save(f"{OUT_DIR}/image_paths.npy", image_paths)
        print(f"  - Saved image paths to {OUT_DIR}/image_paths.npy")
    else:
        print("\nLoading feature extractor...")
        classes = np.load(f"{OUT_DIR}/classes.npy", allow_pickle=True)
        features = np.load(f"{OUT_DIR}/features.npy", allow_pickle=True)
        image_paths = np.load(f"{OUT_DIR}/image_paths.npy", allow_pickle=True).tolist()
        print(f"  - Loaded classes from {OUT_DIR}/classes.npy")
        print(f"  - Loaded features from {OUT_DIR}/features.npy")
        print(f"  - Loaded image paths from {OUT_DIR}/image_paths.npy")
        unique_classes = np.unique(classes)

        print("\nLoaded features:")
        print("  - features.shape:", features.shape)
        print("  - Num of samples:", len(classes))
        print("  - unique classes:", unique_classes)

    if GRAPH_FLAG:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        print("\nProcessing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(X_scaled)
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
        umap_results = reducer.fit_transform(X_scaled)
        _save_graph(
            umap_results,
            classes,
            unique_classes,
            "UMAP: Data Distribution",
            f"{OUT_DIR}/umap_plot.png",
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

    if DOMAIN_GAP_FLAG and (SYNTH_CLASS in unique_classes):
        print("\nRunning domain classifier (Abnormal: real vs synthetic)...")
        compute_domain_classifier_auc(
            features,
            classes,
            real_class="1.Abnormal",
            synth_class=SYNTH_CLASS,
            out_dir=OUT_DIR,
            balance_mode="downsample",
            repeats=5,
            test_size=0.3,
            random_state=0,
        )

        # Filter synthetic samples to keep only those with small domain gap
        print("\nFiltering synthetic samples by domain gap...")
        filtered_features, filtered_classes, kept_indices = (
            filter_samples_by_domain_gap(
                features,
                classes,
                real_class="1.Abnormal",
                synth_class=SYNTH_CLASS,
                keep_ratio=0.03,  # Keep 50% of synthetic samples with smallest domain gap
                # threshold=0.8,  # Alternative: use threshold instead of keep_ratio
                random_state=0,
            )
        )

        # Save filtered features and classes
        np.save(f"{OUT_DIR}/filtered_features.npy", filtered_features)
        np.save(f"{OUT_DIR}/filtered_classes.npy", filtered_classes)
        print(f"  - Saved filtered features to {OUT_DIR}/filtered_features.npy")
        print(f"  - Saved filtered classes to {OUT_DIR}/filtered_classes.npy")

        # Save filtered images to disk
        filtered_images_dir = os.path.join(OUT_DIR, "filtered_images")
        save_filtered_images(
            image_paths,
            classes,
            kept_indices,
            synth_class=SYNTH_CLASS,
            output_dir=filtered_images_dir,
        )

        # Re-run domain classifier on filtered data
        print("\nRunning domain classifier on filtered data...")
        compute_domain_classifier_auc(
            filtered_features,
            filtered_classes,
            real_class="1.Abnormal",
            synth_class=SYNTH_CLASS,
            out_dir=OUT_DIR,
            balance_mode="downsample",
            repeats=5,
            test_size=0.3,
            random_state=0,
            filtered=True,
        )
