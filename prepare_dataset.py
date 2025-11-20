import argparse
import os
import random
import shutil
from glob import glob


def split_dataset_into_train_val_by_ratio(
    dir_path: str,
    train_dir: str,
    val_dir: str,
    val_ratio: float,
) -> tuple[int, int]:
    # 1. Glob normal and abnormal images
    images = glob(os.path.join(dir_path, "*.png"))

    # 2. Shuffle them
    random.shuffle(images)

    # 3. Split them into train and val
    val_count = int(len(images) * val_ratio)
    val = images[:val_count]
    train = images[val_count:]

    # 4. Copy images to new folders
    for img in train:
        shutil.copy(img, train_dir)

    for img in val:
        shutil.copy(img, val_dir)

    return len(train), len(val)


def prepare_dataset():
    ORI_NORMAL_DIR = "./data/stats/0.Normal/"
    ORI_ABNORMAL_DIR = "./data/stats/1.Abnormal/"
    ORI_SYNTH_DIR = "./data/stats/2.Synthesized_Abnormal/"
    VAL_RATIO = 0.2
    SYNTH_RATIO = 0.3
    SYNTH_COUNT = 1000
    RANDOM_SEED = 0
    USE_SYNTH_RATIO = "synth-ratio"
    USE_SYNTH_COUNT = "synth-count"
    NOT_USE_SYNTH = "no-synth"
    CHOICE = [USE_SYNTH_RATIO, USE_SYNTH_COUNT, NOT_USE_SYNTH]
    MODE = CHOICE[0]
    USE_REMAINING_NORMAL_VAL = False

    if MODE == USE_SYNTH_RATIO:
        suffix = f"_{USE_SYNTH_RATIO}-{SYNTH_RATIO}"
    elif MODE == USE_SYNTH_COUNT:
        suffix = f"_{USE_SYNTH_COUNT}-{SYNTH_COUNT}"
    else:
        suffix = f"_{NOT_USE_SYNTH}"
    suffix += "_imbalanced-val" if USE_REMAINING_NORMAL_VAL else "_balanced-val"
    suffix += f"_seed{RANDOM_SEED}"
    TRAIN_DIR = f"./data/train{suffix}"
    VAL_DIR = f"./data/val{suffix}"

    os.makedirs(TRAIN_DIR)
    os.makedirs(VAL_DIR)

    random.seed(RANDOM_SEED)

    # Get all abnormal images and split
    abnormal_images = glob(os.path.join(ORI_ABNORMAL_DIR, "*.png"))
    random.shuffle(abnormal_images)

    val_count = int(len(abnormal_images) * VAL_RATIO)
    abnormal_val = abnormal_images[:val_count]
    abnormal_train = abnormal_images[val_count:]

    # Copy abnormal images to train/val
    os.makedirs(os.path.join(TRAIN_DIR, "1.Abnormal"), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, "1.Abnormal"), exist_ok=True)

    for img in abnormal_train:
        shutil.copy(img, os.path.join(TRAIN_DIR, "1.Abnormal"))

    for img in abnormal_val:
        shutil.copy(img, os.path.join(VAL_DIR, "1.Abnormal"))

    if MODE == USE_SYNTH_RATIO:
        # Calculate how many synthesized images to use based on SYNTH_RATIO
        # We want: original_abnormal / (original_abnormal + synthesized) = (1 - SYNTH_RATIO)
        # So: synthesized = original_abnormal * SYNTH_RATIO / (1 - SYNTH_RATIO)
        synth_count = int(len(abnormal_train) * SYNTH_RATIO / (1 - SYNTH_RATIO))

        # Get synthesized abnormal images and copy to train
        synth_images = glob(os.path.join(ORI_SYNTH_DIR, "*.png"))
        random.shuffle(synth_images)
        synth_images = synth_images[:synth_count]

        for img in synth_images:
            shutil.copy(img, os.path.join(TRAIN_DIR, "1.Abnormal"))

        # Calculate total abnormal train count
        total_abnormal_train = len(abnormal_train) + len(synth_images)
    elif MODE == USE_SYNTH_COUNT:
        # Use synthesized images
        synth_images = glob(os.path.join(ORI_SYNTH_DIR, "*.png"))
        random.shuffle(synth_images)
        synth_images = synth_images[:SYNTH_COUNT]

        for img in synth_images:
            shutil.copy(img, os.path.join(TRAIN_DIR, "1.Abnormal"))

        # Calculate total abnormal train count
        total_abnormal_train = len(abnormal_train) + len(synth_images)
    else:  # MODE == NOT_USE_SYNTH
        synth_images = []
        total_abnormal_train = len(abnormal_train)

    # Get normal images and split to match abnormal train count
    normal_images = glob(os.path.join(ORI_NORMAL_DIR, "*.png"))
    random.shuffle(normal_images)

    if len(normal_images) < total_abnormal_train:
        raise ValueError("Not enough normal images to match abnormal training count.")

    if MODE == NOT_USE_SYNTH:
        normal_abnormal_ratio = len(normal_images) / len(abnormal_images)
        normal_train_count = int(total_abnormal_train * normal_abnormal_ratio)
    else:
        normal_train_count = total_abnormal_train
    normal_train = normal_images[:normal_train_count]
    normal_val = (
        normal_images[normal_train_count:]
        if USE_REMAINING_NORMAL_VAL
        else normal_images[normal_train_count : normal_train_count + len(abnormal_val)]
    )

    # Copy normal images to train/val
    os.makedirs(os.path.join(TRAIN_DIR, "0.Normal"), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, "0.Normal"), exist_ok=True)

    for img in normal_train:
        shutil.copy(img, os.path.join(TRAIN_DIR, "0.Normal"))

    for img in normal_val:
        shutil.copy(img, os.path.join(VAL_DIR, "0.Normal"))

    print(f"Abnormal - Train: {len(abnormal_train)}, Val: {len(abnormal_val)}")
    print(f"Synthesized - Train: {len(synth_images)}")
    print(f"Normal - Train: {len(normal_train)}, Val: {len(normal_val)}")
    print(
        f"Total Train - Abnormal: {total_abnormal_train}, Normal: {len(normal_train)}"
    )
    print(f"Total Val - Abnormal: {len(abnormal_val)}, Normal: {len(normal_val)}")


def copy_synthesized_images():
    ORI_ABNORMAL_DIR = "./data/stats/1.Abnormal/"
    ORI_SYNTH_DIR = "./data/stats/2.Synthesized_Abnormal/"
    DEST_DIR = "./data/train_synth-ratio0.3_imbalanced-val_seed0/1.Abnormal/"
    SYNTH_RATIO = 0.3

    seed = 0
    random.seed(seed)

    abnormal_train = len(glob(os.path.join(ORI_ABNORMAL_DIR, "*.png")))

    synth_images = glob(os.path.join(ORI_SYNTH_DIR, "*.png"))
    random.shuffle(synth_images)

    synth_count = int(abnormal_train * SYNTH_RATIO / (1 - SYNTH_RATIO))

    for img in synth_images[:synth_count]:
        shutil.copy(img, os.path.join(DEST_DIR, os.path.basename(img)))

    print(f"Copied {len(synth_images)} synthesized images to {DEST_DIR}")


def split_dataset_into_train_val_by_count(data_dir, val_count):
    all_images = glob(os.path.join(data_dir, "*.png"))
    random.shuffle(all_images)

    val_images = all_images[:val_count]
    train_images = all_images[val_count:]

    return train_images, val_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train and validation sets"
    )
    parser.add_argument(
        "--normal-dir",
        type=str,
        required=True,
        help="Path to directory containing normal images",
    )
    parser.add_argument(
        "--abnormal-dir",
        type=str,
        required=True,
        help="Path to directory containing abnormal images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        required=True,
        help="Validation split ratio (e.g., 0.2 for 20%% validation)",
    )
    parser.add_argument(
        "--train-normal-dir",
        type=str,
        default="out/data/train/normal",
        help="Path to the directory for training normal images (default: out/data/train/normal)",
    )
    parser.add_argument(
        "--train-abnormal-dir",
        type=str,
        default="out/data/train/abnormal",
        help="Path to the directory for training abnormal images (default: out/data/train/abnormal)",
    )
    parser.add_argument(
        "--val-normal-dir",
        type=str,
        default="out/data/val/normal",
        help="Path to the directory for validation normal images (default: out/data/val/normal)",
    )
    parser.add_argument(
        "--val-abnormal-dir",
        type=str,
        default="out/data/val/abnormal",
        help="Path to the directory for validation abnormal images (default: out/data/val/abnormal)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    train_normal, val_normal = split_dataset_into_train_val_by_ratio(
        args.normal_dir, args.train_normal_dir, args.val_normal_dir, args.split_ratio
    )
    train_abnormal, val_abnormal = split_dataset_into_train_val_by_ratio(
        args.abnormal_dir,
        args.train_abnormal_dir,
        args.val_abnormal_dir,
        args.split_ratio,
    )

    print("\nDataset split completed:")
    print(f"  - Seed: {args.seed}")
    print(f"  - Split ratio: {args.split_ratio}")
    print(f"  - Normal - Train: {train_normal}, Val: {val_normal}")
    print(f"  - Abnormal - Train: {train_abnormal}, Val: {val_abnormal}")
    print("  - Output directories:")
    print(f"    - Train Normal:   {args.train_normal_dir}")
    print(f"    - Train Abnormal: {args.train_abnormal_dir}")
    print(f"    - Val Normal:     {args.val_normal_dir}")
    print(f"    - Val Abnormal:   {args.val_abnormal_dir}")

    # prepare_dataset()
    # copy_synthesized_images()
