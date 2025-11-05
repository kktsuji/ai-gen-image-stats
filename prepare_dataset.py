import os
import random
import shutil
from glob import glob

if __name__ == "__main__":
    ORI_NORMAL_DIR = "./data/stats/0.Normal/"
    ORI_ABNORMAL_DIR = "./data/stats/1.Abnormal/"
    ORI_SYNTH_DIR = "./data/stats/2.Synthesized_Abnormal/"
    TRAIN_DIR = "./data/train"
    VAL_DIR = "./data/val"
    VAL_RATIO = 0.2
    SYNTH_RATIO = 0.3
    RANDOM_SEED = 42
    USE_SYNTH = True

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

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

    if USE_SYNTH:
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
    else:
        total_abnormal_train = len(abnormal_train)

    # Get normal images and split to match abnormal train count
    normal_images = glob(os.path.join(ORI_NORMAL_DIR, "*.png"))
    random.shuffle(normal_images)

    normal_train_count = total_abnormal_train
    normal_train = normal_images[:normal_train_count]
    normal_val = normal_images[
        normal_train_count : normal_train_count + len(abnormal_val)
    ]

    # Copy normal images to train/val
    os.makedirs(os.path.join(TRAIN_DIR, "0.Normal"), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, "0.Normal"), exist_ok=True)

    for img in normal_train:
        shutil.copy(img, os.path.join(TRAIN_DIR, "0.Normal"))

    for img in normal_val:
        shutil.copy(img, os.path.join(VAL_DIR, "0.Normal"))

    print(f"Abnormal - Train: {len(abnormal_train)}, Val: {len(abnormal_val)}")
    print(f"Synthesized - Train: {len(synth_images) if USE_SYNTH else 0}")
    print(f"Normal - Train: {len(normal_train)}, Val: {len(normal_val)}")
    print(
        f"Total Train - Abnormal: {total_abnormal_train}, Normal: {len(normal_train)}"
    )
