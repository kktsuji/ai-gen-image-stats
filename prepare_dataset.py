import os
import random
import shutil
from glob import glob

if __name__ == "__main__":
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
