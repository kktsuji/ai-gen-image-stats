import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    ENABLE_DATASET_PREPARATION = (
        os.getenv("ENABLE_DATASET_PREPARATION", "false").lower() == "true"
    )

    DATA_NORMAL_DIR_PATH = os.getenv("DATA_NORMAL_DIR_PATH")
    DATA_ABNORMAL_DIR_PATH = os.getenv("DATA_ABNORMAL_DIR_PATH")
    SEED = os.getenv("SEED", "0")
    DATA_SPLIT_RATIO = float(os.getenv("DATA_SPLIT_RATIO", "0.2"))
    OUTPUT_DIR_BASE = os.getenv("OUTPUT_DIR_BASE", "out/")

    work_dir = os.path.join(
        OUTPUT_DIR_BASE, f"split-ratio{DATA_SPLIT_RATIO}_seed{SEED}"
    )
    normal_dir_name = os.path.basename(DATA_NORMAL_DIR_PATH.rstrip("/"))
    abnormal_dir_name = os.path.basename(DATA_ABNORMAL_DIR_PATH.rstrip("/"))
    train_normal_dir = os.path.join(work_dir, "data/train", normal_dir_name)
    train_abnormal_dir = os.path.join(work_dir, "data/train", abnormal_dir_name)
    val_normal_dir = os.path.join(work_dir, "data/val", normal_dir_name)
    val_abnormal_dir = os.path.join(work_dir, "data/val", abnormal_dir_name)

    if ENABLE_DATASET_PREPARATION:
        result = subprocess.run(
            [
                sys.executable,
                "prepare_dataset.py",
                "--normal-dir",
                DATA_NORMAL_DIR_PATH,
                "--abnormal-dir",
                DATA_ABNORMAL_DIR_PATH,
                "--seed",
                SEED,
                "--split-ratio",
                str(DATA_SPLIT_RATIO),
                "--train-normal-dir",
                train_normal_dir,
                "--train-abnormal-dir",
                train_abnormal_dir,
                "--val-normal-dir",
                val_normal_dir,
                "--val-abnormal-dir",
                val_abnormal_dir,
            ],
            check=True,
        )
    else:
        print("\nDataset preparation is disabled. Skipping.")
