import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Enable or disable specific functionalities
    ENABLE_DATASET_PREPARATION = (
        os.getenv("ENABLE_DATASET_PREPARATION", "false").lower() == "true"
    )
    ENABLE_DDPM_TRAINING = os.getenv("ENABLE_DDPM_TRAINING", "true").lower() == "true"

    # Data configuration
    DATA_NORMAL_DIR_PATH = os.getenv("DATA_NORMAL_DIR_PATH")
    DATA_ABNORMAL_DIR_PATH = os.getenv("DATA_ABNORMAL_DIR_PATH")
    SEED = os.getenv("SEED", "0")
    DATA_SPLIT_RATIO = os.getenv("DATA_SPLIT_RATIO", "0.2")
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
                DATA_SPLIT_RATIO,
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

    DOCKER_COMMAND_PREFIX = os.getenv("DOCKER_COMMAND_PREFIX", "")
    # Build command based on whether Docker is used
    if DOCKER_COMMAND_PREFIX:
        # Use Docker command with shell expansion
        # Expand shell variables before passing to subprocess
        import shlex

        # Replace shell variables with actual values
        docker_cmd = DOCKER_COMMAND_PREFIX
        docker_cmd = docker_cmd.replace("$PWD", os.getcwd())
        docker_cmd = docker_cmd.replace("$(id -u)", str(os.getuid()))
        docker_cmd = docker_cmd.replace("$(id -g)", str(os.getgid()))

    if ENABLE_DDPM_TRAINING:
        # DDPM training configuration
        EPOCHS = os.getenv("EPOCHS", "200")
        BATCH_SIZE = os.getenv("BATCH_SIZE", "8")
        LEARNING_RATE = os.getenv("LEARNING_RATE", "0.00005")
        NUM_CLASSES = os.getenv("NUM_CLASSES", "2")
        IMG_SIZE = os.getenv("IMG_SIZE", "40")
        NUM_TIMESTEPS = os.getenv("NUM_TIMESTEPS", "1000")
        MODEL_CHANNELS = os.getenv("MODEL_CHANNELS", "64")
        BETA_SCHEDULE = os.getenv("BETA_SCHEDULE", "cosine")
        BETA_START = os.getenv("BETA_START", "0.0001")
        BETA_END = os.getenv("BETA_END", "0.02")
        CLASS_DROPOUT_PROB = os.getenv("CLASS_DROPOUT_PROB", "0.3")
        USE_WEIGHTED_SAMPLING = (
            os.getenv("USE_WEIGHTED_SAMPLING", "true").lower() == "true"
        )
        OUTPUT_DIR_DDPM = os.getenv("OUTPUT_DIR_DDPM", "ddpm")
        train_data_path = os.path.join(work_dir, "data/train")
        val_data_path = os.path.join(work_dir, "data/val")
        output_dir_ddpm = os.path.join(work_dir, OUTPUT_DIR_DDPM)

        os.makedirs(output_dir_ddpm, exist_ok=True)

        if DOCKER_COMMAND_PREFIX:
            command = shlex.split(docker_cmd) + ["ddpm_train.py"]
        else:
            command = [sys.executable, "ddpm_train.py"]

        # Add common arguments
        command.extend(
            [
                "--epochs",
                EPOCHS,
                "--batch-size",
                BATCH_SIZE,
                "--learning-rate",
                LEARNING_RATE,
                "--num-classes",
                NUM_CLASSES,
                "--img-size",
                IMG_SIZE,
                "--num-timesteps",
                NUM_TIMESTEPS,
                "--model-channels",
                MODEL_CHANNELS,
                "--beta-schedule",
                BETA_SCHEDULE,
                "--beta-start",
                BETA_START,
                "--beta-end",
                BETA_END,
                "--class-dropout-prob",
                CLASS_DROPOUT_PROB,
                "--out-dir",
                output_dir_ddpm,
                "--train-data-path",
                train_data_path,
                "--val-data-path",
                val_data_path,
            ]
        )

        # Add optional flag
        if USE_WEIGHTED_SAMPLING:
            command.append("--use-weighted-sampling")

        result = subprocess.run(command, check=True, shell=False)
