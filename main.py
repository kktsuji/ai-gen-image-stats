import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()


def post_requests(text, webhook_url):
    import json

    import requests

    message = {"text": text}
    requests.post(webhook_url, data=json.dumps(message), timeout=5.0)


if __name__ == "__main__":
    # General settings
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    NUM_WORKERS = os.getenv("NUM_WORKERS", "4")

    # Enable or disable specific functionalities
    ENABLE_DATASET_PREPARATION = (
        os.getenv("ENABLE_DATASET_PREPARATION", "false").lower() == "true"
    )
    ENABLE_DDPM_TRAINING = os.getenv("ENABLE_DDPM_TRAINING", "true").lower() == "true"
    ENABLE_DDPM_SAMPLING = os.getenv("ENABLE_DDPM_SAMPLING", "true").lower() == "true"
    ENABLE_TRAINING = os.getenv("ENABLE_TRAINING", "true").lower() == "true"

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

    os.makedirs(train_normal_dir, exist_ok=True)
    os.makedirs(train_abnormal_dir, exist_ok=True)
    os.makedirs(val_normal_dir, exist_ok=True)
    os.makedirs(val_abnormal_dir, exist_ok=True)

    print("\n" + "=" * 60)
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

    # Common DDPM configuration
    DDPM_NUM_CLASSES = os.getenv("DDPM_NUM_CLASSES", "2")
    DDPM_IMG_SIZE = os.getenv("DDPM_IMG_SIZE", "40")
    DDPM_NUM_TIMESTEPS = os.getenv("DDPM_NUM_TIMESTEPS", "1000")
    DDPM_MODEL_CHANNELS = os.getenv("DDPM_MODEL_CHANNELS", "64")
    DDPM_BETA_SCHEDULE = os.getenv("DDPM_BETA_SCHEDULE", "cosine")
    DDPM_OUTPUT_DIR = os.getenv("DDPM_OUTPUT_DIR", "ddpm")

    print("\n" + "=" * 60)
    if ENABLE_DDPM_TRAINING:
        # DDPM training configuration
        DDPM_TRAIN_EPOCHS = os.getenv("DDPM_TRAIN_EPOCHS", "200")
        DDPM_TRAIN_BATCH_SIZE = os.getenv("DDPM_TRAIN_BATCH_SIZE", "8")
        DDPM_TRAIN_LEARNING_RATE = os.getenv("DDPM_TRAIN_LEARNING_RATE", "0.00005")
        DDPM_TRAIN_BETA_START = os.getenv("DDPM_TRAIN_BETA_START", "0.0001")
        DDPM_TRAIN_BETA_END = os.getenv("DDPM_TRAIN_BETA_END", "0.02")
        DDPM_TRAIN_CLASS_DROPOUT_PROB = os.getenv(
            "DDPM_TRAIN_CLASS_DROPOUT_PROB", "0.3"
        )
        DDPM_TRAIN_USE_WEIGHTED_SAMPLING = (
            os.getenv("DDPM_TRAIN_USE_WEIGHTED_SAMPLING", "true").lower() == "true"
        )
        train_data_path = os.path.join(work_dir, "data/train")
        val_data_path = os.path.join(work_dir, "data/val")
        output_dir_ddpm = os.path.join(work_dir, DDPM_OUTPUT_DIR)

        os.makedirs(output_dir_ddpm, exist_ok=True)

        if DOCKER_COMMAND_PREFIX:
            command = shlex.split(docker_cmd) + ["ddpm_train.py"]
        else:
            command = [sys.executable, "ddpm_train.py"]

        # Add common arguments
        command.extend(
            [
                "--epochs",
                DDPM_TRAIN_EPOCHS,
                "--batch-size",
                DDPM_TRAIN_BATCH_SIZE,
                "--learning-rate",
                DDPM_TRAIN_LEARNING_RATE,
                "--num-classes",
                DDPM_NUM_CLASSES,
                "--img-size",
                DDPM_IMG_SIZE,
                "--num-timesteps",
                DDPM_NUM_TIMESTEPS,
                "--model-channels",
                DDPM_MODEL_CHANNELS,
                "--beta-schedule",
                DDPM_BETA_SCHEDULE,
                "--beta-start",
                DDPM_TRAIN_BETA_START,
                "--beta-end",
                DDPM_TRAIN_BETA_END,
                "--class-dropout-prob",
                DDPM_TRAIN_CLASS_DROPOUT_PROB,
                "--out-dir",
                output_dir_ddpm,
                "--train-data-path",
                train_data_path,
                "--val-data-path",
                val_data_path,
                "--seed",
                SEED,
                "--num-workers",
                NUM_WORKERS,
            ]
        )

        # Add optional flag
        if DDPM_TRAIN_USE_WEIGHTED_SAMPLING:
            command.append("--use-weighted-sampling")

        result = subprocess.run(command, check=True, shell=False)
    else:
        print("\nDDPM training is disabled. Skipping.")

    print("\n" + "=" * 60)
    if ENABLE_DDPM_SAMPLING:
        # DDPM sampling configuration
        DDPM_GEN_NUM_SAMPLES = os.getenv("DDPM_GEN_NUM_SAMPLES", "20")
        DDPM_GEN_BATCH_SIZE = os.getenv("DDPM_GEN_BATCH_SIZE", "4")
        DDPM_GEN_CLASS_LABEL = os.getenv(
            "DDPM_GEN_CLASS_LABEL", "1"
        )  # e.g., "0" or "1"
        DDPM_GEN_MODEL_NAME = os.getenv("DDPM_GEN_MODEL_NAME", "ddpm_model_ema.pth")
        DDPM_GEN_GUIDANCE_SCALE = os.getenv("DDPM_GEN_GUIDANCE_SCALE", "2.0")
        DDPM_GEN_USE_DYNAMIC_THRESHOLD = (
            os.getenv("DDPM_GEN_USE_DYNAMIC_THRESHOLD", "true").lower() == "true"
        )
        model_path = os.path.join(work_dir, DDPM_OUTPUT_DIR, DDPM_GEN_MODEL_NAME)
        out_dir = os.path.join(work_dir, DDPM_OUTPUT_DIR, "samples")

        # Update train and val paths to include generated samples
        train_abnormal_dir_with_gen = os.path.join(out_dir, "train", abnormal_dir_name)
        val_abnormal_dir_with_gen = os.path.join(out_dir, "val", abnormal_dir_name)

        os.makedirs(out_dir, exist_ok=True)

        if DOCKER_COMMAND_PREFIX:
            command = shlex.split(docker_cmd) + ["ddpm_gen.py"]
        else:
            command = [sys.executable, "ddpm_gen.py"]

        # Add arguments
        command.extend(
            [
                "--num-samples",
                DDPM_GEN_NUM_SAMPLES,
                "--batch-size",
                DDPM_GEN_BATCH_SIZE,
                "--class-label",
                DDPM_GEN_CLASS_LABEL,
                "--guidance-scale",
                DDPM_GEN_GUIDANCE_SCALE,
                "--img-size",
                DDPM_IMG_SIZE,
                "--num-classes",
                DDPM_NUM_CLASSES,
                "--model-channels",
                DDPM_MODEL_CHANNELS,
                "--num-timesteps",
                DDPM_NUM_TIMESTEPS,
                "--beta-schedule",
                DDPM_BETA_SCHEDULE,
                "--model-path",
                model_path,
                "--out-dir",
                out_dir,
                "--seed",
                SEED,
            ]
        )

        if DDPM_GEN_USE_DYNAMIC_THRESHOLD:
            command.append("--use-dynamic-threshold")

        result = subprocess.run(command, check=True, shell=False)
    else:
        print("\nDDPM sampling is disabled. Skipping.")
        train_abnormal_dir_with_gen = train_abnormal_dir
        val_abnormal_dir_with_gen = val_abnormal_dir

    print("\n" + "=" * 60)
    if ENABLE_TRAINING:
        # Training configuration
        TRAIN_MODEL_TYPE = os.getenv("TRAIN_MODEL_TYPE", "inception_v3")
        TRAIN_EPOCHS = os.getenv("TRAIN_EPOCHS", "10")
        TRAIN_BATCH_SIZE = os.getenv("TRAIN_BATCH_SIZE", "16")
        TRAIN_LEARNING_RATE = os.getenv("TRAIN_LEARNING_RATE", "0.00005")
        TRAIN_NUM_CLASSES = os.getenv("TRAIN_NUM_CLASSES", "2")
        TRAIN_IMG_SIZE_ORIGINAL = os.getenv("TRAIN_IMG_SIZE_ORIGINAL", "40")
        TRAIN_UNDER_SAMPLING = (
            os.getenv("TRAIN_UNDER_SAMPLING", "false").lower() == "true"
        )
        TRAIN_USE_CLASS_WEIGHTS = (
            os.getenv("TRAIN_USE_CLASS_WEIGHTS", "true").lower() == "true"
        )
        TRAIN_USE_WEIGHTED_SAMPLING = (
            os.getenv("TRAIN_USE_WEIGHTED_SAMPLING", "false").lower() == "true"
        )
        TRAIN_SEEDS = os.getenv("TRAIN_SEEDS", "0").split(",")
        TRAIN_USE_SYNTH = os.getenv("TRAIN_USE_SYNTH", "true").lower() == "true"
        TRAIN_OUTPUT_DIR = (
            os.getenv("TRAIN_OUTPUT_DIR", "train")
            if not TRAIN_USE_SYNTH
            else os.getenv("TRAIN_OUTPUT_DIR", "train") + "-synth"
        )

        train_data_path = os.path.join(work_dir, "data/train")
        val_data_path = os.path.join(work_dir, "data/val")
        out_dir = os.path.join(work_dir, TRAIN_OUTPUT_DIR)
        os.makedirs(out_dir, exist_ok=True)

        if DOCKER_COMMAND_PREFIX:
            command = shlex.split(docker_cmd) + ["train.py"]
        else:
            command = [sys.executable, "train.py"]

        for train_seed in TRAIN_SEEDS:
            print(f"\n--- Training with seed {train_seed} ---\n")
            output_dir_train = os.path.join(out_dir, f"seed_{train_seed}")
            os.makedirs(output_dir_train, exist_ok=True)

            # Add arguments
            command.extend(
                [
                    "--epochs",
                    TRAIN_EPOCHS,
                    "--batch-size",
                    TRAIN_BATCH_SIZE,
                    "--learning-rate",
                    TRAIN_LEARNING_RATE,
                    "--num-classes",
                    TRAIN_NUM_CLASSES,
                    "--img-size-original",
                    TRAIN_IMG_SIZE_ORIGINAL,
                    "--model-type",
                    TRAIN_MODEL_TYPE,
                    "--seed",
                    train_seed,
                    "--train-data-path",
                    train_data_path,
                    "--val-data-path",
                    val_data_path,
                    "--out-dir",
                    output_dir_train,
                    "--num-workers",
                    NUM_WORKERS,
                ]
            )

            # Add optional flags
            if TRAIN_UNDER_SAMPLING:
                command.append("--under-sampling")
            if TRAIN_USE_CLASS_WEIGHTS:
                command.append("--use-class-weights")
            if TRAIN_USE_WEIGHTED_SAMPLING:
                command.append("--use-weighted-sampling")

            result = subprocess.run(command, check=True, shell=False)
    else:
        print("\nTraining is disabled. Skipping.")

    if WEBHOOK_URL:
        post_requests(
            "AI-generated Image Statistics pipeline has completed.", WEBHOOK_URL
        )
