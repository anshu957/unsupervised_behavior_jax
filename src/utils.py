import glob
import numpy as np
import os
import tqdm
import pandas as pd  # Import pandas
import subprocess
import datetime
import logging


def load_keypoints_pd(dir_name):
    keypoint_files = glob.glob(os.path.join(dir_name, "*.csv"))

    coordinates = {}
    confidences = {}

    for filepath in tqdm.tqdm(keypoint_files):
        try:
            # Read the CSV file in chunks
            chunk_iterator = pd.read_csv(
                filepath, skiprows=1, header=None, chunksize=1000
            )
        except ValueError:
            print(f"Error reading {filepath}")
            continue

        name = os.path.basename(filepath)  # Use filename as key

        # Initialize lists to accumulate results from chunks
        coords_list = []
        confs_list = []

        for chunk in chunk_iterator:

            # NOTE: This might reduce the memory usage but can cause numerical instability
            # If using float32, make sure to enable jax's float32 precision
            # ()
            # using jax.config.update("jax_enable_x64", False) at the beginning of the script

            # data = chunk.values.astype(
            #    np.float32
            # )  # Use float32 for reduced memory usage

            data = chunk.values.astype(np.float64)

            # Reshape data: (n_frames, n_keypoints, 3)
            data = data.reshape(data.shape[0], -1, 3)

            # Extract coordinates and swap x and y
            coords = data[:, :, :2][:, :, ::-1]
            coords_list.append(coords)

            # Extract confidences
            confs = data[:, :, 2]
            confs_list.append(confs)

        # Concatenate results from all chunks
        coordinates[name] = np.concatenate(coords_list, axis=0)
        confidences[name] = np.concatenate(confs_list, axis=0)

    print(f"Done reading all files in {dir_name}")

    return coordinates, confidences


def print_gpu_usage():
    try:
        # Run nvidia-smi and capture the output
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        logging.info(result.stdout)
    except FileNotFoundError:
        logging.info(
            "nvidia-smi not found. Ensure NVIDIA drivers are installed.")
    except Exception as e:
        logging.info(f"An error occurred: {e}")


def set_up_logging(log_dir):
    """Set up logging configuration."""
    log_dir.mkdir(exist_ok=True)
    logfilename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_dir/logfilename,
        force=True
    )
    logging.info("Logging setup complete")
