import vame
from vame.util.auxiliary import read_config, update_config
import pathlib
import logging
import os
import shutil
import datetime
from tqdm import tqdm
import numpy as np
import sys


PROJECT_PATH = pathlib.Path("/projects/kumar-lab/chouda/VAME/grooming_8k-Sep11-2024")

CONFIG_PATH = PROJECT_PATH / "config.yaml"

SEG_METHOD = "kmeans"  # Options: ["hmm", "kmeans"]
NUM_MOTIFS = 15  # Number of motifs to segment the grooming video into

# NOTE: If HMM is set, please ensure you request memory ~ 64GB on GPU and run time > 10 hours. For kmeans, you can run on a CPU with less than 16GB memory and in less than 1 hour.


# main function
if __name__ == "__main__":

    config_path = PROJECT_PATH / "config.yaml"
    cfg = read_config(config_path)

    # Set the segmentation method
    cfg["parameterization"] = SEG_METHOD

    # Set n_init_kmeans in config
    cfg["n_init_kmeans"] = NUM_MOTIFS

    vame.pose_segmentation(config=CONFIG_PATH)
    print("Segmentation done!")
