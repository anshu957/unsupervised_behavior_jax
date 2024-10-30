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


PROJECT_PATH = pathlib.Path(
    "/flashscratch/chouda/vame_grooming/grooming_10k_filtered-Sep5-2024"
)

CONFIG_PATH = PROJECT_PATH / "config.yaml"

BASE_NECK_INDEX = 3  # indexing starts from 0
BASE_TAIL_INDEX = (
    9  # for (9: 12 keypoint model, 9: 10 keypoint model, 7: 8 keypoint model)
)

# main function
if __name__ == "__main__":

    vame.motif_videos(CONFIG_PATH, videoType=".avi")

    # OPTIONAL: Create behavioural hierarchies via community detection
    # vame.community(CONFIG_PATH, show_umap=False, cut_tree=2)

    # OPTIONAL: Create community videos to get insights about behavior on a hierarchical scale
    # vame.community_videos(CONFIG_PATH)

    # vame.visualization(CONFIG_PATH, label="motif")

    # vame.gif(
    #    CONFIG_PATH,
    #    pose_ref_index=[BASE_NECK_INDEX, BASE_TAIL_INDEX],
    #    subtract_background=False,
    #    start=None,
    #    length=500,
    #    max_lag=30,
    #    label="motif",
    #    file_format=".avi",
    #    crop_size=(400, 400),
    # )
