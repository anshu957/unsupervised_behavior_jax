import vame
from vame.util.auxiliary import read_config, update_config
import pathlib
import logging
import os
import shutil
import datetime
from tqdm import tqdm

# Global Variables
VIDEO_LOC = pathlib.Path("/projects/kumar-lab/manohh/VAME/VAME/groom/videos/")
POSES_LOC = pathlib.Path("/projects/kumar-lab/chouda/VAME/grooming_poses/poses_8k/")
BASE_NECK_INDEX = 3  # indexing starts from 0
BASE_TAIL_INDEX = 7  # for (9: 12 keypoint model, 9: 10 keypoint model, 7: 8 keypoint model)
VIDEO_FORMAT = ".avi"
POSE_FORMAT = ".csv"
CHECK_ALIGNMENT = False
EXCLUDE_VIDEO_LIST = ["3758", "4015", "3957", "4113", "3875", "3759", "2450", "3347", "2072", "3874", "2328", "4119"]
TEST_VIDEOS = ["2001", "2232", "3750", "4043"]

# Helper Functions

def filter_videos(videos, exclude_list):
    """
    Filter out videos based on an exclusion list.

    Parameters:
    videos (list of Path): List of video paths.
    exclude_list (list of str): List of video names to exclude.

    Returns:
    list of Path: Filtered list of videos.
    """
    return [video for video in videos if video.stem not in exclude_list]

def ensure_directories_exist(directories):
    """
    Ensure that a list of directories exists, creating them if necessary.

    Parameters:
    directories (list of Path): List of directory paths to ensure exist.
    """
    for dir_ in directories:
        dir_.mkdir(exist_ok=True)

def setup_logging(logs_loc):
    """
    Set up logging configuration.

    Parameters:
    logs_loc (Path): Directory where log files will be saved.
    """
    log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_loc / log_filename),
            logging.StreamHandler(),
        ],
    )
    logging.info("Logging setup complete")

def copy_pose_files(pose_files, project_path):
    """
    Copy pose files to the project directory.

    Parameters:
    pose_files (list of Path): List of pose file paths to copy.
    project_path (Path): Path to the project directory.
    """
    for pose_file in tqdm(pose_files, desc="copying pose files"):
        try:
            shutil.copy(pose_file, project_path / "videos/pose_estimation/")
        except Exception as e:
            logging.error(f"Error copying {pose_file} to project directory: {e}")
    logging.info(f"Copied {len(pose_files)} pose files to project directory")

# Main Function

def main():
    """
    Main function to create a new VAME project or update an existing one.
    """
    videos = sorted(list(VIDEO_LOC.glob(f"*{VIDEO_FORMAT}")))
    logging.info(f"Number of videos before filtering: {len(videos)}")

    # Filter out videos in the EXCLUDE_VIDEO_LIST
    videos = filter_videos(videos, EXCLUDE_VIDEO_LIST)
    logging.info(f"Number of videos after filtering: {len(videos)}")

    # Initialize new VAME project
    new_proj_ret = vame.init_new_project(
        "grooming_8k",
        videos=videos,
        working_directory="/flashscratch/chouda/vame_grooming/",
        videotype=VIDEO_FORMAT,
    )

    # Handle the project configuration
    if isinstance(new_proj_ret, tuple):
        config, proj_path = new_proj_ret
        logging.info("Project created successfully!")
    else:
        proj_path = new_proj_ret
        config = read_config(proj_path / "config.yaml")

    logging.info("Project config:")
    logging.info(config)

    # Define project paths
    project_path = config["project_path"]
    working_directory = pathlib.Path(project_path).parent
    output_loc = working_directory / "output"
    logs_loc = working_directory / "logs"

    # Ensure output and logs directories exist
    ensure_directories_exist([logs_loc, output_loc])

    # Set up logging
    setup_logging(logs_loc)

    # If the project is new, copy the pose files
    if isinstance(new_proj_ret, tuple):
        pose_files = sorted(list(POSES_LOC.glob(f"*{POSE_FORMAT}")))
        logging.info(f"Number of pose files before filtering: {len(pose_files)}")
        pose_files = filter_videos(pose_files, EXCLUDE_VIDEO_LIST)
        logging.info(f"Number of pose files after filtering: {len(pose_files)}")

        # Ensure number of pose files matches the number of videos
        assert len(pose_files) == len(videos), "Number of pose files and videos do not match"

        # Copy pose files to project directory
        copy_pose_files(pose_files, pathlib.Path(project_path))

        logging.info("Project setup complete")
        logging.info("=====================================")

if __name__ == "__main__":
    main()
