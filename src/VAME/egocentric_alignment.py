import vame
from vame.util.auxiliary import read_config
import pathlib
import logging
import datetime
from tqdm import tqdm
import sys

# Global Variables
VIDEO_LOC = pathlib.Path("/projects/kumar-lab/manohh/VAME/VAME/groom/videos/")
POSES_LOC = pathlib.Path("/projects/kumar-lab/manohh/VAME/VAME/groom/poses_8k/")
BASE_NECK_INDEX = 3  # indexing starts from 0
BASE_TAIL_INDEX = 7  # for (9: 12 keypoint model, 9: 10 keypoint model, 7: 8 keypoint model)
VIDEO_FORMAT = ".avi"
CHECK_ALIGNMENT = False  # True when testing
VIDEOS_PER_JOB = 50  # Videos per parallel job
PROJECT_PATH = pathlib.Path("/flashscratch/chouda/vame_grooming/grooming_8k-Sep11-2024")

# Helper Functions

def set_up_logging(config):
    """
    Set up logging for the project.

    Parameters:
    config (dict): Configuration dictionary containing project information.

    Returns:
    logger: Configured logger object.
    """
    project_path = config["project_path"]
    working_directory = pathlib.Path(project_path).parent
    output_loc = working_directory / "output"
    logs_loc = working_directory / "logs"

    # Create logs and output directories if they don't exist
    for dir_ in [logs_loc, output_loc]:
        dir_.mkdir(exist_ok=True)

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
    return logging.getLogger(__name__)

# Main Function

def main(job_index):
    """
    Main function to align egocentric videos using VAME.

    Parameters:
    job_index (int): The job index used to split video processing.
    """
    config_path = PROJECT_PATH / "config.yaml"
    cfg = read_config(config_path)
    logger = set_up_logging(cfg)

    vame.egocentric_alignment(
        config=config_path,
        logging=logger,
        pose_ref_index=[BASE_NECK_INDEX, BASE_TAIL_INDEX],
        video_format=VIDEO_FORMAT,
        use_video=CHECK_ALIGNMENT,
        check_video=CHECK_ALIGNMENT,
        crop_size=(400, 400),
        job_index=job_index,
        videos_per_job=VIDEOS_PER_JOB,
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <job_index> ")
        sys.exit(1)

    job_index = int(sys.argv[1])
    main(job_index)
