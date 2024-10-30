import vame
from vame.util.auxiliary import read_config
import pathlib
import logging
import datetime
import sys

# Global Variables
PROJECT_PATH = pathlib.Path("/flashscratch/chouda/vame_grooming/grooming_8k-Sep11-2024")
CONFIG_PATH = PROJECT_PATH / "config.yaml"

NUM_FEATURES = 16  # {"8k": 16, "10k": 20, "12k": 24} Dictionary mapping number of keypoints to the number of features in the VAME model

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

def main():
    """
    Main function to train and evaluate the VAME model.
    """
    # Load configuration
    cfg = read_config(CONFIG_PATH)

    # Set up logging
    logger = set_up_logging(cfg)

    # Ensure the number of features is set correctly for the model
    cfg["num_features"] = NUM_FEATURES


    # Train model
    vame.train_model(CONFIG_PATH, logger=logger)
    logging.info("Model trained successfully!")

    # Evaluate model
    vame.evaluate_model(CONFIG_PATH, logger=logging)
    logging.info("Model evaluated successfully!")

if __name__ == "__main__":
    main()
