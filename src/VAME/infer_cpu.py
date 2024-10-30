import vame
from vame.util.auxiliary import read_config
import pathlib
import logging
import datetime
import os
import torch
import numpy as np
from vame.model.rnn_model import RNN_VAE
import sys

# Global Variables
PROJECT_PATH = pathlib.Path("/flashscratch/chouda/vame_grooming/grooming_8k-Sep11-2024")
CONFIG_PATH = PROJECT_PATH / "config.yaml"

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

def load_model(cfg, model_name, fixed):
    """
    Load the VAME model from a state dictionary.

    Parameters:
    cfg (dict): Configuration dictionary with model and project settings.
    model_name (str): Name of the model to load.
    fixed (bool): Boolean indicating if the model uses ego-centric pose data or not.

    Returns:
    RNN_VAE: The loaded RNN_VAE model.
    """
    # Set computation device
    device = torch.device("cpu")

    # Model parameters
    ZDIMS = cfg["zdims"]
    FUTURE_DECODER = cfg["prediction_decoder"]
    TEMPORAL_WINDOW = cfg["time_window"] * 2
    FUTURE_STEPS = cfg["prediction_steps"]
    NUM_FEATURES = cfg["num_features"] - 2 if not fixed else cfg["num_features"]

    # Model architecture
    model = RNN_VAE(
        TEMPORAL_WINDOW,
        ZDIMS,
        NUM_FEATURES,
        FUTURE_DECODER,
        FUTURE_STEPS,
        cfg["hidden_size_layer_1"],
        cfg["hidden_size_layer_2"],
        cfg["hidden_size_rec"],
        cfg["hidden_size_pred"],
        cfg["dropout_encoder"],
        cfg["dropout_rec"],
        cfg["dropout_pred"],
        cfg["softplus"],
    ).to(device)

    # Load model state
    model_path = os.path.join(
        cfg["project_path"], "model", "best_model", f"{model_name}_{cfg['Project']}.pkl"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def embed_latent_vectors(cfg, files, model):
    """
    Embed latent vectors for given files using the specified model.

    Parameters:
    cfg (dict): Configuration dictionary with model and project settings.
    files (list): List of file names to process.
    model (RNN_VAE): The loaded RNN_VAE model.

    Returns:
    list: List of latent vector arrays for each file.
    """
    project_path = cfg["project_path"]
    temp_win = cfg["time_window"]
    num_features = (
        cfg["num_features"] - 2 if not cfg["egocentric_data"] else cfg["num_features"]
    )

    latent_vector_files = []

    for file in files:
        logging.info(f"Embedding latent vector for file {file}")
        data_path = os.path.join(project_path, "data", file, f"{file}-PE-seq-clean.npy")
        data = np.load(data_path)
        latent_vectors = []

        with torch.no_grad():
            for i in range(data.shape[1] - temp_win):
                data_sample = data[:, i : temp_win + i].T.reshape((1, temp_win, num_features))
                data_tensor = torch.from_numpy(data_sample).type(torch.FloatTensor).to(torch.device("cpu"))
                mu, _, _ = model.lmbda(model.encoder(data_tensor))
                latent_vectors.append(mu.cpu().numpy())

        latent_vector_files.append(np.concatenate(latent_vectors, axis=0))

    return latent_vector_files

def safe_mkdir(path):
    """
    Create a directory if it does not exist.

    Parameters:
    path (str or Path): Path of the directory to create.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Directory {path} created successfully.")
        else:
            logging.info(f"Directory {path} already exists.")
    except Exception as e:
        logging.error(f"Error creating directory {path}: {e}")

# Main Function

def main(start_index, videos_per_job):
    """
    Main function to load the model, embed latent vectors for specified video files, and save them.

    Parameters:
    start_index (int): Index of the first video to process.
    videos_per_job (int): Number of videos to process in this job.
    """
    # Load configuration
    cfg = read_config(CONFIG_PATH)

    # Set up logging
    logger = set_up_logging(cfg)

    # Prepare results directory
    results_folder = os.path.join(PROJECT_PATH, "inference")
    safe_mkdir(results_folder)

    # Determine files to process
    files = cfg["video_sets"][start_index : start_index + videos_per_job]
    logging.info(f"Processing files: {files}")

    # Load model
    model = load_model(cfg, cfg["model_name"], cfg["egocentric_data"])
    logging.info("Model loaded successfully.")

    # Embed latent vectors
    latent_vectors = embed_latent_vectors(cfg, files, model)
    logging.info(f"Latent vectors computed successfully for {len(files)} files.")

    # Save latent vectors
    for idx, file in enumerate(files):
        save_path = os.path.join(cfg["project_path"], "results", file, cfg["model_name"])
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f"latent_vector_{file}.npy"), latent_vectors[idx])
        logging.info(f"Latent vector saved for file {file}")

if __name__ == "__main__":
    # Read command-line arguments
    start_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    videos_per_job = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    main(start_index, videos_per_job)
