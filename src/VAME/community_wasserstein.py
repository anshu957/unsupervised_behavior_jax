import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import tqdm
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, to_tree
from scipy.stats import wasserstein_distance
from utils import hierarchy_pos
import matplotlib

# Global Variables
CUT_TREE_HEIGHT = 0.2
SEED = 42
PERCENTAGE_JABS = 0.2  # Percentage of data to be used for JABS clustering
DATA_SUBDIR = "VAME/kmeans-15/"
LATENT_FILENAME_TEMPLATE = "latent_vector_{}.npy"
MOTIF_LABEL_FILENAME_TEMPLATE = "15_km_label_{}.npy"

# Paths
BASE_PATH = pathlib.Path(__file__).parent.absolute()
DATA_DIR = BASE_PATH / "grooming_12k-Aug22-2024/results"
OUTPUT_DIR = BASE_PATH / "output"

# Helper Functions

def ensure_directory_exists(directory):
    """
    Ensure a directory exists; create it if it doesn't.

    Parameters:
    directory (Path or str): The path of the directory to ensure exists.
    """
    """
    Ensure a directory exists; create it if it doesn't.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Output directory created: {directory}")
    else:
        print(f"Output directory already exists: {directory}")

def save_list_to_file(output_path, data_list):
    """
    Save a list of items to a text file.

    Parameters:
    output_path (Path or str): The path to save the text file.
    data_list (list): The list of items to save.
    """
    """
    Save a list of items to a text file.
    """
    with open(output_path, "w") as f:
        for item in data_list:
            f.write("%s\n" % item)

def load_latents_and_labels(video_names, data_dir):
    """
    Load latent embeddings and motif labels for given video names.

    Parameters:
    video_names (list of str): List of video names to load data for.
    data_dir (Path or str): Directory containing the video data.

    Returns:
    tuple: A tuple containing dictionaries of latent embeddings and motif labels.
    """
    """
    Load latent embeddings and motif labels for given video names.
    """
    latent_embeddings, motif_labels = {}, {}
    for video_name in tqdm.tqdm(video_names, desc="Reading latents"):
        latent_file = data_dir / video_name / DATA_SUBDIR / LATENT_FILENAME_TEMPLATE.format(video_name)
        motif_labels_file = data_dir / video_name / DATA_SUBDIR / MOTIF_LABEL_FILENAME_TEMPLATE.format(video_name)
        try:
            latent_embeddings[video_name] = np.load(latent_file)
            motif_labels[video_name] = np.load(motif_labels_file)
        except FileNotFoundError:
            print(f"Warning: File not found for video {video_name}. Skipping.")
        except Exception as e:
            print(f"Warning: An error occurred while loading data for video {video_name}: {e}. Skipping.")
    return latent_embeddings, motif_labels

def plot_motif_usage(video_names, motif_labels, output_dir):
    """
    Plot the motif usage for each video.

    Parameters:
    video_names (list of str): List of video names.
    motif_labels (dict): Dictionary containing motif labels for each video.
    output_dir (Path or str): Directory to save the plots.
    """
    """
    Plot the motif usage for each video.
    """
    for video_name in video_names:
        motif_usage_percentage = np.array(list(motif_usage[video_name].values())) / len(motif_labels[video_name]) * 100
        sorted_motif_indices = np.argsort(motif_usage_percentage)[::-1]

        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(motif_usage_percentage)+1), motif_usage_percentage[sorted_motif_indices])
        plt.xticks(range(1, len(motif_usage_percentage)+1), sorted_motif_indices + 1)

        plt.xlabel("Motif ID")
        plt.ylabel("Percentage of frames (%)")
        plt.title(f"Motif usage in video {video_name}")
        plt.xticks(rotation=90)

        plt.savefig(output_dir / f"motif_usage_{video_name}.png")
        plt.close()

def load_filtered_motifs(filtered_motifs_file):
    """
    Load filtered motifs from file.

    Parameters:
    filtered_motifs_file (Path or str): Path to the file containing filtered motifs.

    Returns:
    list: A list of filtered motif IDs.
    """
    """
    Load filtered motifs from file.
    """
    filtered_motifs = []
    with open(filtered_motifs_file, "r") as f:
        for line in f:
            filtered_motifs.append(int(line.strip()))
    return filtered_motifs

def compute_wasserstein_distance_matrix(motif_latent_embeddings, filtered_motifs):
    """
    Compute the Wasserstein distance matrix for motifs.

    Parameters:
    motif_latent_embeddings (dict): Dictionary containing latent embeddings for each motif.
    filtered_motifs (list): List of filtered motif IDs.

    Returns:
    ndarray: A 2D array representing the Wasserstein distance matrix.
    """
    """
    Compute the Wasserstein distance matrix for motifs.
    """
    n = len(filtered_motifs)
    distance_matrix = np.full((n, n), np.inf)
    for i in tqdm.tqdm(range(n), desc="Computing Wasserstein distance matrix"):
        for j in range(i + 1, n):
            distance_matrix[i, j] = wasserstein_distance(
                motif_latent_embeddings[filtered_motifs[i]],
                motif_latent_embeddings[filtered_motifs[j]],
            )
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

# Main Script
def main():
    """
    Main function to execute the script.
    """
    # Ensure output directory exists
    ensure_directory_exists(OUTPUT_DIR)

    # Load video names and select a subset
    video_names = os.listdir(DATA_DIR)
    np.random.seed(SEED)
    num_videos = int(PERCENTAGE_JABS * len(video_names))
    if num_videos > len(video_names):
        num_videos = len(video_names)
    selected_video_names = np.random.choice(video_names, num_videos, replace=False)
    save_list_to_file(OUTPUT_DIR / f"selected_videos_{PERCENTAGE_JABS}.txt", selected_video_names)
    
    # Load latent embeddings and motif labels
    latent_embeddings, motif_labels = load_latents_and_labels(selected_video_names, DATA_DIR)

    # Load filtered motifs
    filtered_motifs = load_filtered_motifs(OUTPUT_DIR / "filtered_motifs.txt")
    print(f"Number of filtered motifs: {len(filtered_motifs)}")

    # Filter latent embeddings by motifs and combine
    latent_embeddings_filtered, motif_labels_filtered = {}, {}
    for video_name in selected_video_names:
        if video_name not in latent_embeddings:
            continue
        filtered_idxs = np.where(np.isin(motif_labels[video_name], filtered_motifs))[0]
        latent_embeddings_filtered[video_name] = latent_embeddings[video_name][filtered_idxs, :]
        motif_labels_filtered[video_name] = motif_labels[video_name][filtered_idxs]

        # Sanity check
        assert latent_embeddings_filtered[video_name].shape[0] == motif_labels_filtered[video_name].shape[0], "Number of frames and labels do not match!"

    # Combine latent embeddings from all videos
    combined_latent_embeddings = np.concatenate([latent_embeddings_filtered[v] for v in selected_video_names if v in latent_embeddings_filtered], axis=0)
    combined_motif_labels = np.concatenate([motif_labels_filtered[v] for v in selected_video_names if v in motif_labels_filtered], axis=0)

    # Create motif latent embeddings dictionary
    motif_latent_embeddings = {i: [] for i in filtered_motifs}
    for video_name in selected_video_names:
        if video_name not in motif_labels:
            continue
        for i in filtered_motifs:
            motif_idxs = np.where(motif_labels[video_name] == i)[0]
            if len(motif_idxs) > 0:
                motif_latent_embeddings[i].append(latent_embeddings[video_name][motif_idxs, :].flatten())
    
    # Concatenate motif latent embeddings
    for i in filtered_motifs:
        motif_latent_embeddings[i] = np.concatenate(motif_latent_embeddings[i], axis=0)

    # Compute Wasserstein distance matrix
    wasserstein_distance_matrix = compute_wasserstein_distance_matrix(motif_latent_embeddings, filtered_motifs)

    # Plotting
    Z = linkage(wasserstein_distance_matrix, "average")
    fig = plt.figure(figsize=(10, 10))
    dn = dendrogram(Z, labels=filtered_motifs)
    plt.axhline(y=CUT_TREE_HEIGHT, color="r", linestyle="--")
    plt.xlabel("Motif ID")
    plt.ylabel("Wasserstein distance")
    plt.title("Hierarchical clustering of motifs based on Wasserstein distance")
    plt.savefig(OUTPUT_DIR / f"wasserstein_distance_matrix_jabs_perc_{PERCENTAGE_JABS}.pdf")
    plt.close()

if __name__ == "__main__":
    main()
