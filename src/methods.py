# %%
import keypoint_moseq as kpms
import logging
from src.utils import load_keypoints_pd
import numpy as np
from src.utils import print_gpu_usage


def load_and_format_data(pose_dir, project_path):
    """Load keypoints and format data."""
    coordinates, confidences = load_keypoints_pd(pose_dir)

    total_frames = sum(coord.shape[0] for coord in coordinates.values())
    logging.info(f"Total number of frames: {total_frames}")

    # how many frames have NaN as confidence
    nan_frames = sum(np.isnan(conf).sum() for conf in confidences.values())
    logging.info(f"Number of frames with NaN confidence: {nan_frames}")

    def config_func(): return kpms.load_config(project_path)

    data, metadata = kpms.format_data(
        coordinates, confidences, **config_func())

    logging.info("Data formatting done")

    return data, metadata, coordinates


def perform_pca(data, config_func, project_path):
    """Perform PCA on the data."""

    # Perform PCA
    pca = kpms.fit_pca(**data, **config_func())

    # Save the PCA object
    kpms.save_pca(pca, project_path)

    # Additional analysis and plotting
    kpms.print_dims_to_explain_variance(pca, 0.9)
    kpms.plot_scree(pca, project_dir=project_path, savefig=True)
    kpms.plot_pcs(pca, project_dir=project_path, **config_func(), savefig=True)

    logging.info("PCA done")

    return pca


def fit_and_save_model(data, metadata, pca, config_func, project_path, G_KAPPA=0.1, G_ARHMM_ITERS=10, G_FULL_MODEL_ITERS=10):
    """Fit model and save results."""
    model = kpms.init_model(data, pca=pca, **config_func())

    # Update kappa and fit additional iterations
    model = kpms.update_hypparams(model, kappa=G_KAPPA)

    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        project_path,
        ar_only=True,
        num_iters=G_ARHMM_ITERS,
        parallel_message_passing=False,
    )

    logging.info("GPU usage after fitting ARHMM: ")
    print_gpu_usage()

    # Load model checkpoint
    # model, data, metadata, current_iter = kpms.load_checkpoint(
    #    project_path, model_name, iteration=G_ARHMM_ITERS
    # )

    # Update kappa and fit additional iterations
    model = kpms.update_hypparams(model, kappa=0.1*G_KAPPA)

    model = kpms.fit_model(
        model,
        data,
        metadata,
        project_path,
        model_name,
        ar_only=False,
        start_iter=G_ARHMM_ITERS,
        num_iters=G_ARHMM_ITERS + G_FULL_MODEL_ITERS,
        parallel_message_passing=False,
    )[0]

    logging.info("GPU usage after fitting full model: ")
    print_gpu_usage()

    # Reindex syllables in checkpoint
    kpms.reindex_syllables_in_checkpoint(project_path, model_name)

    results = kpms.extract_results(model, metadata, project_path, model_name)

    kpms.save_results_as_csv(results, project_path, model_name)
    return model, model_name, results


def generate_plots_and_movies(model_name, results, coordinates, project_path):
    """Generate plots and movies from results."""
    # results = kpms.load_results(project_path, model_name)

    def config_func(): return kpms.load_config(project_path)

    kpms.generate_trajectory_plots(
        coordinates, results, project_path, model_name, **config_func()
    )
    logging.info("Trajectory plots generated")

    kpms.generate_grid_movies(
        results, project_path, model_name, coordinates=coordinates, **config_func()
    )
    logging.info("Grid movies generated")

    kpms.plot_similarity_dendrogram(
        coordinates, results, project_path, model_name, **config_func()
    )
    logging.info("Similarity dendrogram generated")
