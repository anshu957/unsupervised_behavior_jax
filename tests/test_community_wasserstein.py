import unittest
import numpy as np
from pathlib import Path
import shutil
from refactored_script import ensure_directory_exists, save_list_to_file, load_filtered_motifs, compute_wasserstein_distance_matrix

class TestCommunityWassertein(unittest.TestCase):

    def test_load_filtered_motifs(self):
        test_file = Path("test_filtered_motifs.txt")
        test_data = [1, 2, 3, 4]
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(f"{item}\n")
        loaded_data = load_filtered_motifs(test_file)
        self.assertEqual(loaded_data, test_data)
        # Clean up
        if test_file.exists():
            test_file.unlink()

    def test_compute_wasserstein_distance_matrix(self):
        # Mock data for latent embeddings of motifs
        motif_latent_embeddings = {
            1: np.array([0.1, 0.2, 0.3]),
            2: np.array([0.4, 0.5, 0.6]),
            3: np.array([0.7, 0.8, 0.9])
        }
        filtered_motifs = [1, 2, 3]

        # Compute Wasserstein distance matrix
        distance_matrix = compute_wasserstein_distance_matrix(motif_latent_embeddings, filtered_motifs)

        # Check that the matrix is symmetric
        self.assertTrue(np.allclose(distance_matrix, distance_matrix.T), "Distance matrix should be symmetric")

        # Check that diagonal elements are zero
        self.assertTrue(np.all(np.diag(distance_matrix) == 0), "Diagonal elements of distance matrix should be zero")

        # Check that non-diagonal elements are non-negative
        self.assertTrue(np.all(distance_matrix >= 0), "All distances should be non-negative")

if __name__ == "__main__":
    unittest.main()

