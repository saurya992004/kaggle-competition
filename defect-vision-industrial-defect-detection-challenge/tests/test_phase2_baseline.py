import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.phase2_baseline import CategoryAnomalyModel, add_synthetic_defect, best_f1_threshold, mean_top_fraction


class TestPhase2Baseline(unittest.TestCase):
    def test_tail_score_prefers_local_outlier(self):
        reference = np.full((3, 12, 12, 3), 0.5, dtype=np.float32)
        model = CategoryAnomalyModel(tail_fraction=0.05).fit(reference)
        normal = np.full((1, 12, 12, 3), 0.5, dtype=np.float32)
        anomalous = normal.copy()
        anomalous[:, 4:8, 4:8] = 0.0
        self.assertGreater(model.score(anomalous)["knn_tail"][0], model.score(normal)["knn_tail"][0])

    def test_synthetic_generation_changes_pixels(self):
        image = np.full((32, 32, 3), 0.5, dtype=np.float32)
        modified = add_synthetic_defect(image, "cutout", np.random.default_rng(7))
        self.assertFalse(np.array_equal(image, modified))
        self.assertEqual(modified.shape, image.shape)

    def test_f1_threshold(self):
        threshold, f1 = best_f1_threshold(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
        self.assertGreaterEqual(threshold, 0.8)
        self.assertEqual(f1, 1.0)

    def test_tail_fraction(self):
        values = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        self.assertEqual(mean_top_fraction(values, 0.25)[0], 4.0)


if __name__ == "__main__":
    unittest.main()
