import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.phase3_benchmark import (BenchmarkConfig, GlobalDensity, PCAReconstruction, PatchMemory, PixelKNN,
                                  average_precision, load_detector, roc_auc)


class TestPhase3Benchmark(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.normal = np.clip(0.5 + rng.normal(0, 0.015, (6, 32, 32, 3)), 0, 1).astype(np.float32)
        self.anomaly = self.normal[:1].copy(); self.anomaly[:, 10:20, 10:20] = 0.0

    def test_detectors_have_finite_scores(self):
        config = BenchmarkConfig(image_size=32, reconstruction_size=16, pca_components=3, patch_size=8, patch_stride=8)
        for detector in [PixelKNN(.01), GlobalDensity(), PatchMemory(8, 8, .1, 64), PCAReconstruction(16, 3, .01)]:
            detector.fit(self.normal)
            scores = detector.score(np.concatenate([self.normal[:1], self.anomaly]))
            self.assertTrue(np.isfinite(scores).all())
            self.assertGreaterEqual(scores[1], scores[0])

    def test_saved_detector_reloads(self):
        detector = PixelKNN(.01).fit(self.normal)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"; detector.save(path)
            self.assertTrue(np.allclose(detector.score(self.anomaly), load_detector(path).score(self.anomaly)))

    def test_proxy_metrics_ordering(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([.1, .2, .8, .9])
        self.assertEqual(roc_auc(labels, scores), 1.0)
        self.assertEqual(average_precision(labels, scores), 1.0)


if __name__ == "__main__":
    unittest.main()
