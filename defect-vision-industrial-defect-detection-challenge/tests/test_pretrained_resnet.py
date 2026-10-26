import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from src.pretrained_resnet_benchmark import Config, global_knn_score, patch_score


class TestPretrainedScores(unittest.TestCase):
    def test_global_score_rises_for_shifted_feature(self):
        reference = np.array([[1., 0.], [.98, .02]], dtype=np.float32)
        score = global_knn_score(reference, np.array([[1., 0.], [0., 1.]], dtype=np.float32))
        self.assertGreater(score[1], score[0])

    def test_spatial_score_finite_and_ordered(self):
        rng = np.random.default_rng(2); reference = rng.normal(0, .01, (5, 4, 4, 3)).astype(np.float32)
        normal, anomalous = reference[:1], reference[:1].copy(); anomalous[:, 1:3, 1:3] += 2
        self.assertGreater(patch_score(reference, anomalous, .1)[0], patch_score(reference, normal, .1)[0])

    def test_vit_tokens_become_spatial_patches(self):
        tokens = torch.zeros(2, 197, 3)
        patches = tokens[:, 1:].reshape(-1, 14, 14, tokens.shape[-1])
        self.assertEqual(tuple(patches.shape), (2, 14, 14, 3))


if __name__ == "__main__": unittest.main()
