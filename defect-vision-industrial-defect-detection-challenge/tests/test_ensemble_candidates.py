import unittest

import pandas as pd

from src.ensemble_candidates import ensemble_scores


class TestEnsembleCandidates(unittest.TestCase):
    def test_averages_aligned_category_ranks(self):
        first = pd.DataFrame({"id": ["a", "b"], "category": ["x", "x"], "rank_resnet18_spatial": [.2, .8]})
        second = pd.DataFrame({"id": ["a", "b"], "category": ["x", "x"], "rank_resnet50_spatial": [.4, .6]})
        result = ensemble_scores(first, second)
        self.assertEqual(result.id.tolist(), ["a", "b"])
        self.assertAlmostEqual(result.score_spatial_rank_ensemble.iloc[0], .3)
        self.assertAlmostEqual(result.score_spatial_rank_ensemble.iloc[1], .7)

    def test_rejects_misaligned_ids(self):
        first = pd.DataFrame({"id": ["a"], "category": ["x"], "rank_resnet18_spatial": [.2]})
        second = pd.DataFrame({"id": ["b"], "category": ["x"], "rank_resnet50_spatial": [.4]})
        with self.assertRaises(ValueError):
            ensemble_scores(first, second)


if __name__ == "__main__":
    unittest.main()
