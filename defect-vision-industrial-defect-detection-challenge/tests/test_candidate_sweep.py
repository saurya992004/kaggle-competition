import unittest

import pandas as pd

from src.candidate_sweep import make_candidate


class TestCandidateSweep(unittest.TestCase):
    def test_keeps_ids_and_selects_top_per_category(self):
        frame = pd.DataFrame({
            "id": ["a", "b", "c", "d", "e", "f"],
            "category": ["one", "one", "one", "two", "two", "two"],
            "score_resnet18_spatial": [0.1, 0.9, 0.4, 0.2, 0.8, 0.5],
        })
        candidate = make_candidate(frame, "score_resnet18_spatial", 2 / 3)
        self.assertEqual(candidate.id.tolist(), frame.id.tolist())
        self.assertEqual(candidate.label.tolist(), [0, 1, 1, 0, 1, 1])

    def test_rejects_invalid_rate(self):
        frame = pd.DataFrame({"id": ["a"], "category": ["one"], "score_resnet18_spatial": [1.0]})
        with self.assertRaises(ValueError):
            make_candidate(frame, "score_resnet18_spatial", 1.0)


if __name__ == "__main__":
    unittest.main()
