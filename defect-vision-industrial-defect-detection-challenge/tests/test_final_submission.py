import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from src.submission_validator import validate


class TestFinalSubmission(unittest.TestCase):
    def test_validator_rejects_bad_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pd.DataFrame({"id": ["a", "b"], "path": ["a.png", "b.png"], "category": ["x", "x"]}).to_csv(root / "test.csv", index=False)
            pd.DataFrame({"id": ["a", "b"], "label": [0, 0]}).to_csv(root / "sample_submission.csv", index=False)
            pd.DataFrame({"id": ["b", "a"], "label": [0, 1]}).to_csv(root / "submission.csv", index=False)
            self.assertFalse(validate(root, root / "submission.csv")["valid"])

    def test_validator_accepts_valid_numeric_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pd.DataFrame({"id": ["a", "b"], "path": ["a.png", "b.png"], "category": ["x", "x"]}).to_csv(root / "test.csv", index=False)
            pd.DataFrame({"id": ["a", "b"], "label": [0, 0]}).to_csv(root / "sample_submission.csv", index=False)
            pd.DataFrame({"id": ["a", "b"], "label": [0, 1]}).to_csv(root / "submission.csv", index=False)
            self.assertTrue(validate(root, root / "submission.csv")["valid"])


if __name__ == "__main__": unittest.main()
