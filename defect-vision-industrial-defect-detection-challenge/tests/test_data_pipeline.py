import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data_loader import DataConfig, MetadataImageDataset, load_pil_image, preprocess_image, resolve_image_path


class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = PROJECT
        cls.train = pd.read_csv(cls.root / "train.csv")

    def test_metadata_contract(self):
        for name in ("train.csv", "test.csv"):
            frame = pd.read_csv(self.root / name)
            self.assertTrue({"id", "path", "category"}.issubset(frame.columns))
            self.assertTrue(frame.id.is_unique)
            self.assertFalse(frame.isna().any().any())
            self.assertTrue(all(resolve_image_path(self.root, name).is_file() for name in frame.path))

    def test_training_labels_are_single_valued(self):
        self.assertEqual(self.train.label.nunique(), 1)

    def test_real_image_preprocessing_shape(self):
        row = self.train.iloc[0]
        image = load_pil_image(resolve_image_path(self.root, row.path))
        output = preprocess_image(image, DataConfig(image_size=256))
        self.assertEqual(output.shape, (3, 256, 256))
        self.assertEqual(str(output.dtype), "float32")

    def test_dataset_preserves_metadata_order(self):
        dataset = MetadataImageDataset(self.root / "test.csv", self.root)
        self.assertEqual(len(dataset), len(pd.read_csv(self.root / "test.csv")))
        self.assertEqual(dataset[0]["id"], pd.read_csv(self.root / "test.csv").iloc[0].id)


if __name__ == "__main__":
    unittest.main()
