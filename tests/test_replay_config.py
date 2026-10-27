import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_config import load, write_config


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).resolve().parents[1] / "tests" / "vision_sample.yaml"

    def test_full_copy_and_override(self):
        base, values = load(self.source)
        values["strength"] = 0.4
        values["tile_width"] = 4
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.yaml"
            write_config(base, values, path)
            changed, actual = load(path)
        self.assertEqual(actual["strength"], 0.4)
        self.assertEqual(changed["segmentation_model"]["line_preprocessing"]["clahe"]["tile_grid_size"], [4, 4])
        self.assertTrue(changed["publish_preprocess_video"])
        self.assertFalse(changed["use_depth"])
        self.assertEqual(base["segmentation_model"]["line_preprocessing"]["white_enhancement"]["strength"], 0.9)

    def test_invalid_value(self):
        base, values = load(self.source)
        values["min_value"] = 300
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                write_config(base, values, Path(folder) / "out.yaml")


if __name__ == "__main__":
    unittest.main()
