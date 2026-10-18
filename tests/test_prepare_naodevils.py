import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_naodevils import image_remote, rasterize


def rectangle(category, x1, y1, x2, y2):
    return {"category_id": category, "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]}


class PreparationTests(unittest.TestCase):
    def test_distinct_classes_and_background(self):
        mask = rasterize([rectangle(4, 1, 1, 3, 3), rectangle(1, 5, 1, 7, 3)], (10, 10))
        self.assertEqual(mask[2, 2], 1)
        self.assertEqual(mask[2, 6], 2)
        self.assertEqual(mask[9, 9], 0)
        self.assertEqual(mask.dtype, np.uint8)

    def test_overlap_ignore_occlusion_and_order_independence(self):
        annotations = [rectangle(4, 1, 1, 6, 6), rectangle(1, 3, 3, 8, 8),
                       rectangle(6, 0, 8, 2, 9), rectangle(3, 4, 4, 5, 5)]
        mask = rasterize(annotations, (10, 10))
        np.testing.assert_array_equal(mask, rasterize(annotations[::-1], (10, 10)))
        self.assertEqual(mask[3, 3], 255)
        self.assertEqual(mask[9, 1], 255)
        self.assertEqual(mask[4, 4], 0)

    def test_source_batch_boundaries(self):
        for index, folder in [(500, "00000_00500"), (501, "00500_01000"), (1000, "00500_01000"),
                              (2000, "01500_02000"), (2001, "02000_03000"), (10000, "09000_10000")]:
            self.assertEqual(image_remote(f"{index:05d}_frame.png"), f"upper_{folder}/{index:05d}_frame.png")

    def test_invalid_annotation_rejected(self):
        with self.assertRaises(ValueError):
            rasterize([rectangle(99, 0, 0, 1, 1)], (10, 10))
        with self.assertRaises(ValueError):
            rasterize([{"category_id": 1, "segmentation": [[1, 2, 3]]}], (10, 10))

    def test_thin_source_contours_retained(self):
        mask = rasterize([{"category_id": 1, "segmentation": [[1, 2], [4, 5, 7, 5]]}], (10, 10))
        self.assertEqual(mask[2, 1], 2)
        np.testing.assert_array_equal(mask[5, 4:8], [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
