"""Synthetic checks only: no weights, datasets, downloads or benchmarks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from vision_eval.metrics import MaskMetrics, detection_metrics
from vision_eval.data import load_manifest
import evaluate


class MetricTests(unittest.TestCase):
    def test_perfect_and_duplicate_detection(self):
        gt = {"a": [[0, 0, 0, 10, 10]], "b": []}
        result = detection_metrics(gt, {"a": [[0, 0, 0, 10, 10, .9], [0, 0, 0, 10, 10, .8]], "b": []}, ["line"])
        self.assertEqual(result["map50_95"], 1.)
        self.assertEqual(result["per_class"][0]["at_iou50"]["fp"], 1)
        self.assertEqual(result["per_class"][0]["at_iou50"]["precision"], .5)

    def test_false_positive_before_true_positive_and_missing(self):
        gt = {"a": [[0, 0, 0, 10, 10]], "b": []}
        result = detection_metrics(gt, {"a": [[0, 0, 0, 10, 10, .5]], "b": [[0, 0, 0, 10, 10, .9]]}, ["line"])
        self.assertAlmostEqual(result["map50"], .5)
        missing = detection_metrics(gt, {"a": [], "b": []}, ["line"])
        self.assertEqual(missing["map50"], 0)
        self.assertEqual(missing["per_class"][0]["at_iou50"]["fn"], 1)

    def test_wrong_class_and_no_ground_truth(self):
        result = detection_metrics({"a": [[0, 0, 0, 10, 10]]}, {"a": [[1, 0, 0, 10, 10, .9]]}, ["line", "ball"])
        self.assertEqual(result["map50"], 0)
        self.assertIsNone(result["per_class"][1]["ap50"])
        self.assertEqual(result["per_class"][1]["at_iou50"]["fp"], 1)

    def test_mask_ignore_and_background_false_positive(self):
        metric = MaskMetrics(["line", "absent"])
        metric.add(np.array([[1, 0], [1, 255]], dtype=np.uint8),
                   np.array([[[True, True], [False, True]], [[False, False], [False, False]]]))
        result = metric.result()
        self.assertAlmostEqual(result["miou"], 1 / 3)
        self.assertEqual(result["per_class"][0]["dice"], .5)
        self.assertIsNone(result["per_class"][1]["iou"])


class ManifestTests(unittest.TestCase):
    def test_runner_exports_unlabelled_outputs_without_quality_claim(self):
        class Array:
            def cpu(self):
                return self

            def numpy(self):
                return np.array([[0, 0, 2, 2, .9, 0]])

        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, *args, **kwargs):
                return [SimpleNamespace(boxes=SimpleNamespace(data=Array()), masks=None)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "image.png").write_bytes(b"synthetic image placeholder")
            (root / "model.pt").write_bytes(b"synthetic model placeholder")
            manifest = root / "data.json"
            manifest.write_text(json.dumps({"classes": ["line"], "images": [{"id": "a", "image": "image.png"}]}))
            config = root / "models.json"
            config.write_text(json.dumps({"models": [{"name": "test", "weights": "model.pt", "task": "detect", "class_map": {"0": 0}, "imgsz": [32, 32]}]}))
            args = evaluate.parser().parse_args(["run", "--manifest", str(manifest), "--models", str(config),
                                                "--labels", "none", "--output", str(root / "out"),
                                                "--warmup", "0", "--repeat", "2"])
            modules = {
                "cv2": SimpleNamespace(__version__="fake", imread=lambda *a: np.zeros((4, 4, 3), dtype=np.uint8)),
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), version=SimpleNamespace(cuda=None)),
                "ultralytics": SimpleNamespace(YOLO=FakeModel),
            }
            with patch.dict(sys.modules, modules), patch("importlib.metadata.version", return_value="test"):
                evaluate.run(args)
            result = json.loads((root / "out/comparison.json").read_text())[0]
            self.assertIsNone(result["metrics"])
            self.assertEqual(result["predict_wall_latency"]["samples"], 2)
            predictions = json.loads((root / "out/test/original/predictions.json").read_text())
            self.assertEqual(len(predictions["a"]), 1)
            self.assertTrue((root / "out/comparison.csv").exists())
            self.assertTrue((root / "out/run.json").exists())

    def test_explicit_empty_labels_and_validate_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Deliberately not valid model/image bytes: validate-only checks paths/schema.
            (root / "a.png").write_bytes(b"fixture")
            (root / "model.pt").write_bytes(b"fixture")
            manifest = root / "data.json"
            manifest.write_text(json.dumps({"classes": ["line"], "images": [{"id": "a", "image": "a.png", "boxes": []}]}))
            self.assertEqual(load_manifest(manifest, "boxes")["images"][0]["boxes"], [])
            config = root / "models.json"
            config.write_text(json.dumps({"models": [{"name": "test", "weights": "model.pt", "task": "detect", "class_map": {"0": 0}, "imgsz": [32, 32]}]}))
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "tools/evaluate.py"),
                                        "run", "--manifest", str(manifest), "--models", str(config),
                                        "--labels", "boxes", "--output", str(root / "out"), "--validate-only"],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((root / "out").exists())

    def test_missing_labels_not_silently_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").touch()
            manifest = root / "data.json"
            manifest.write_text(json.dumps({"classes": ["line"], "images": [{"id": "a", "image": "a.png"}]}))
            with self.assertRaises(KeyError):
                load_manifest(manifest, "boxes")


if __name__ == "__main__":
    unittest.main()
