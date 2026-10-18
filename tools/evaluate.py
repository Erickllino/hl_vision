#!/usr/bin/env python3
"""Prepare shared preprocessing and evaluate local YOLO artifacts offline."""
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from vision_eval.data import VARIANTS, digest, load_manifest, write_json


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="generate variants using shared C++ executable")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--binary", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--params", type=Path, help="JSON object of C++ parameter flags")
    prepare.add_argument("--repeat", type=int, default=1)
    prepare.add_argument("--warmup", type=int, default=1)
    run = sub.add_parser("run", help="evaluate models (only when explicitly invoked)")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--models", type=Path, required=True)
    run.add_argument("--prepared", type=Path)
    run.add_argument("--variants", nargs="+", choices=VARIANTS, default=["original"])
    run.add_argument("--labels", choices=("none", "boxes", "masks"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--validate-only", action="store_true", help="check inputs, no backend import or inference")
    return ap


def prepare(args):
    data = load_manifest(args.manifest, "none")
    if not args.binary.is_file():
        raise ValueError(f"build the shared preprocessor first: {args.binary}")
    params = json.loads(args.params.read_text()) if args.params else {}
    allowed = {"clip-limit", "tile-size", "min-value", "max-saturation", "strength"}
    if set(params) - allowed:
        raise ValueError(f"unknown preprocess parameters: {set(params) - allowed}")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for item in data["images"]:
        for variant in VARIANTS:
            dest = args.output / variant
            dest.mkdir(exist_ok=True)
            command = [str(args.binary.resolve()), "--input", item["image"],
                       "--output", str(dest / f'{item["id"]}.png'),
                       "--variant", variant,
                       "--repeat", str(args.repeat), "--warmup", str(args.warmup)]
            for key, value in params.items():
                command += ["--" + key, str(value)]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            records.append({"id": item["id"], "variant": variant,
                            "source_sha256": digest(item["image"]),
                            "output_sha256": digest(dest / f'{item["id"]}.png'),
                            "timing": json.loads(completed.stdout)})
    write_json(args.output / "preparation.json", {
        "manifest_sha256": digest(args.manifest), "binary_sha256": digest(args.binary),
        "params_overrides": params, "records": records})


def run(args):
    data = load_manifest(args.manifest, args.labels)
    models = json.loads(args.models.read_text())["models"]
    if not models:
        raise ValueError("models list is empty")
    names = set()
    for model in models:
        name = model["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) or name in names:
            raise ValueError("model names must be unique path-safe identifiers")
        names.add(name)
        model["weights"] = str((args.models.resolve().parent / model["weights"]).resolve())
        if not Path(model["weights"]).is_file():
            raise ValueError(f'local weights missing: {model["weights"]}')
        if Path(model["weights"]).suffix not in (".pt", ".onnx", ".engine"):
            raise ValueError("only Ultralytics .pt, .onnx and .engine artifacts supported")
        if model["task"] not in ("detect", "segment") or (args.labels == "masks" and model["task"] != "segment"):
            raise ValueError("mask labels require task=segment; supported tasks: detect, segment")
        mapping = model["class_map"]
        if not mapping or any(not str(k).isdigit() or not isinstance(v, int) or not 0 <= v < len(data["classes"]) for k, v in mapping.items()):
            raise ValueError("class_map must map source class IDs to manifest class IDs")
        size = model["imgsz"]
        if not isinstance(size, list) or len(size) != 2 or any(not isinstance(s, int) or s <= 0 for s in size):
            raise ValueError("imgsz must be [height,width] with positive integers")
        if not 0 < model.get("conf", .001) <= 1 or not 0 < model.get("iou", .7) <= 1:
            raise ValueError("conf and iou must be in (0,1]")
        if model.get("max_det", 300) <= 0:
            raise ValueError("max_det must be positive")
        device = str(model.get("device", "cpu"))
        if device != "cpu" and not device.isdigit():
            raise ValueError("device must be cpu or one CUDA index, e.g. 0")
    preparation = None
    if any(v != "original" for v in args.variants):
        if args.prepared is None:
            raise ValueError("non-original variants require --prepared")
        preparation = json.loads((args.prepared / "preparation.json").read_text())
        if preparation["manifest_sha256"] != digest(args.manifest):
            raise ValueError("prepared manifest differs from evaluation manifest")
        indexed = {(r["id"], r["variant"]): r for r in preparation["records"]}
        for item in data["images"]:
            for variant in args.variants:
                if variant == "original":
                    continue
                record = indexed[item["id"], variant]
                if record["source_sha256"] != digest(item["image"]) or record["output_sha256"] != digest(args.prepared / variant / f'{item["id"]}.png'):
                    raise ValueError("prepared/source image changed; regenerate variants")
    if args.validate_only:
        print(f'Valid configuration: {len(data["images"])} images, {len(models)} models; no inference run')
        return
    import cv2
    import numpy as np
    import torch
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO
    from vision_eval.metrics import MaskMetrics, detection_metrics, latency

    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for spec in models:
        model = YOLO(spec["weights"], task=spec["task"])
        for variant in args.variants:
            target = args.output / spec["name"] / variant
            target.mkdir(parents=True)
            times, rows, truth, predictions = [], [], {}, {}
            masks_metric = MaskMetrics(data["classes"])
            def predict(image):
                return model.predict(image, imgsz=spec["imgsz"], device=spec.get("device", "cpu"),
                                     conf=spec.get("conf", .001), iou=spec.get("iou", .7),
                                     max_det=spec.get("max_det", 300), half=spec.get("half", False),
                                     retina_masks=True, verbose=False)[0]
            def sync():
                if torch.cuda.is_available() and str(spec.get("device", "cpu")) != "cpu":
                    torch.cuda.synchronize(torch.device("cuda:" + str(spec.get("device", "0"))))
            for index, item in enumerate(data["images"]):
                path = Path(item["image"]) if variant == "original" else args.prepared / variant / f'{item["id"]}.png'
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"cannot decode: {path}")
                if index == 0:
                    for _ in range(args.warmup):
                        predict(image)
                    sync()
                for repeat in range(args.repeat):
                    sync()
                    start = time.perf_counter_ns()
                    result = predict(image)
                    boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else np.empty((0, 6))
                    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
                    sync()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    times.append(elapsed)
                    rows.append({"id": item["id"], "repeat": repeat, "predict_wall_ms": elapsed})
                output_boxes = []
                union = np.zeros((len(data["classes"]), *image.shape[:2]), dtype=bool)
                for i, box in enumerate(boxes):
                    source_class = str(int(box[5]))
                    if source_class not in spec["class_map"]:
                        raise ValueError(f"unmapped output class {source_class} from {spec['name']}")
                    cls = spec["class_map"][source_class]
                    output_boxes.append([cls, *map(float, box[:4]), float(box[4])])
                    if masks is not None:
                        mask = masks[i]
                        if mask.shape != image.shape[:2]:
                            raise ValueError("backend did not return original-resolution retina masks")
                        union[cls] |= mask > .5
                predictions[item["id"]] = output_boxes
                if spec["task"] == "segment":
                    if len(boxes) and masks is None:
                        raise ValueError("segmentation backend returned boxes without masks")
                    np.savez_compressed(target / f'{item["id"]}_masks.npz', masks=union)
                if args.labels == "boxes":
                    if any(box[3] > image.shape[1] or box[4] > image.shape[0] for box in item["boxes"]):
                        raise ValueError("ground truth box exceeds image dimensions")
                    truth[item["id"]] = item["boxes"]
                elif args.labels == "masks":
                    gt = cv2.imread(item["mask"], cv2.IMREAD_UNCHANGED)
                    if gt is None or gt.shape != image.shape[:2] or gt.dtype != np.uint8:
                        raise ValueError("labels must be uint8 grayscale masks at original image resolution")
                    if not np.isin(gt, [0, 255, *range(1, len(data["classes"]) + 1)]).all():
                        raise ValueError("unknown mask class ID")
                    masks_metric.add(gt, union)
            metrics = (detection_metrics(truth, predictions, data["classes"]) if args.labels == "boxes"
                       else masks_metric.result() if args.labels == "masks" else None)
            summary = {"model": spec["name"], "variant": variant, "labels": args.labels,
                       "metrics": metrics, "predict_wall_latency": latency(times)}
            summaries.append(summary)
            write_json(target / "predictions.json", predictions)
            write_json(target / "summary.json", summary)
            with (target / "latency.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    write_json(args.output / "comparison.json", summaries)
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "variant", "labels", "mean_ms", "p95_ms", "map50", "map50_95", "miou"])
        writer.writeheader()
        for summary in summaries:
            writer.writerow({**{k: summary[k] for k in ("model", "variant", "labels")},
                             **{k: summary["predict_wall_latency"][k] for k in ("mean_ms", "p95_ms")},
                             **{k: (summary["metrics"] or {}).get(k) for k in ("map50", "map50_95", "miou")}})
    write_json(args.output / "run.json", {
        "manifest_sha256": digest(args.manifest), "models": models,
        "weights_sha256": {m["name"]: digest(m["weights"]) for m in models},
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("ultralytics", "torch", "numpy")},
        "opencv": cv2.__version__, "cuda": torch.version.cuda,
        "warmup": args.warmup, "repeat": args.repeat, "preparation": preparation,
        "classes": data["classes"], "manifest": data,
        "image_sha256": {item["id"]: digest(item["image"]) for item in data["images"]},
        "mask_sha256": ({item["id"]: digest(item["mask"]) for item in data["images"]}
                        if args.labels == "masks" else None),
        "latency_scope": "predict + CPU output transfer; excludes image IO, shared preprocessing, metrics, output encoding"})


def main():
    ap = parser()
    args = ap.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        ap.error("repeat must be >=1 and warmup >=0")
    try:
        (prepare if args.command == "prepare" else run)(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        ap.exit(2, f"evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
