#!/usr/bin/env python3
"""Download only NAODevils v36 manual validation and prepare two-class masks."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.parse import quote
import zipfile

import cv2
import numpy as np
import requests

SOURCE = "pietbroemmel/naodevils-segmentation-upper-camera"
VERSION = 36
CLASSES = ["CircleLine", "Line"]
CATEGORIES = {1: "line", 2: "ball", 3: "robot", 4: "centercircle", 5: "goal", 6: "penaltycross"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(remote, destination):
    """Resume from complete local files; never persist an HTTP error as data."""
    if destination.exists():
        return destination
    url = f"https://www.kaggle.com/api/v1/datasets/download/{SOURCE}/{quote(remote, safe='')}"
    for attempt in range(4):
        try:
            response = requests.get(url, params={"datasetVersionNumber": VERSION}, timeout=90)
            response.raise_for_status()
            data = response.content
            if data.startswith(b"PK"):
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    data = archive.read(Path(remote).name)
            if remote.endswith(".json"):
                json.loads(data)
            elif not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Expected PNG for {remote}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(destination)
            return destination
        except (requests.RequestException, ValueError, zipfile.BadZipFile):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def image_remote(filename):
    # Published batches have inclusive upper bounds: image 01000 is in 00500_01000.
    index = max(0, int(filename.split("_", 1)[0]) - 1)
    step = 500 if index < 2000 else 1000
    start = index // step * step
    return f"upper_{start:05d}_{start + step:05d}/{filename}"


def rasterize(annotations, shape):
    """Match the author's rounded OpenCV polygon rasterization, preserving holes.

    Union separately per annotation, then per category; unknown class overlap is
    ignored rather than resolved by arbitrary JSON order. Visible objects override
    field paint; penalty marks are ignored until the model's ontology is confirmed.
    """
    layers = {key: np.zeros(shape, np.uint8) for key in CATEGORIES}
    for annotation in annotations:
        category = annotation["category_id"]
        if category not in layers:
            raise ValueError(f"Unexpected category {category}")
        polygons = annotation["segmentation"]
        if not isinstance(polygons, list):
            raise ValueError("Expected original polygon annotations, not RLE")
        points = []
        for polygon in polygons:
            values = np.asarray(polygon, dtype=np.float64)
            # Source includes single-pixel and two-point contours for thin paint.
            # OpenCV supports these; retaining them matches the author's rasterizer.
            if values.size < 2 or values.size % 2 or not np.isfinite(values).all():
                raise ValueError("Invalid polygon")
            points.append(values.reshape(-1, 2).round().astype(np.int32))
        if points:
            instance = np.zeros(shape, np.uint8)
            cv2.fillPoly(instance, points, 1)
            layers[category] |= instance
    circle, line = layers[4].astype(bool), layers[1].astype(bool)
    mask = np.zeros(shape, np.uint8)
    mask[circle] = 1
    mask[line] = 2
    mask[circle & line] = 255
    mask[layers[6].astype(bool)] = 255
    occluders = (layers[2] | layers[3] | layers[5]).astype(bool)
    mask[occluders] = 0
    return mask


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def prepare(root, workers):
    raw = root / "raw" / "naodevils-v36"
    annotation_path = download("annotations/manual_val.json", raw / "annotations/manual_val.json")
    source = json.loads(annotation_path.read_text())
    if {c["id"]: c["name"] for c in source["categories"]} != CATEGORIES:
        raise ValueError("Source category schema changed")
    images = source["images"]
    if len(images) != 151 or len({i["id"] for i in images}) != 151:
        raise ValueError("Expected the 151-image v36 manual validation split")
    for image in images:
        name = image["file_name"]
        if Path(name).name != name or "RoboCup2019_Team-Team" not in name:
            raise ValueError(f"Unexpected filename/split: {name}")
    annotations = defaultdict(list)
    for annotation in source["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    def fetch(image):
        name = image["file_name"]
        return download(image_remote(name), raw / "images" / name)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(pool.map(fetch, images), 1):
            if count % 25 == 0 or count == len(images):
                print(f"Images available: {count}/{len(images)}", flush=True)
    output = root / "eval" / "naodevils-v36-manual-val"
    output.mkdir(parents=True, exist_ok=True)
    entries, records, thumbnails = [], [], []
    totals = np.zeros(256, dtype=np.int64)
    presence = {name: 0 for name in CLASSES}
    seen_hashes = set()
    for index, image in enumerate(images):
        path = raw / "images" / image["file_name"]
        bgr = cv2.imread(str(path))
        if bgr is None or bgr.shape != (480, 640, 3):
            raise ValueError(f"Invalid image or unexpected annotation resolution: {path}")
        image_hash = digest(path)
        if image_hash in seen_hashes:
            raise ValueError(f"Exact duplicate image: {path}")
        seen_hashes.add(image_hash)
        sample_id = f"naodevils_v36_val_{image['id']:05d}"
        mask = rasterize(annotations[image["id"]], bgr.shape[:2])
        mask_path = output / "masks" / f"{sample_id}.png"
        write_png(mask_path, mask)
        for value, name in enumerate(CLASSES, 1):
            presence[name] += int(np.any(mask == value))
        totals += np.bincount(mask.ravel(), minlength=256)
        entries.append({"id": sample_id, "image": str(Path("../../raw/naodevils-v36/images") / path.name),
                        "mask": f"masks/{sample_id}.png"})
        records.append({"id": sample_id, "source_file": path.name, "image_sha256": image_hash,
                        "mask_sha256": digest(mask_path)})
        overlay = bgr.copy()
        for value, color in [(1, (255, 0, 255)), (2, (0, 255, 0)), (255, (0, 165, 255))]:
            selected = mask == value
            overlay[selected] = (0.45 * bgr[selected] + 0.55 * np.array(color)).astype(np.uint8)
        write_png(output / "overlays" / f"{sample_id}.png", overlay)
        if index % 6 == 0:
            thumb = cv2.resize(overlay, (320, 240))
            cv2.putText(thumb, str(image["id"]), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            thumbnails.append(thumb)
    while len(thumbnails) % 4:
        thumbnails.append(np.zeros_like(thumbnails[0]))
    sheet = np.vstack([np.hstack(thumbnails[i:i + 4]) for i in range(0, len(thumbnails), 4)])
    write_png(output / "contact_sheet.jpg", sheet)
    report = {"source": f"https://www.kaggle.com/datasets/{SOURCE}", "version": VERSION,
              "license": "CC BY 4.0", "attribution": "Piet Broemmel / NAODevils",
              "split": "manual_val / RoboCup2019_Team-Team", "images": len(entries),
              "classes": CLASSES, "mask_ids": {"background": 0, "CircleLine": 1, "Line": 2, "ignore": 255},
              "mapping": {"centercircle": "CircleLine", "line": "Line", "penaltycross": "ignore",
                          "ball,robot,goal": "background (occluders take priority)", "line/circle overlap": "ignore"},
              "rasterizer": "OpenCV rounded fillPoly, per annotation, as in original example",
              "annotations_sha256": digest(annotation_path), "opencv": cv2.__version__,
              "images_with_class": presence, "pixel_counts": {str(i): int(totals[i]) for i in (0, 1, 2, 255)},
              "training_overlap": "Unknown: verify model training provenance", "files": records}
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "dataset.json").write_text(json.dumps({"classes": CLASSES, "images": entries}, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("images", "images_with_class", "pixel_counts")}, indent=2))
    print(f"Manifest: {output / 'dataset.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "Data")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    prepare(args.data_root.resolve(), args.workers)
