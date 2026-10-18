import hashlib
import json
from pathlib import Path

VARIANTS = ("original", "clahe", "white", "combined")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path, labels):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    classes = data["classes"]
    if not classes or len(classes) > 254 or len(set(classes)) != len(classes):
        raise ValueError("classes must contain 1..254 unique names")
    seen = set()
    if not data["images"]:
        raise ValueError("manifest has no images")
    for item in data["images"]:
        key = item["id"]
        if not isinstance(key, str) or not key or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in key):
            raise ValueError("image id must use only letters, digits, _ or -")
        if key in seen:
            raise ValueError(f"duplicate id: {key}")
        seen.add(key)
        for field in ("image", "mask") if labels == "masks" else ("image",):
            item[field] = str((path.parent / item[field]).resolve())
            if not Path(item[field]).is_file():
                raise ValueError(f"missing {field}: {item[field]}")
        if labels == "boxes":
            for box in item["boxes"]:
                if len(box) != 5 or not isinstance(box[0], int) or not 0 <= box[0] < len(classes):
                    raise ValueError(f"invalid box class/shape: {key}")
                if not all(isinstance(v, (int, float)) and float('-inf') < v < float('inf') for v in box):
                    raise ValueError(f"nonfinite box: {key}")
                if not (0 <= box[1] < box[3] and 0 <= box[2] < box[4]):
                    raise ValueError(f"invalid xyxy box: {key}")
    return data


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
