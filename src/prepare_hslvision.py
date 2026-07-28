#!/usr/bin/env python3
"""
prepare_hslvision.py
====================
Baixa o dataset HSLVision (whirlwind-ams/hslvision) do Hugging Face e o
converte para o layout YOLO esperado pela Ultralytics, gerando também o
arquivo `hslvision.yaml`.

Como o layout exato do repo pode variar, o script AUTO-DETECTA o formato:
  1. Layout YOLO nativo  (images/ + labels/ com .txt)  -> apenas organiza
  2. COCO JSON           (annotations/*.json)          -> converte
  3. HF parquet/arrow    (load_dataset com campo de bboxes) -> converte

Saída padrão: `data/hslvision_yolo/` na raiz do repo (ver src/paths.py).
Sobrescreva com --out, ou com HSL_DATA_DIR para mover a pasta data/ inteira
(no cluster, aponte para um scratch — o home é NFS e tem cota).

Uso — precisa de rede, então no Apuana rode na partição `install`:
    ./setup_env.sh --alloc --dataset       # cluster
    ./run_local.sh --setup-only            # máquina local
    uv run python src/prepare_hslvision.py # direto

Depois confira o resumo impresso no final e o hslvision.yaml gerado.
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import HSLVISION_DIR  # noqa: E402

# As 7 classes do HSLVision (ordem canônica do paper).
# ATENÇÃO: se o dataset trouxer seu próprio mapeamento (names no yaml ou
# categories no COCO json), o script usa o do dataset e só avisa se divergir.
HSL_CLASSES = [
    "ball",
    "goalpost",
    "robot",
    "l_intersection",
    "t_intersection",
    "x_intersection",
    "penalty_mark",
]

SPLITS = ("train", "val", "validation", "test")


def log(msg: str) -> None:
    print(f"[prepare_hslvision] {msg}", flush=True)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_snapshot(repo_id: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    log(f"Baixando snapshot de {repo_id} ...")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return Path(path)


# --------------------------------------------------------------------------
# Detecção de formato
# --------------------------------------------------------------------------
def detect_format(root: Path) -> str:
    txt_labels = list(root.rglob("labels/**/*.txt")) or list(root.rglob("labels/*.txt"))
    if txt_labels:
        return "yolo"

    coco_jsons = [
        p for p in root.rglob("*.json")
        if p.stat().st_size > 1024 and _looks_like_coco(p)
    ]
    if coco_jsons:
        return "coco"

    if list(root.rglob("*.parquet")) or list(root.rglob("*.arrow")):
        return "hf"

    return "unknown"


def _looks_like_coco(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(4096)
        return '"images"' in head and '"annotations"' in head
    except Exception:
        return False


# --------------------------------------------------------------------------
# Conversores
# --------------------------------------------------------------------------
def organize_yolo(root: Path, out: Path) -> dict:
    """Dataset já em formato YOLO: só espelha para o diretório de saída."""
    stats = defaultdict(int)
    for split in SPLITS:
        for kind in ("images", "labels"):
            candidates = [
                root / kind / split,
                root / split / kind,
            ]
            src = next((c for c in candidates if c.is_dir()), None)
            if src is None:
                continue
            dst = out / kind / _norm_split(split)
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
                    stats[f"{_norm_split(split)}/{kind}"] += 1
    return dict(stats)


def convert_coco(root: Path, out: Path) -> dict:
    """Converte COCO JSON -> YOLO txt (bbox normalizado cx cy w h)."""
    stats = defaultdict(int)
    coco_files = [p for p in root.rglob("*.json") if _looks_like_coco(p)]

    for coco_path in coco_files:
        split = _guess_split(coco_path.name) or _guess_split(str(coco_path.parent)) or "train"
        split = _norm_split(split)
        with open(coco_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        cats = sorted(coco["categories"], key=lambda c: c["id"])
        cat_names = [c["name"] for c in cats]
        if [n.lower().replace("-", "_").replace(" ", "_") for n in cat_names] != HSL_CLASSES:
            log(f"AVISO: classes do dataset {cat_names} != esperadas {HSL_CLASSES}. "
                "Usando a ordem do dataset — confira o yaml gerado.")
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

        images = {im["id"]: im for im in coco["images"]}
        anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)

        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id, im in images.items():
            file_name = im["file_name"]
            src_img = _find_image(root, file_name)
            if src_img is None:
                stats[f"{split}/missing_images"] += 1
                continue
            shutil.copy2(src_img, img_out / Path(file_name).name)

            w, h = im["width"], im["height"]
            lines = []
            for ann in anns_by_img.get(img_id, []):
                x, y, bw, bh = ann["bbox"]  # COCO: top-left x,y + w,h em pixels
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                lines.append(
                    f"{cat_id_to_idx[ann['category_id']]} "
                    f"{cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}"
                )
            (lbl_out / (Path(file_name).stem + ".txt")).write_text("\n".join(lines))
            stats[f"{split}/images"] += 1

        stats["_class_names"] = cat_names  # type: ignore[assignment]
    return dict(stats)


def convert_hf(repo_id: str, out: Path) -> dict:
    """Converte um dataset HF (parquet/arrow com campo de objetos) -> YOLO."""
    from datasets import load_dataset

    stats = defaultdict(int)
    ds_dict = load_dataset(repo_id)

    for split_name, ds in ds_dict.items():
        split = _norm_split(split_name)
        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        cols = ds.column_names
        # Campos comuns: "image", "objects" {"bbox"/"bboxes", "category"/"categories"}
        obj_col = next((c for c in ("objects", "annotations", "labels") if c in cols), None)
        img_col = next((c for c in ("image", "rgb", "img") if c in cols), None)
        if obj_col is None or img_col is None:
            log(f"ERRO: não achei colunas de imagem/objetos em {cols}. "
                "Ajuste convert_hf() para o schema real do dataset.")
            sys.exit(2)

        for i, ex in enumerate(ds):
            img = ex[img_col]
            w, h = img.size
            fname = f"{split}_{i:06d}.jpg"
            img.convert("RGB").save(img_out / fname, quality=95)

            objs = ex[obj_col]
            bboxes = objs.get("bbox") or objs.get("bboxes") or []
            categories = objs.get("category") or objs.get("categories") or []
            lines = []
            for bbox, cat in zip(bboxes, categories):
                x, y, bw, bh = bbox  # assumindo COCO-style; ajuste se for xyxy
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                lines.append(f"{int(cat)} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines))
            stats[f"{split}/images"] += 1
    return dict(stats)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _norm_split(s: str) -> str:
    s = s.lower()
    if "val" in s:
        return "val"
    if "test" in s:
        return "test"
    return "train"


def _guess_split(name: str) -> str | None:
    name = name.lower()
    for s in SPLITS:
        if s in name:
            return s
    return None


_IMG_INDEX: dict[str, Path] = {}


def _find_image(root: Path, file_name: str) -> Path | None:
    if not _IMG_INDEX:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in root.rglob(ext):
                # ignora depth maps se estiverem em pasta separada
                if "depth" in str(p).lower():
                    continue
                _IMG_INDEX[p.name] = p
    return _IMG_INDEX.get(Path(file_name).name)


def write_yaml(out: Path, class_names: list[str]) -> Path:
    yaml_path = out / "hslvision.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    test_line = "test: images/test\n" if (out / "images" / "test").is_dir() else ""
    yaml_path.write_text(
        f"# HSLVision — gerado por prepare_hslvision.py\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"{test_line}"
        f"names:\n{names_block}\n"
    )
    return yaml_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="whirlwind-ams/hslvision")
    ap.add_argument("--out", type=Path, default=HSLVISION_DIR,
                    help="Diretório de saída (default: data/hslvision_yolo/; "
                         "no cluster prefira um scratch via HSL_DATA_DIR)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="Onde salvar o snapshot bruto (default: <out>/_raw)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or (args.out / "_raw")

    root = download_snapshot(args.repo, cache)
    fmt = detect_format(root)
    log(f"Formato detectado: {fmt}")

    class_names = HSL_CLASSES
    if fmt == "yolo":
        stats = organize_yolo(root, args.out)
        # tenta reaproveitar um yaml existente para os nomes de classe
        existing = list(root.rglob("*.yaml"))
        if existing:
            log(f"YAMLs encontrados no repo (confira nomes de classe): "
                f"{[str(p) for p in existing]}")
    elif fmt == "coco":
        stats = convert_coco(root, args.out)
        class_names = stats.pop("_class_names", HSL_CLASSES)  # type: ignore[arg-type]
    elif fmt == "hf":
        stats = convert_hf(args.repo, args.out)
    else:
        log("Formato não reconhecido. Liste os arquivos do snapshot e adapte "
            "o script:\n  find <cache> | head -50")
        sys.exit(1)

    yaml_path = write_yaml(args.out, list(class_names))

    log("=" * 60)
    log("Resumo:")
    for k, v in sorted(stats.items()):
        log(f"  {k}: {v}")
    log(f"Dataset YAML: {yaml_path}")
    log("Valide visualmente algumas labels antes de treinar:")
    log("  yolo val model=yolo26n.pt data=<yaml> plots=True  (ou inspecione "
        "com fiftyone/labelImg)")


if __name__ == "__main__":
    main()
