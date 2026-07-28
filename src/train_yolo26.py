#!/usr/bin/env python3
"""
train_yolo26.py
===============
Fine-tuning do YOLO26 no HSLVision (ou qualquer dataset em formato YOLO).

Pensado para rodar em cluster (SLURM) — sem estado interativo, tudo por
argumento, resume automático e saída organizada por experimento.

Por padrão lê o dataset de `data/hslvision_yolo/hslvision.yaml` e escreve em
`runs/hslvision/<name>/` (ver src/paths.py).

Exemplos:
    # Baseline rápido (sanidade)
    uv run python src/train_yolo26.py --model yolo26n.pt \
        --imgsz 640 --epochs 50 --name sanity_n640

    # Candidato principal (small + resolução alta)
    uv run python src/train_yolo26.py --model yolo26s.pt \
        --imgsz 960 --epochs 150 --name s960

    # Variante P2 (head extra para objetos pequenos; treina do zero a partir
    # do YAML de arquitetura, pois não há pesos -p2 pré-treinados)
    uv run python src/train_yolo26.py --model yolo26s-p2.yaml \
        --imgsz 960 --epochs 200 --name s960_p2

Na prática você não chama isto direto: use ./run_local.sh (máquina local) ou
sbatch train_yolo26.sbatch (cluster).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_PROJECT, HSLVISION_YAML  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo26s.pt",
                    help="Checkpoint .pt (fine-tune) ou .yaml de arquitetura "
                         "(treino do zero, ex.: yolo26s-p2.yaml)")
    ap.add_argument("--data", default=str(HSLVISION_YAML),
                    help="Caminho do dataset YAML (default: data/hslvision_yolo/"
                         "hslvision.yaml, gerado por prepare_hslvision.py)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="Resolução de entrada. Para bola distante/pênalti, "
                         "896–1280 costuma render mais que trocar de variante.")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=-1,
                    help="-1 = auto-batch (usa ~60%% da VRAM). Fixe manualmente "
                         "se treinar multi-GPU.")
    ap.add_argument("--device", default=None,
                    help="ex.: 0 | 0,1 | cpu. Default: todas as GPUs visíveis "
                         "(respeita CUDA_VISIBLE_DEVICES do SLURM).")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--project", default=str(DEFAULT_PROJECT),
                    help="Diretório-pai das saídas (default: runs/hslvision)")
    ap.add_argument("--name", required=True, help="Nome do experimento")
    ap.add_argument("--resume", action="store_true",
                    help="Retoma do last.pt deste experimento (para requeue "
                         "de jobs SLURM com timeout)")
    ap.add_argument("--patience", type=int, default=40,
                    help="Early stopping (épocas sem melhora)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=False,
                    help="False | 'ram' | 'disk'. 'ram' acelera muito se o nó "
                         "tiver memória (HSLVision é pequeno, ~6k imgs).")
    ap.add_argument("--export", action="store_true",
                    help="Ao final, exporta ONNX (para depois gerar o engine "
                         "TensorRT NA Jetson — nunca exporte TRT no cluster, "
                         "o engine é específico do hardware!)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"[train] Dataset YAML não encontrado: {args.data}\n"
            f"        Prepare-o antes:  uv run python src/prepare_hslvision.py\n"
            f"        (ou ./run_local.sh --setup-only / ./setup_env.sh --alloc --dataset)"
        )

    from ultralytics import YOLO

    # Resume: recarrega o last.pt do próprio experimento
    last = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"[train] Retomando de {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        return

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        exist_ok=True,
        # ------------------------------------------------------------------
        # Augmentations — defaults do YOLO26 são bons; os ajustes abaixo são
        # os que fazem sentido para futebol de robôs:
        # ------------------------------------------------------------------
        # O campo é simétrico à esquerda/direita -> fliplr ok (default 0.5).
        # flipud NÃO: o robô nunca vê o campo de cabeça pra baixo em jogo.
        flipud=0.0,
        # Mosaic ajuda objetos pequenos; desligar nas últimas épocas (default
        # close_mosaic=10) já é o comportamento padrão do YOLO26.
        # Cores do campo/bola importam para as classes -> HSV moderado:
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        # Perspectiva leve simula variação de pose da cabeça do robô:
        degrees=5.0, perspective=0.0005,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    # Validação final no split de teste, se existir no YAML
    print("[train] Validação final (split=test se disponível, senão val):")
    try:
        model.val(data=args.data, split="test", imgsz=args.imgsz)
    except Exception:
        model.val(data=args.data, imgsz=args.imgsz)

    if args.export:
        # ONNX aqui; o engine TensorRT deve ser gerado NA Jetson AGX Orin
        # (trtexec ou yolo export format=engine int8=True lá).
        path = model.export(format="onnx", imgsz=args.imgsz, dynamic=False,
                            simplify=True)
        print(f"[train] ONNX exportado: {path}")

    print(f"[train] Concluído. Resultados em {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
