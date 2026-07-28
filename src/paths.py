"""
paths.py
========
Caminhos do repositório, resolvidos a partir da localização deste arquivo —
assim os scripts funcionam de qualquer diretório de trabalho (o SLURM roda o
job de `SLURM_SUBMIT_DIR`, que nem sempre é onde você acha).

Layout:
    <repo>/
        data/        datasets (fora do git)
        runs/        saídas de treino (fora do git)
        logs/        stdout/stderr dos jobs
        src/         código

`HSL_DATA_DIR` sobrescreve o diretório de dados — útil no cluster, para
apontar para um scratch em vez do home NFS (que tem cota).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("HSL_DATA_DIR", REPO_ROOT / "data"))
RUNS_DIR = Path(os.environ.get("HSL_RUNS_DIR", REPO_ROOT / "runs"))
LOGS_DIR = REPO_ROOT / "logs"

# Dataset padrão preparado por prepare_hslvision.py
HSLVISION_DIR = DATA_DIR / "hslvision_yolo"
HSLVISION_YAML = HSLVISION_DIR / "hslvision.yaml"

# Projeto default da Ultralytics: runs/hslvision/<name>/
DEFAULT_PROJECT = RUNS_DIR / "hslvision"
