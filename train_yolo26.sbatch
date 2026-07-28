#!/bin/bash
#===========================================================================
# train_yolo26.sbatch — fine-tuning do YOLO26 no HSLVision (cluster Apuana)
#
# ANTES DA PRIMEIRA SUBMISSÃO, rode no login node:
#     ./setup_env.sh --alloc --dataset
# (o SLURM não cria o diretório de --output; sem logs/ o job morre na hora)
#
# Submissão:
#     sbatch train_yolo26.sbatch                     # exp C: yolo26s @960
#
#     # variantes: env vars vão pelo --export, e o -p troca a partição
#     sbatch -p short-simple --time=04:00:00 \
#         --export=ALL,MODEL=yolo26n.pt,IMGSZ=640,EPOCHS=50,NAME=sanity_n640 \
#         train_yolo26.sbatch
#
# Partições do Apuana: emergency, install, debug, short-simple, short-complex,
# long-simple, long-complex. `short-simple` tem a maior prioridade (100) e
# `long-complex` a menor (25); as `short` limitam 2 GPUs por job. Interativo
# (salloc/srun --pty) só nas partições `install` e `debug`.
#===========================================================================
#SBATCH --job-name=yolo26-hsl
#SBATCH --partition=long-simple     # 150 épocas não cabem numa short
#SBATCH --gpus=1                    # Apuana usa --gpus, não --gres
#SBATCH --cpus-per-task=8           # dataloader workers
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL,END
##SBATCH --mail-user=SEU_USER@cin.ufpe.br   # <-- descomente e preencha

set -euo pipefail

# sbatch preserva o diretório de submissão; nele estão logs/, src/ e data/.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

# ----- Parâmetros do experimento (via --export=ALL,VAR=valor) -------------
MODEL="${MODEL:-yolo26s.pt}"
IMGSZ="${IMGSZ:-960}"
EPOCHS="${EPOCHS:-150}"
NAME="${NAME:-s${IMGSZ}}"

# ----- Ambiente -----------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
export HSL_DATA_DIR="${HSL_DATA_DIR:-$REPO_ROOT/data}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"

command -v uv >/dev/null 2>&1 \
    || { echo "uv não encontrado. Rode ./setup_env.sh --alloc primeiro."; exit 1; }

# --no-sync: nada de rede/instalação num nó de compute — o ambiente já foi
# montado pelo setup_env.sh na partição `install`.
UV="uv run --no-sync"

DATA="${DATA:-$HSL_DATA_DIR/hslvision_yolo/hslvision.yaml}"
[[ -f "$DATA" ]] || {
    echo "Dataset não encontrado em $DATA."
    echo "Rode ./setup_env.sh --alloc --dataset (compute node não tem rede)."
    exit 1
}

echo "== Job $SLURM_JOB_ID | node $(hostname) | partição $SLURM_JOB_PARTITION =="
nvidia-smi
$UV python -c "import ultralytics, torch; print('ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ----- Treino (resume automático em caso de requeue/timeout) --------------
RESUME_FLAG=""
if [[ -f "runs/hslvision/${NAME}/weights/last.pt" ]]; then
    RESUME_FLAG="--resume"
    echo "Checkpoint encontrado — retomando ${NAME}."
fi

srun $UV python src/train_yolo26.py \
    --model  "$MODEL" \
    --data   "$DATA" \
    --imgsz  "$IMGSZ" \
    --epochs "$EPOCHS" \
    --name   "$NAME" \
    --cache  ram \
    --export \
    $RESUME_FLAG

echo "== Fim. Métricas: runs/hslvision/${NAME}/results.csv =="
