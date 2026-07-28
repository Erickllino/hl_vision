#!/usr/bin/env bash
#===========================================================================
# setup_env.sh — prepara o ambiente no cluster Apuana antes do sbatch
#
# No Apuana só as partições `install` e `debug` aceitam execução interativa;
# a `install` existe justamente para montar ambiente (é onde há rede). Por
# isso o modo --alloc reexecuta este script dentro de um salloc nessa
# partição, em vez de instalar no login node.
#
# Para rodar na sua máquina, use ./run_local.sh (não precisa deste script).
#
# Uso:
#   ./setup_env.sh --alloc            # login node: pega salloc -p install
#   ./setup_env.sh --alloc --dataset  # idem + baixa/converte o HSLVision
#   ./setup_env.sh                    # prepara aqui (dentro de um salloc)
#   ./setup_env.sh --check            # só diagnostica, não instala
#===========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ----- Configuração (sobrescreva via env) ---------------------------------
INSTALL_PARTITION="${INSTALL_PARTITION:-install}"
ALLOC_CPUS="${ALLOC_CPUS:-8}"
ALLOC_MEM="${ALLOC_MEM:-32G}"
ALLOC_TIME="${ALLOC_TIME:-02:00:00}"
ALLOC_GPUS="${ALLOC_GPUS:-1}"     # 1 GPU só para validar torch.cuda no fim

# Onde ficam os dados. Se o cluster te der um scratch, aponte para lá — o
# home é NFS e tem cota; o dataset descomprimido passa de 10 GB.
#   HSL_DATA_DIR=/scratch/$USER/hl_vision_data ./setup_env.sh --alloc --dataset
export HSL_DATA_DIR="${HSL_DATA_DIR:-$REPO_ROOT/data}"

# Cache do uv fora do repo (as wheels do torch passam de 5 GB).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
# venv sempre no repo, para o sbatch achar sem adivinhação.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"

DO_ALLOC=0
DO_DATASET=0
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --alloc)   DO_ALLOC=1 ;;
        --dataset) DO_DATASET=1 ;;
        --check)   CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Argumento desconhecido: $arg" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

on_cluster() { command -v sinfo >/dev/null 2>&1; }
in_alloc()   { [[ -n "${SLURM_JOB_ID:-}" ]]; }

#---------------------------------------------------------------------------
# Modo --alloc: reexecuta este mesmo script dentro de uma alocação SLURM
#---------------------------------------------------------------------------
if [[ $DO_ALLOC -eq 1 ]]; then
    on_cluster || die "--alloc só funciona no cluster (sinfo não encontrado). Localmente use ./run_local.sh"
    in_alloc   && die "Você já está dentro da alocação $SLURM_JOB_ID. Rode sem --alloc."

    inner=()
    [[ $DO_DATASET -eq 1 ]] && inner+=(--dataset)
    [[ $CHECK_ONLY -eq 1 ]] && inner+=(--check)

    log "Alocando em '$INSTALL_PARTITION': ${ALLOC_CPUS} CPUs | ${ALLOC_MEM} | ${ALLOC_GPUS} GPU | ${ALLOC_TIME}"
    log "Se ficar em PD (pending), 'squeue -u $USER' mostra a fila."
    exec salloc \
        --partition="$INSTALL_PARTITION" \
        --cpus-per-task="$ALLOC_CPUS" \
        --mem="$ALLOC_MEM" \
        --gpus="$ALLOC_GPUS" \
        --time="$ALLOC_TIME" \
        --job-name=hlvision-setup \
        srun --pty "$REPO_ROOT/setup_env.sh" "${inner[@]}"
fi

#---------------------------------------------------------------------------
# Diagnóstico
#---------------------------------------------------------------------------
log "Repositório: $REPO_ROOT"
if on_cluster; then
    if in_alloc; then
        log "SLURM: job $SLURM_JOB_ID em $(hostname) (partição ${SLURM_JOB_PARTITION:-?})"
    else
        warn "Você está num login node. O certo no Apuana é: ./setup_env.sh --alloc"
    fi
    log "Partições visíveis:"
    sinfo -o '%-16P %-6a %-12l %-6D %G' | sed 's/^/    /'
else
    warn "SLURM não detectado — este script é para o cluster. Local: ./run_local.sh"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | sed 's/^/    GPU: /'
else
    warn "nvidia-smi ausente (normal em login node)."
fi

#---------------------------------------------------------------------------
# uv
#---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    [[ $CHECK_ONLY -eq 1 ]] && die "uv não instalado."
    log "Instalando uv (binário estático em ~/.local/bin, sem root)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 \
        || die "uv instalado mas fora do PATH — adicione ~/.local/bin ao PATH e rode de novo."
fi
log "uv $(uv --version | awk '{print $2}') | cache: $UV_CACHE_DIR"

#---------------------------------------------------------------------------
# Diretórios e dependências
#---------------------------------------------------------------------------
mkdir -p logs runs "$HSL_DATA_DIR"

if [[ $CHECK_ONLY -eq 0 ]]; then
    log "uv sync — a primeira vez baixa o torch (~3 GB), demora."
    uv sync
fi

log "Verificando a stack:"
uv run --no-sync python - <<'PY'
import platform, torch, ultralytics
print(f"    python       {platform.python_version()}")
print(f"    ultralytics  {ultralytics.__version__}")
print(f"    torch        {torch.__version__}")
print(f"    cuda         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    gpu{i}         {p.name} | {p.total_memory / 2**30:.1f} GiB | sm_{p.major}{p.minor}")
else:
    print("    (sem GPU aqui — esperado em login node ou alocação sem --gpus)")
PY

#---------------------------------------------------------------------------
# Dataset
#---------------------------------------------------------------------------
DATA_YAML="$HSL_DATA_DIR/hslvision_yolo/hslvision.yaml"
if [[ $DO_DATASET -eq 1 && $CHECK_ONLY -eq 0 && ! -f "$DATA_YAML" ]]; then
    log "Baixando e convertendo o HSLVision (precisa de rede — por isso a partição install)..."
    uv run --no-sync python src/prepare_hslvision.py
fi

if [[ -f "$DATA_YAML" ]]; then
    log "Dataset OK: $DATA_YAML"
    for s in train val test; do
        d="$HSL_DATA_DIR/hslvision_yolo/images/$s"
        [[ -d "$d" ]] && log "    $s: $(find "$d" -type f | wc -l) imagens"
    done
else
    warn "Dataset não preparado ainda. Rode: ./setup_env.sh --alloc --dataset"
fi

#---------------------------------------------------------------------------
# Próximo passo
#---------------------------------------------------------------------------
echo
log "Pronto. Saia da alocação (exit) e submeta do login node:"
cat <<EOF
    sbatch train_yolo26.sbatch                     # exp C: yolo26s @960, 150 épocas

    # sanidade rápida (exp B) — cabe na short-simple, que tem prioridade maior
    sbatch -p short-simple --time=04:00:00 \\
        --export=ALL,MODEL=yolo26n.pt,IMGSZ=640,EPOCHS=50,NAME=sanity_n640 \\
        train_yolo26.sbatch

    squeue -u $USER
    tail -f logs/yolo26-hsl_<jobid>.out
EOF
