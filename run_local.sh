#!/usr/bin/env bash
#===========================================================================
# run_local.sh — prepara o ambiente e roda o treino na sua máquina, sem SLURM
#
# Faz tudo: instala uv se faltar, sincroniza deps, baixa/converte o dataset
# em data/ e dispara o treino, com log em logs/.
#
# Presets (mesma nomenclatura da matriz T1.1 do plano_de_testes.md):
#   ./run_local.sh              # C  — yolo26s @960, 150 épocas (candidato principal)
#   ./run_local.sh sanity       # B  — yolo26n @640,  50 épocas (~1h, valida o pipeline)
#   ./run_local.sh m960         # D  — yolo26m @960
#   ./run_local.sh p2           # E  — yolo26s-p2 @960 (head P2, objetos pequenos)
#   ./run_local.sh s1280        # G  — yolo26s @1280
#
# Só preparar, sem treinar:
#   ./run_local.sh --setup-only
#
# Qualquer coisa depois de `--` vai direto para o train_yolo26.py:
#   ./run_local.sh sanity -- --batch 8 --patience 10
#
# Overrides finos por env:
#   MODEL=yolo26s.pt IMGSZ=960 EPOCHS=150 NAME=meu_exp BATCH=8 ./run_local.sh
#===========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export HSL_DATA_DIR="${HSL_DATA_DIR:-$REPO_ROOT/data}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"

log()  { printf '\033[1;36m[local]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[local]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[local]\033[0m %s\n' "$*" >&2; exit 1; }

# ----- Argumentos ---------------------------------------------------------
PRESET="main"
SETUP_ONLY=0
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup-only) SETUP_ONLY=1; shift ;;
        --)           shift; EXTRA=("$@"); break ;;
        -h|--help)    sed -n '2,27p' "$0"; exit 0 ;;
        -*)           die "Flag desconhecida: $1 (flags do treino vão depois de --)" ;;
        *)            PRESET="$1"; shift ;;
    esac
done

# ----- Presets ------------------------------------------------------------
case "$PRESET" in
    main|c|C)        P_MODEL=yolo26s.pt      P_IMGSZ=960  P_EPOCHS=150 P_NAME=s960 ;;
    sanity|b|B)      P_MODEL=yolo26n.pt      P_IMGSZ=640  P_EPOCHS=50  P_NAME=sanity_n640 ;;
    m960|d|D)        P_MODEL=yolo26m.pt      P_IMGSZ=960  P_EPOCHS=150 P_NAME=m960 ;;
    p2|e|E)          P_MODEL=yolo26s-p2.yaml P_IMGSZ=960  P_EPOCHS=200 P_NAME=s960_p2 ;;
    s1280|g|G)       P_MODEL=yolo26s.pt      P_IMGSZ=1280 P_EPOCHS=150 P_NAME=s1280 ;;
    *) die "Preset desconhecido: '$PRESET' (use main|sanity|m960|p2|s1280)" ;;
esac

MODEL="${MODEL:-$P_MODEL}"
IMGSZ="${IMGSZ:-$P_IMGSZ}"
EPOCHS="${EPOCHS:-$P_EPOCHS}"
NAME="${NAME:-${P_NAME}_local}"
BATCH="${BATCH:--1}"          # -1 = auto-batch (negocia com a VRAM disponível)
# 'disk' e não 'ram': o dataset descomprimido não cabe na RAM de um desktop.
CACHE="${CACHE:-disk}"

#---------------------------------------------------------------------------
# 1. uv
#---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv fora do PATH — adicione ~/.local/bin ao PATH."
fi

mkdir -p logs runs "$HSL_DATA_DIR"

log "Sincronizando dependências (uv sync)..."
uv sync

#---------------------------------------------------------------------------
# 2. Checagem de GPU
#---------------------------------------------------------------------------
log "Ambiente:"
uv run --no-sync python - <<'PY'
import platform, sys, torch, ultralytics
print(f"    python       {platform.python_version()}")
print(f"    ultralytics  {ultralytics.__version__}")
print(f"    torch        {torch.__version__} (cuda {torch.version.cuda})")
if not torch.cuda.is_available():
    print("    \033[1;33mSEM GPU\033[0m — o treino vai cair para CPU e levar dias.")
    sys.exit(0)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    sm = f"sm_{p.major}{p.minor}"
    vram = p.total_memory / 2**30
    print(f"    gpu{i}         {p.name} | {vram:.1f} GiB | {sm}")
    if sm not in torch.cuda.get_arch_list():
        print(f"    \033[1;31mERRO\033[0m: este torch não tem kernels para {sm} "
              f"(tem: {' '.join(torch.cuda.get_arch_list())}).")
        print("    GPUs Blackwell (RTX 50xx, sm_120) precisam de torch com CUDA 12.8+.")
        print("    Apague o .venv e rode de novo: rm -rf .venv && ./run_local.sh --setup-only")
        sys.exit(1)
    if vram < 10:
        print(f"    \033[1;33mAviso\033[0m: {vram:.0f} GiB de VRAM. Com imgsz>=960 o "
              "auto-batch vai cair para ~8; se der OOM, use: -- --batch 4")
PY

#---------------------------------------------------------------------------
# 3. Dataset
#---------------------------------------------------------------------------
DATA_YAML="$HSL_DATA_DIR/hslvision_yolo/hslvision.yaml"
if [[ ! -f "$DATA_YAML" ]]; then
    log "Dataset ausente — baixando e convertendo o HSLVision para $HSL_DATA_DIR"
    log "(são ~6k imagens; a primeira vez demora e ocupa alguns GB)"
    uv run --no-sync python src/prepare_hslvision.py
fi
[[ -f "$DATA_YAML" ]] || die "Falhou preparar o dataset — veja o erro acima."
log "Dataset: $DATA_YAML"

if [[ $SETUP_ONLY -eq 1 ]]; then
    log "Ambiente pronto (--setup-only). Para treinar: ./run_local.sh $PRESET"
    exit 0
fi

#---------------------------------------------------------------------------
# 4. Treino
#---------------------------------------------------------------------------
RESUME=()
if [[ -f "runs/hslvision/${NAME}/weights/last.pt" ]]; then
    log "Checkpoint encontrado — retomando '${NAME}'."
    RESUME=(--resume)
fi

LOGFILE="logs/local_${NAME}_$(date +%Y%m%d_%H%M%S).log"
log "Preset '$PRESET' -> $MODEL @${IMGSZ}, ${EPOCHS} épocas, cache=$CACHE"
log "Log: $LOGFILE   (interrompeu? é só rodar de novo, ele retoma do last.pt)"
echo

set -o pipefail
uv run --no-sync python src/train_yolo26.py \
    --model  "$MODEL" \
    --imgsz  "$IMGSZ" \
    --epochs "$EPOCHS" \
    --batch  "$BATCH" \
    --name   "$NAME" \
    --cache  "$CACHE" \
    --export \
    "${RESUME[@]}" "${EXTRA[@]}" 2>&1 | tee "$LOGFILE"

echo
log "Métricas: runs/hslvision/${NAME}/results.csv"
log "Curvas/plots: runs/hslvision/${NAME}/"
