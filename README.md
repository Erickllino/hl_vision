# hl_vision

Treino e avaliação de detectores de objetos para a visão do RoboCIn Humanoid
(Booster T1), no dataset **HSLVision**. O protocolo de decisão está em
[`plano_de_testes.md`](plano_de_testes.md).

## Layout

```
.
├── run_local.sh             # roda tudo na sua máquina (sem SLURM)
├── setup_env.sh             # prepara o ambiente no cluster Apuana
├── train_yolo26.sbatch      # job SLURM
├── src/
│   ├── paths.py             # caminhos do repo (data/, runs/, logs/)
│   ├── prepare_hslvision.py # baixa o HSLVision e converte p/ formato YOLO
│   └── train_yolo26.py      # fine-tuning
├── data/                    # datasets (fora do git)
├── runs/                    # saídas de treino (fora do git)
└── logs/                    # stdout/stderr dos jobs (fora do git)
```

Ambiente gerenciado por [uv](https://docs.astral.sh/uv/). Os scripts instalam
o uv se ele não existir — não é preciso `module load python` nem conda.

## Rodando localmente

```bash
./run_local.sh sanity     # yolo26n @640, 50 épocas — valida o pipeline em ~1h
./run_local.sh            # yolo26s @960, 150 épocas — candidato principal
```

Da primeira vez ele instala as dependências e baixa/converte o dataset para
`data/`. Interrompeu no meio? Rode de novo: retoma do `last.pt`.

Presets seguem a matriz T1.1 do plano: `sanity` (B), `main` (C), `m960` (D),
`p2` (E), `s1280` (G). Flags extras passam depois de `--`:

```bash
./run_local.sh main -- --batch 8 --patience 20
```

## Rodando no cluster Apuana (CIn-UFPE)

```bash
ssh <user>@apuana.cin.ufpe.br          # login node
git clone <este repo> && cd hl_vision

./setup_env.sh --alloc --dataset       # monta o ambiente e baixa os dados
exit                                   # sai da alocação
sbatch train_yolo26.sbatch             # submete o treino
squeue -u $USER
tail -f logs/yolo26-hsl_<jobid>.out
```

Por que `--alloc`: no Apuana **só as partições `install` e `debug` aceitam
execução interativa**, e é na `install` que há rede para baixar torch e
dataset. O script faz o `salloc -p install` e se reexecuta lá dentro.

### Partições

| Partição | Uso |
|---|---|
| `short-simple` | maior prioridade (100); bom para os experimentos curtos |
| `short-complex` | idem, recursos maiores; as `short` limitam 2 GPUs/job |
| `long-simple` | default do sbatch — 150 épocas não cabem numa `short` |
| `long-complex` | menor prioridade (25) |
| `install`, `debug` | únicas interativas |
| `emergency` | reservada |

GPU se pede com `--gpus=N` (não `--gres`). Trocar partição na submissão:

```bash
sbatch -p short-simple --time=04:00:00 \
    --export=ALL,MODEL=yolo26n.pt,IMGSZ=640,EPOCHS=50,NAME=sanity_n640 \
    train_yolo26.sbatch
```

O cluster tem 11 RTX 3090 e 5 A100 em 10 nós. Quanto mais recurso você pede,
menor a prioridade — 1 GPU basta para este dataset (~6k imagens).

### Dados fora do home

O home é NFS com cota e o dataset descomprimido passa de 10 GB. Se houver
scratch disponível:

```bash
export HSL_DATA_DIR=/scratch/$USER/hl_vision_data
./setup_env.sh --alloc --dataset
sbatch train_yolo26.sbatch     # o sbatch respeita a mesma variável
```

## Notas

O `--cache ram` do sbatch depende dos 48 G reservados no job; localmente o
`run_local.sh` usa `--cache disk`, porque o dataset descomprimido não cabe na
RAM de um desktop.

O export ONNX acontece no fim do treino. **O engine TensorRT deve ser gerado
na própria Jetson AGX Orin** — engines não são portáveis entre GPUs (Fase 3
do plano de testes).

Documentação do cluster: <https://apuana.cin.ufpe.br> ·
[helpdesk](https://helpdesk.cin.ufpe.br/servicos/cluster-apuana) ·
`cluster.apuana-l@cin.ufpe.br`
