# hl_vision

Treino e avaliação de detectores de objetos para a visão do RoboCIn Humanoid
(Booster T1). O dataset local é o **HSLVision**; complementos para bola pequena
e segmentação estão em avaliação.

## Planejamento e auditoria

- [Ferramentas de avaliação offline](docs/avaliacao_offline.md): comparação futura
  de modelos e quatro variantes de preprocessamento, sem treinamento; CLI em
  `tools/evaluate.py`, core C++ compartilhado com `hsl-player` via CMake.

- [Plano experimental](plano_de_testes.md): dados, resoluções retangulares,
  treinamento, avaliação no robô e contrato de integração com o hsl-player.
- [Pesquisa de datasets Booster T1/HSL](docs/datasets_booster_t1_hsl.md):
  fontes, links de acesso, contagens medidas, limitações e recomendação.
- [Auditoria local de 2026-10-10](docs/auditoria_2026-10-10.md): ambiente,
  classes, dados, pesos e problemas encontrados.

**Estado atual:** corrigir o remapeamento de classes do conversor HSLVision,
revisar os splits por sessão e implementar/verificar shapes retangulares antes
dos treinos longos. Os scripts e presets abaixo ainda são os do pipeline inicial;
não implementam integralmente o plano revisado. Não há benchmark de qualidade ou
latência no T1 que permita selecionar um vencedor.

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

## Pipeline inicial — execução local

```bash
./run_local.sh sanity     # preset existente: yolo26n, imgsz=640, 50 épocas
./run_local.sh            # preset existente: yolo26s, imgsz=960, 150 épocas
```

Da primeira vez ele instala as dependências e baixa/converte o dataset para
`data/`. Interrompeu no meio? Rode de novo: retoma do `last.pt`.

Presets legados: `sanity`, `main`, `m960`, `p2`, `s1280`.
Eles não correspondem aos IDs da matriz experimental revisada, e `imgsz`
escalar não garante entrada retangular. Flags extras passam depois de `--`:

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

O script solicita export ONNX ao fim de um treino novo; o caminho atual de
resume retorna antes dessa etapa. **Construir e validar TensorRT no ambiente-alvo
compatível do robô**, confirmando GPU, JetPack, CUDA e TensorRT. O modelo exato
da GPU do T1 ainda não foi verificado; os engines existentes não devem ser
presumidos portáveis para outra GPU/versão.

Documentação do cluster: <https://apuana.cin.ufpe.br> ·
[helpdesk](https://helpdesk.cin.ufpe.br/servicos/cluster-apuana) ·
`cluster.apuana-l@cin.ufpe.br`
