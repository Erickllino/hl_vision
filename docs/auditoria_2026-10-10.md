# Auditoria local — 2026-10-10

Escopo: repositório `hl_vision` no commit `8118565`, arquivos locais de modelos/dados e cópia irmã `../hsl-player`. Resultados abaixo foram obtidos por leitura de código, metadados, Parquets, cabeçalhos de imagem e pequenos ensaios em memória. **Não são resultados de treino ou de desempenho de inferência.**

Plano vigente: [plano_de_testes.md](../plano_de_testes.md). Pesquisa externa: [datasets Booster T1/HSL](datasets_booster_t1_hsl.md).

## Ambiente observado

| Item | Valor |
|---|---|
| Python | 3.13.12 |
| Ultralytics | 8.4.109 |
| PyTorch / torchvision | 2.13.0 / 0.28.0 |
| CUDA do PyTorch | 13.0 |
| ONNX / ONNX Runtime | 1.22.0 / 1.31.0 |
| datasets / pyarrow / Pillow | 5.0.1 / 25.0.0 / 12.3.0 |
| TensorRT | Não instalado localmente |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Memória reportada por nvidia-smi | 16.311 MiB |
| Driver | 595.91.07 |
| CUDA disponível no PyTorch | Sim; build inclui `sm_120` |

Versões principais do `uv.lock` coincidem com o ambiente observado. `onnxruntime` está instalado, mas não consta como dependência explícita no `pyproject.toml` auditado. Não foi confirmada a disponibilidade de provider CUDA do ONNX Runtime. O hardware local não é o hardware de implantação.

## Estado do repositório

- `runs/`: sem resultados de treinamento encontrados.
- `logs/`: apenas `.gitkeep`.
- Dados: Parquets brutos em `data/hslvision_yolo/_raw/data/`; YAML/dataset YOLO convertido não encontrado.
- `.env`, `Models/` e `src/Test/` já eram arquivos locais não rastreados antes desta documentação; não foram modificados.

### Pontos do pipeline a corrigir

- `src/prepare_hslvision.py`: classe HSL de origem é gravada diretamente; YAML gerado usa sete nomes em ordem incompatível. O caminho HF recodifica imagens em JPEG e substitui proveniência por nomes sequenciais.
- A ordem real HSLVision é `background, Ball, Goalpost, K1, L-Intersections, Penalty Marks, T-Intersections, X-Intersections`, IDs 0–7, com objetos nos IDs 1–7.
- O caminho YOLO apenas informa a existência de YAML original, sem adotar seu mapeamento. Conversão precisa validar o schema e os IDs, não adivinhar.
- `src/train_yolo26.py`: `--imgsz` inteiro; exportação herda tamanho escalar, sem garantia do retângulo desejado.
- `model.val(split='test')` ao final de cada treino expõe teste ao ciclo de seleção; exceção genérica mascara falhas.
- Resume chama `model.train(resume=True)` e retorna antes da avaliação/exportação posterior.
- `exist_ok=True` e nomes reaproveitados requerem validação de configuração para evitar misturar experimentos.
- Defaults ativos incluem mosaic 1, scale 0,5 e optimizer auto; a justificativa de que mosaic ajuda bola pequena não foi medida.
- `run_local.sh` e `train_yolo26.sbatch` ainda oferecem os presets antigos. Não tratá-los como implementação da matriz revisada.
- `src/Test/run_model.py` é visualizador, não avaliador temporal. Default é engine de segmentação com `task=detect` e tamanho 320.

## Modelos disponíveis

Seis engines são binários reais, não ponteiros Git LFS:

| Arquivo em `Models/` | Bytes |
|---|---:|
| `best_digua_second_10.3.engine` | 25.281.156 |
| `best_digua_10.3.engine` | 25.369.092 |
| `best_digua_1223_10.3.engine` | 25.859.924 |
| `best_seg_orin_10.3.engine` | 27.018.900 |
| `best_seg_orin.engine` | 27.283.116 |
| `k1_realsense_10.3.engine` | 26.101.404 |

Começam com bytes `ftrt`, sem cabeçalho JSON Ultralytics identificado. Não foram desserializados. Variante, bindings e precisão interna continuam desconhecidos; tamanho do arquivo não identifica n/s/m/l/x.

| ONNX | Entrada | Saída | Metadados |
|---|---|---|---|
| `sim_data_det_0126.onnx` | FP32 `[1,3,320,320]` | FP32 `[1,13,2100]` | Nove classes; variante não identificada |
| `sim_data_seg_0126.onnx` | FP32 `[1,3,320,320]` | FP32 `[1,37,2100]`, `[1,32,80,80]` | Declara YOLOv8n-seg; apenas `field_line` |

Ambos: opset 19, exportador PyTorch 2.5.1, metadado Ultralytics 8.3.62, sem nó `NonMaxSuppression`. Esses ONNX de simulação não demonstram equivalência com os engines padrão de produção.

## HSLVision: contagens medidas

Método: leitura de todas as linhas Parquet com PyArrow; caixas COCO `x,y,w,h` e dimensões declaradas. As 65.309 áreas correspondem a `w*h` dentro de tolerância numérica. Cabeçalhos das 6.068 imagens verificados com Pillow. Hash SHA-256 dos bytes codificados para duplicação exata; não foi feita comparação perceptual de todas as imagens.

| Split | Imagens | Sem anotação Ball | Sem objetos |
|---|---:|---:|---:|
| train | 4.247 | 2.295 | 75 |
| valid | 910 | 489 | 22 |
| test | 911 | 482 | 27 |
| Total | 6.068 | 3.266 | 124 |

Não houve divergência de dimensões nos cabeçalhos, nomes repetidos entre splits ou hashes idênticos de imagens. Isso não exclui quase duplicatas ou vazamento temporal. Sem anotação de bola não equivale a negativo validado visualmente.

| Resolução original W × H | Imagens |
|---|---:|
| 1280 × 720 | 2.945 |
| 1280 × 1088 | 1.166 |
| 544 × 448 | 935 |
| 1280 × 800 | 874 |
| 1920 × 1080 | 80 |
| 640 × 360 | 40 |
| 800 × 600 | 28 |

Faixas de `min(w,h)` original, antes de clipping/resize:

| Classe | Total | Mediana W × H em px | <8 | 8–<16 | 16–<32 | >=32 |
|---|---:|---|---:|---:|---:|---:|
| Ball | 3.368 | 34,69 × 34,25 | 5 | 340 | 1.245 | 1.778 |
| Goalpost | 7.084 | 25,34 × 123,63 | 124 | 1.590 | 2.376 | 2.994 |
| K1 | 14.996 | 47,30 × 100,64 | 0 | 719 | 3.321 | 10.956 |
| L-Intersections | 17.918 | 33,03 × 18,05 | 1.004 | 6.549 | 6.224 | 4.141 |
| Penalty Marks | 3.820 | 26,03 × 13,22 | 734 | 1.784 | 856 | 446 |
| T-Intersections | 13.117 | 39,00 × 16,17 | 1.256 | 5.165 | 4.175 | 2.521 |
| X-Intersections | 5.006 | 41,95 × 25,09 | 3 | 637 | 2.859 | 1.507 |

Nas imagens 1280 × 720 há 887 bolas, das quais 107 têm menor lado <16 px. A redução atual para conteúdo útil 640 × 360 leva essas caixas a menos de 8 px no tensor.

Encontradas 2.205 caixas com `x<0` ou `y<0`, ou ultrapassagem de mais de 1 px à direita/embaixo: 2.130 têm coordenada inicial negativa e 90 ultrapassam limites, com sobreposição entre os critérios. Nenhuma largura/altura não positiva. Rever clipping e convenção de truncamento; não concluir automaticamente que são erros humanos.

Os 16 grupos inferidos por nomes aparecem em train/valid/test. Exemplo: `salvador-g5-booster27-2` tem 711/148/141 imagens. Uma revisão por sessão é necessária.

Não foram auditadas visualmente todas as anotações. Iluminação, oclusão e blur ainda precisam de amostragem estruturada. A documentação do dataset informa filtragem dessas condições difíceis e depth estimado por Depth Anything V3, não medido independentemente.

## Verificação em memória do pré-processamento

Executados os componentes `check_imgsz`, `LetterBox` e `BaseDataset.set_rectangle` extraídos da versão instalada, em memória e sem inicializar treino.

- `check_imgsz([736,1280], max_dim=1)` retorna 1280.
- Caminho de exportação com `min_dim=2` preserva `[736,1280]`.
- Batches sintéticos de proporções reais, `pad=0`, stride 32: `imgsz=960` gera `[544,960]`, `[800,960]`, `[832,960]`; `imgsz=1280` gera `[736,1280]`, `[1056,1280]`, `[1088,1280]`.
- `LetterBox(auto=False, scale_fill=False, center=False, padding_value=114)` transforma imagem 1280 × 720 em conteúdo 640 × 360, 960 × 540 ou 1280 × 720, nos canvases propostos.

São testes de funções e cálculo de shapes, não batches reais de treinamento nem artefatos exportados. Ambos permanecem como verificação obrigatória antes dos pilotos.

## Integração C++ confirmada

Referência: `../hsl-player/src/vision/src/model/trt/impl.cpp` e `../hsl-player/src/vision/config/vision.yaml`.

- `PreProcess` (linhas 463–495): RGB, FP32, `/255`, NCHW, padding 114 superior-esquerdo; assert 640/quadrado. Interpreta H como largura.
- `ProcessBoundingBox` (498–528): inteiros precoces e descarte <3 px.
- `LoadEngine` (621–651): consulta shapes, mas monta estrutura usando `d[3]` de saída possivelmente rank 3.
- Detector (655–720): 8.400 fixo, NMS 0,25/0,4 apesar dos thresholds YAML; NMS reúne classes.
- Segmentação (845–951): NMS fixo, 32 coeficientes assumidos, escala por `mask_width/1280.0`.
- ONNX: lê shape do modelo, porém usa padding preto; comparações precisam de pré-processamento unificado.

## Conclusões e limites

**Conclusão:** corrigir dados, splits, shapes e integração é pré-requisito para uma comparação confiável. **Hipótese:** resolução nativa e complemento de bolas pequenas podem melhorar recall. **Pendente:** qualquer medida de AP, recall, FP/min, latência, memória operacional e escolha de vencedor.

A pesquisa posterior identificou grupos T1 já presentes e alternativas públicas; suas contagens e fontes estão no documento de datasets. O holdout Abu Dhabi é declarado na página pública, mas sua disponibilidade local não foi confirmada.
