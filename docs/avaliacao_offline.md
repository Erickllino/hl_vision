# Avaliação offline no robô (execução posterior)

Dataset de duas classes: veja [preparação CircleLine / Line](dataset_circleline_line.md)
para obter o NAODevils manual já convertido ao formato deste runner.

Ferramentas para comparar modelos locais e `original`, `clahe`, `white`, `combined` nas mesmas imagens. Não treinam nem baixam pesos. Nenhum benchmark real foi executado nesta implementação. Exemplos em `config/evaluation/` são templates; substitua caminhos, classes, shapes e anotações pelos dados reais.

## Dependências e compilação

Python >=3.11, NumPy para métricas/testes; para inferência: OpenCV Python, Ultralytics e PyTorch compatível com o robô. ONNX requer ONNX Runtime (CPU ou GPU compatível); `.engine` requer TensorRT/CUDA/JetPack compatíveis. Instale previamente no robô; `YOLO_AUTOINSTALL=false` desativa instalação automática durante avaliação. Use o ambiente existente ou um ambiente isolado específico do robô. Não execute `run_local.sh` para avaliar: é o pipeline legado de treino.

O executável requer CMake >=3.16, compilador C++17 e **OpenCV de desenvolvimento** (Ubuntu: `libopencv-dev`). A wheel Python não fornece os headers/CMake necessários.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DHSL_PLAYER_ROOT=/home/yienvee/Documents/hsl-player
cmake --build build --parallel
build/hl_preprocess --help
python tools/evaluate.py --help
```

O CMake compila diretamente `src/vision/src/model/line_preprocessor.cpp` e inclui `src/vision/include` do checkout indicado. Não compila ROS/YAML/TensorRT e não copia o algoritmo. API consumida: `booster_vision::LinePreprocessorConfig`, construtor `LinePreprocessor(config)` e `cv::Mat Apply(const cv::Mat& bgr)`. Os campos utilizados são `clahe_enabled`, `clip_limit`, `tile_grid_size`, `use_white_enhancement`, `min_value`, `max_saturation`, `strength`. O CLI oferece grid quadrado (`--tile-size`).

## Dados e rótulos explícitos

Manifesto JSON (caminhos relativos ao próprio manifesto, ou absolutos):

```json
{"classes":["CircleLine","Line"],"images":[
  {"id":"session01_frame0001","image":"images/frame0001.png",
   "boxes":[[1,20,30,160,90]],"mask":"masks/frame0001.png"}
]}
```

- IDs únicos com letras ASCII, números, `_` ou `-`. Imagens devem ser frames originais sem resize/crop prévio diferente entre variantes.
- `--labels boxes`: cada imagem deve ter `boxes`, inclusive `[]` para negativos confirmados. Formato `[class_id,x1,y1,x2,y2]`, pixels na imagem original, coordenadas contínuas xyxy, IDs de classe base zero. Sem crowd/ignore/caixas orientadas.
- `--labels masks`: cada imagem deve ter máscara PNG **grayscale uint8** no tamanho original: `0` fundo, `1..N` classe do manifesto +1, `255` ignorar. Máscara RGB colorida não é aceita. Rótulos de instância/polígonos YOLO precisam ser convertidos explicitamente antes.
- `--labels none`: nenhuma métrica de qualidade; apenas predições e latência. Não assume que ausência de rótulo significa imagem negativa.
- `boxes` e `mask` podem coexistir; só o modo solicitado é usado. O exemplo contém coordenadas fictícias, não anotações de dados existentes.

Separe sessões/condições (sol, sombra, reflexos, bola, robôs) em manifestos fixos. Não ajuste thresholds no conjunto final. A ferramenta não faz split nem captura/sincroniza câmera.

## Preparar quatro variantes

Execute **posteriormente**, quando os frames estiverem definidos:

```bash
python tools/evaluate.py prepare \
  --manifest config/evaluation/dataset.local.json \
  --binary build/hl_preprocess --params config/evaluation/preprocess.json \
  --output runs/eval-prepared --warmup 3 --repeat 20
```

Salva PNG por ID/variante e `preparation.json` com hashes, parâmetros efetivos e amostras `Apply` em ms. Diretório de saída deve ser novo. Cada processo carrega uma imagem e reutiliza uma instância para warmup/repetições; startup, decode, encode e criação da instância ficam fora da medida. Isso é microbenchmark offline, não latência do loop contínuo do robô. `original` chama `Apply` com ambos os estágios desligados (cópia exata de pixels).

A API compartilhada não expõe máscara HSV; **não há máscara de candidatos exportada**. Nenhuma versão duplicada do algoritmo foi criada para produzi-la. Imagens das quatro pastas podem ser comparadas lado a lado externamente.

## Modelos e execução futura

Copie `models.example.json` para `models.local.json`. Adicione uma entrada por modelo, com nome único, arquivo local, `task=detect|segment`, `imgsz=[altura,largura]`, `device="cpu"` ou índice CUDA `"0"`, `half`, `conf`, `iou` (NMS), `max_det`. `class_map` traduz **cada ID produzido pelo modelo** para os IDs do manifesto; saída não mapeada aborta. Para avaliar bola/robôs, use manifesto e mapeamento próprios. Não compare médias de taxonomias diferentes.

```bash
# Somente valida schema/caminhos/hashes; não importa backend nem carrega pesos.
python tools/evaluate.py run --manifest config/evaluation/dataset.local.json \
  --models config/evaluation/models.local.json --labels masks \
  --output runs/eval-check --validate-only

# Executar no robô somente quando for hora da avaliação:
python tools/evaluate.py run --manifest config/evaluation/dataset.local.json \
  --models config/evaluation/models.local.json --labels masks \
  --prepared runs/eval-prepared --variants original clahe white combined \
  --output runs/eval-comparison --warmup 5 --repeat 20
```

O primeiro comando não valida bytes de imagens, semântica dos pesos ou compatibilidade do runtime; esses checks acontecem na execução. Para detecção, use `--labels boxes`; para dados não anotados, `--labels none`. A variante padrão é somente `original`, sem dependência do executável C++.

### Backends e diferenças do hsl-player

Usa o backend **Ultralytics Python** para artifacts YOLO compatíveis `.pt`, `.onnx`, `.engine`. Não é um interpretador genérico de qualquer ONNX nem um backend de segmentação semântica arbitrário. Engines customizados do hsl-player (plugins, bindings, layouts tensorrtx, ausência de metadata Ultralytics) podem ser incompatíveis; não são implicitamente convertidos. Exporte um artifact compatível ou implemente um runner nativo separado antes de comparar esses engines.

Inspeção somente leitura: `model/segmentor.h`, `detector.h`, `src/model/segmentor.cc`, `model/onnx/segmentation_impl.h`, `vision_node.cpp`. As interfaces nativas `Inference(cv::Mat)` são ligadas a factories YAML e implementações ONNX/TensorRT próprias; não são reutilizadas aqui. Letterbox, RGB/BGR interno, normalização, NMS, interpolação das máscaras e transfers do runner Python podem diferir do runtime C++. Reuso exato garantido aqui é apenas o **preprocessamento fotométrico**. Confirmar equivalência de outputs antes de usar o ranking como decisão de implantação.

O nó usa pose sincronizada, intrínsecos/extrínsecos e `FieldMarkerPoseEstimator` após segmentação. Sem esses dados não há avaliação de segmentos em coordenadas de campo, continuidade geométrica ou qualidade de localização. IoU/Dice de máscaras não substituem esse teste.

## Resultados e interpretação

- `comparison.json`/`.csv`: modelo × variante, métricas e latência.
- Por modelo/variante: `predictions.json` (caixas `[class,x1,y1,x2,y2,score]`), `*_masks.npz` (segmentação, boolean `masks[C,H,W]`, união das instâncias por classe), `summary.json`, `latency.csv` por frame/repetição.
- `run.json`: configurações, manifesto, hashes de pesos/manifesto, versões Python/NumPy/Torch/Ultralytics/OpenCV, CUDA e plataforma. Preparações são verificadas por hash para impedir uso acidental de frames alterados.
- Boxes: AP50 e média AP nas IoU `.50:.05:.95`, interpolação de 101 pontos, matching guloso por confiança e classe, um GT no máximo por predição. TP/FP/FN, precisão e recall em IoU .5 no corte `conf`. Não equivale integralmente a COCOeval (sem crowd, escalas ou maxDets de COCO). Classes sem GT têm AP `null` e ficam fora da média, mas seus FP são registrados. Use `conf` baixo para AP; AP fica truncado pelo corte e `max_det` configurados.
- Masks: IoU/Dice/precisão/recall por classe, pixels agregados no dataset, mIoU das classes com união não vazia. Sem GT e sem predição: `null`; predição em classe ausente: IoU zero. Ignora pixels 255. Uniões de classes podem sobrepor; isso é avaliação binária por classe de máscaras de instância, não instance AP.
- Repetições só aumentam amostras de tempo; qualidade usa a última predição de cada imagem (não conta GT várias vezes).
- Latência `predict_wall_ms`: chamada Python completa + transferência das saídas para CPU, sincronização CUDA, sem leitura/gravação, métricas ou preprocessamento compartilhado. Resumo mean/p50/p95/p99. Warmup usa a primeira imagem de cada modelo/variante. Não somar percentis independentes como se fossem latência end-to-end; FPS real do ROS exige instrumentação no nó.

Registre também modo de energia/clocks/temperatura do robô e repita execuções em ordem alternada para evitar viés térmico. Nenhuma melhoria de detecção foi afirmada.

## Verificações locais sem pesos

```bash
python -m compileall -q tools tests
python -m unittest discover -s tests -v
```

Testes sintéticos cobrem matching, duplicatas, falso positivo com score maior, classes erradas/ausentes, pixels ignorados, negativos explícitos e validação CLI sem inferência. Compilação C++ exige OpenCV dev; no host `pkg-config opencv4` não está disponível. O build integrado de `hl_preprocess` e sua chamada `--help` passaram em container local `hl-robot:humble`, com GCC 11.4 e OpenCV 4.5.4, compilando diretamente o componente do `hsl-player`. Execução com imagens/modelos e validação no robô ficam para a etapa posterior.
