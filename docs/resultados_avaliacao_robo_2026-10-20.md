# Resultados da avaliação offline no robô — 20/10/2026

## O que significam IoU e mIoU

**IoU (Intersection over Union)** de uma classe é a área de pixels corretamente previstos dessa classe dividida pela união de pixels previstos ou anotados: `TP / (TP + FP + FN)`. Uma IoU de 0 significa que nenhum pixel anotado da classe coincidiu com a predição; 1 significa correspondência perfeita. Os pixels anotados como `255` são ignorados. O runner calcula contagens agregadas sobre as 151 imagens, não uma média de IoU por imagem.

**mIoU** é a média aritmética das IoUs das classes com união não vazia. No primeiro teste há duas classes: `mIoU = (IoU_CircleLine + IoU_Line) / 2`. Como `CircleLine` teve IoU 0 em todas as variantes, cada mIoU nessa avaliação é metade da IoU de `Line`. No teste unificado há apenas `field_line`, então a mIoU registrada no CSV é simplesmente a IoU dessa classe. **Não comparar numericamente as mIoUs de um teste com as do outro:** os rótulos e o número de classes mudaram. Campos `map50` e `map50_95` vazios são esperados: foram avaliadas máscaras, não caixas de detecção.

## Ambiente, dados e configurações

- Robô `booster@192.168.0.102`, checkout `/home/booster/Workspace/hl_vision`, execução em 20/10/2026. Python 3.10.12 (a documentação recomenda >=3.11); testes sintéticos passaram mesmo assim. Compilação C++ com CMake/GCC e OpenCV de desenvolvimento 4.8.0 concluída; `hl_preprocess --help` funcionou.
- Dataset de validação manual NAODevils v36: **151 imagens de 640×480**, máscaras de mesmo tamanho. Na avaliação de duas classes, `0=CircleLine` (pixel 1) e `1=Line` (pixel 2); pixel 0 é fundo e 255 é ignorado. Na avaliação unificada, ambos os pixels 1 e 2 viraram 1=`field_line` em uma **cópia derivada**; fundo e pixels 255 foram preservados. O dataset original não foi modificado.
- Conferência pixel a pixel da cópia: **1.273.201** pixels `field_line`, **12.659** pixels ignorados em 28 imagens e três imagens sem pixels positivos (`00008`, `00037`, `00052`). O GT `CircleLine` existe em **68 imagens** do original.
- Modelos: `best_seg_orin_10.3.engine` (TensorRT, entrada 640×640, GPU, IDs 0 e 1 mapeados respectivamente para `CircleLine`/`Line` no primeiro teste e ambos para `field_line` no segundo); `sim_data_seg_0126.onnx` (ONNX Runtime com CUDA, entrada 320×320, única classe `field_line`, apenas segundo teste). O engine não contém nomes de classes embutidos: a ordem `CircleLine`/`Line` vem do código do `hsl-player` e precisa ser confirmada contra a proveniência do treinamento.
- Variantes `original`, `clahe`, `white`, `combined`; parâmetros compartilhados do pré-processador: `clip-limit=2`, `tile-size=8`, `min-value=140`, `max-saturation=110`, `strength=0,9`. Cada comparação usou `warmup=5`, `repeat=20`, `conf=0,001`, `iou=0,7`, `max_det=300`, `half=false`, mesma sequência de imagens e 3.020 amostras de latência por modelo/variante. As imagens das quatro variantes foram preparadas separadamente para cada manifesto com `warmup=3`, `repeat=20`.
- `mean_ms` e `p95_ms` medem a chamada Python de predição, sincronização CUDA e transferência das saídas para CPU; **não incluem pré-processamento compartilhado, leitura/gravação de imagens nem integração ROS**. Modelos têm resoluções e runtimes diferentes, e o robô ainda tinha câmera/serviços auxiliares ativos: os números de tempo não isolam o efeito do modelo/filtro nem representam o FPS do robô.

## Avaliação com duas classes: CircleLine e Line

Arquivo original: `runs/naodevils-comparison-20261020/comparison.csv` no robô e na cópia de Downloads. **`sim_data_seg_0126.onnx` não participou** desta avaliação, pois seus metadados declaram somente `field_line`.

| Modelo | Variante | IoU CircleLine | IoU Line | mIoU | Média (ms) | p95 (ms) |
|---|---|---:|---:|---:|---:|---:|
| best_seg_orin_10.3.engine | original | **0,0000** | 0,2826 | 0,1413 | 61,10 | 117,76 |
| best_seg_orin_10.3.engine | clahe | **0,0000** | 0,2400 | 0,1200 | 76,32 | 158,43 |
| best_seg_orin_10.3.engine | white | **0,0000** | 0,3080 | **0,1540** | **41,69** | **78,84** |
| best_seg_orin_10.3.engine | combined | **0,0000** | 0,2995 | 0,1497 | 43,94 | 80,17 |

Em todas as variantes `CircleLine` teve **TP=0, FP=0 e FN=296.048 pixels**: nenhuma saída foi registrada como classe 0. Isso **não demonstra ausência de marcação visual do círculo**: o modelo pode tê-lo previsto como `Line`. A semântica e a ordem de IDs desse engine não são provadas pelos seus metadados.

Contagens de predições do engine, todas reportadas sob o ID de saída **1**: `original=4.895`, `clahe=6.853`, `white=2.881`, `combined=3.141`. São detecções de instâncias, não pixels TP.

## Avaliação unificada: field_line

Arquivo original: `runs/field-line-comparison-20261020/comparison.csv` no robô e na cópia de Downloads. Para o engine, IDs de saída 0 **e** 1 foram unidos na mesma máscara `field_line`; para o ONNX, seu único ID 0 foi mapeado para `field_line`. Nesta tabela, `mIoU = IoU field_line`.

| Modelo | Variante | IoU = mIoU field_line | Média (ms) | p95 (ms) |
|---|---|---:|---:|---:|
| best_seg_orin_10.3.engine | original | 0,2665 | 60,29 | 115,88 |
| best_seg_orin_10.3.engine | clahe | 0,2472 | 75,35 | 154,46 |
| best_seg_orin_10.3.engine | white | 0,2751 | **38,63** | 76,12 |
| best_seg_orin_10.3.engine | combined | **0,2760** | 40,73 | **71,91** |
| sim_data_seg_0126.onnx | original | 0,1240 | 45,54 | 83,38 |
| sim_data_seg_0126.onnx | clahe | **0,1351** | 61,56 | 101,78 |
| sim_data_seg_0126.onnx | white | 0,1022 | **41,71** | **76,66** |
| sim_data_seg_0126.onnx | combined | 0,1146 | 57,12 | 98,25 |

## Microbenchmark do preparo C++ (separado da inferência)

Cada variante teve 151 imagens × 20 amostras da chamada `LinePreprocessor::Apply` por preparo. O preparo para a avaliação unificada gerou **as mesmas imagens** (hashes idênticos) do preparo original; os tempos mudaram entre as duas execuções. Medidas abaixo não incluem inicialização, leitura, escrita nem predição e não equivalem à latência do nó ROS. **Não some médias ou percentis desses testes independentes para inferir latência ponta a ponta.**

| Variante | Preparo original: média Apply (ms) | Preparo field_line: média Apply (ms) |
|---|---:|---:|
| original | 0,144 | 0,107 |
| clahe | 5,723 | 1,977 |
| white | 7,257 | 4,018 |
| combined | 13,302 | 6,107 |

## Interpretação e limitações

- **Para uma única classe `field_line`,** `best_seg_orin_10.3.engine` teve IoU maior que `sim_data_seg_0126.onnx` nas quatro variantes desse dataset. `combined` teve a maior IoU do engine (0,2760), mas a diferença para `white` (0,2751) é só 0,0010; `white` teve média de predição menor nesta execução. No ONNX, `clahe` teve sua maior IoU (0,1351).
- **Para distinguir `CircleLine` e `Line`,** nenhuma variante do engine acertou um pixel de `CircleLine` como classe 0 sob o mapeamento testado. O ONNX é de uma classe só; não pode cumprir essa tarefa. É preciso confirmar taxonomia e examinar visualmente as predições antes de afirmar por que houve zeros.
- Escolher variante apenas por décimos de milésimo de IoU em um único dataset e em uma única ordem de execução seria precipitado. A qualidade se refere a este conjunto NAODevils, não a frames capturados pela câmera do robô; não houve avaliação de continuidade geométrica ou localização no campo. Cronometragens entre variantes sofreram possível influência de ordem, temperatura e carga de serviços em execução.
- **Diferença importante frente ao pipeline ROS:** o CLI offline aplica CLAHE/realce à imagem inteira de 640×480; o TensorRT do `hsl-player` faz primeiro um recorte central quadrado, depois aplica o mesmo filtro e redimensiona. O ONNX nativo do `hsl-player` também tem resize/padding próprios. Assim, o filtro compartilhado não garante que as variantes offline reproduzam exatamente os pixels ou o desempenho operacional do robô. A configuração atual do `src/vision/config/vision.yaml` inspecionada na auditoria equivale a `white` (CLAHE desligado, realce branco ligado), mas isso é uma configuração mutável e não prova qual YAML foi usado durante avaliações offline.

## Auditoria do código e da execução

- Pré-processador C++ auditado: `original` desativa ambas as etapas e clona a imagem; `clahe` equaliza L em Lab; `white` limiariza baixo S/alto V no HSV, suaviza a máscara com Gaussiana 5×5 e mistura com branco; `combined` executa CLAHE **antes** do realce branco. `clip-limit=2`, grid 8×8, `min-value=140`, `max-saturation=110` e `strength=0,9` coincidem com o YAML inspecionado, mas os parâmetros offline são fornecidos por arquivo separado e não acompanham alterações futuras do YAML automaticamente. Na via TensorRT 10.3 não foi encontrada dupla aplicação de CLAHE.
- Atenção: o runner `run` usa o frame original diretamente quando `variant=original`, em vez do PNG `original` gerado por `prepare`. A checagem inicial confirmou igualdade pixel a pixel nos 151 frames do dataset original; o tempo do clone do pré-processamento não é incluído na medição de inferência.
- Revisão independente de `evaluate.py` e `metrics.py`: as contagens TP/FP/FN são somadas sobre todos os pixels válidos do dataset (fundo não entra como classe); as máscaras de instâncias/IDs mapeadas para a mesma classe são combinadas por **OR**, não somadas em duplicidade. Conferência dos 151 frames × quatro variantes constatou que a saída unificada do engine é a união exata de suas duas máscaras do teste original. A qualidade usa apenas a **última** das 20 predições de cada imagem; as 20 entram somente na estatística de tempo.
- `comparison.json`/`.csv`, `run.json`, resumos, predições, máscaras e latências foram gerados para as 12 combinações; cada combinação teve 151 imagens e 3.020 medições. Hashes de pesos, imagens e máscaras registrados em `run.json` conferiram com os arquivos atuais no robô. **Conclusão operacional:** as execuções terminaram e geraram resultados completos. **Não equivale a “sem problemas”:** o log do teste unificado contém avisos do TensorRT sobre portabilidade do engine/ausência de metadata e do ONNX Runtime sobre inserção de nós `Memcpy` que podem afetar desempenho; a comparação offline difere da geometria de produção. A proveniência dos pesos e eventual sobreposição com as 151 imagens de validação não foram comprovadas.
- Checagem adicional de dados e resultados: 151 máscaras derivadas conferem pixel a pixel com a união das originais, mantendo 255; hashes originais conferem com a proveniência NAODevils, inclusive `annotations/manual_val.json`. Os **604 registros de preparo por avaliação** e os hashes das imagens preparadas conferem; as variantes produzidas nos dois preparos têm os mesmos hashes. Nas 12 combinações, há 151 arquivos de máscaras previstas, 151 IDs no `predictions.json` e 3.020 amostras em `latency.csv`; TP/FP/FN recalculados das máscaras salvas coincidem com `comparison.json`. A proveniência do treinamento continua desconhecida: não foi possível excluir vazamento de treino para validação.
- Antes da inferência, `python3 -m compileall -q tools tests` e `python3 -m unittest discover -s tests -v` passaram (**12/12 testes**); o CMake compilou o executável C++ compartilhado e `build/offline-eval/hl_preprocess --help` e `python3 tools/evaluate.py --help` funcionaram. São verificações funcionais, não validação de qualidade das segmentações.

## Arquivos e reprodução

- Artefatos completos no robô: `/home/booster/Workspace/hl_vision/runs/naodevils-comparison-20261020/` e `/home/booster/Workspace/hl_vision/runs/field-line-comparison-20261020/`. Cada combinação contém `summary.json`, `predictions.json`, `latency.csv` e máscaras de saída (`*_masks.npz`); cada avaliação contém `comparison.csv`, `comparison.json` e `run.json`.
- Variantes PNG e medições de preparo: `runs/naodevils-prepared-20261019/` e `runs/field-line-prepared-20261020/`. O conjunto derivado e sua proveniência estão em `runs/field-line-dataset-20261020/`; configuração dos modelos unificados em `runs/field-line-models-20261020.json` e log em `runs/field-line-eval-20261020.log`.
- Cópia local dos resultados, imagens preparadas, configurações e dados de entrada: `/home/evrc/Downloads/hl_vision_avaliacao_2026-10-20/` (arquivos `runs/`, `Data/`, `config/evaluation/models.local.json`). Relatório também neste arquivo do repositório local. Atenção: manifestos/`run.json` registram caminhos absolutos do robô; para repetir a execução no PC é necessário adaptar esses caminhos e ter os backends e pesos compatíveis.
