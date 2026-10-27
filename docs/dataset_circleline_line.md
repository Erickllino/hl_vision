# Avaliação de linhas (classe `line`)

## Conjunto preparado

NAODevils **versão 36**, somente `annotations/manual_val.json`: 151 imagens
manualmente anotadas da partida `RoboCup2019_Team-Team`. Não inclui pseudo-rótulos.
O split de validação publicado é mantido inteiro, sem sorteio por frame. Sua
independência dos pesos avaliados depende da proveniência do treinamento.

Fonte: https://www.kaggle.com/datasets/pietbroemmel/naodevils-segmentation-upper-camera

Autor: Piet Broemmel / NAODevils. Licença declarada: **CC BY 4.0**.
As máscaras aqui são adaptações rasterizadas/remapeadas das anotações originais.

| Origem | Classe do manifesto / ID do modelo-alvo | Pixel na máscara |
|---|---|---|
| `centercircle` | `line` / 0 | 1 |
| `line` | `line` / 0 | 1 |
| fundo, bola, robô, gol | fundo | 0 |
| `penaltycross` | ignorar | 255 |

A atribuição `centercircle → line` une a pintura do círculo central às linhas de
campo, inclusive nas regiões de sobreposição. Marca de pênalti não é classificada
arbitrariamente como linha. Objetos anotados têm prioridade sobre tinta ocluída. Polígonos usam
arredondamento e `cv2.fillPoly`, seguindo o exemplo oficial. Imagens e máscaras
permanecem em **640×480**, sem crop, resize, realce ou augmentation.
Contornos originais com um ou dois pontos são preservados: representam também
pixels e trechos finos de tinta e são suportados pelo rasterizador do autor.

## Arquivos e reprodução

```bash
cd /home/yienvee/Documents/hl_vision
.venv/bin/python tools/prepare_naodevils.py
```

Baixa apenas o JSON manual de validação e suas 151 imagens, pela API pública do
Kaggle, com versão fixada. Downloads completos são reutilizados; arquivos parciais
não são considerados completos. Reexecutar regenera as máscaras e o manifesto.

- `Data/raw/naodevils-v36/`: JSON original e imagens.
- `Data/eval/naodevils-v36-manual-val/dataset.json`: manifesto pronto para o runner.
- `masks/`: PNGs grayscale uint8.
- `overlays/`: uma sobreposição por imagem; verde = line, laranja = ignorar.
- `contact_sheet.jpg`: amostra distribuída pelo conjunto, para inspeção visual.
- `provenance.json`: versão, mapeamento, hashes, contagens e presença de classes.

Copie a pasta **Data inteira** para o robô mantendo a estrutura relativa. Não é
necessário baixar novamente. Dados e saídas permanecem fora do Git.

## Executar posteriormente

Preencha `config/evaluation/models.local.json` a partir do exemplo existente.
Confirme a ordem das classes nos pesos: para um modelo com classe única de linhas,
use `class_map={"0":0}`. Para um modelo que separa círculo e linha, mapeie ambas
as classes para `line` com `class_map={"0":0,"1":0}`.

```bash
.venv/bin/python tools/evaluate.py run \
  --manifest Data/eval/naodevils-v36-manual-val/dataset.json \
  --models config/evaluation/models.local.json --labels masks \
  --output runs/naodevils-models-01 --warmup 5 --repeat 20
```

Para comparar realces, use esse mesmo manifesto nos comandos `prepare` e `run`
descritos em [avaliacao_offline.md](avaliacao_offline.md).

**Verificação dos pesos locais:** os metadados de
`Models/sim_data_seg_0126.onnx` declaram `{0: 'field_line'}` e entrada 320×320.
Esse modelo não distingue círculo de linha, o que é compatível com a classe única
do manifesto atual, desde que `field_line` inclua ambas as marcações. A taxonomia
dos engines TensorRT ainda precisa ser confirmada.

## Outras fontes

TORSO-21 e SoS possuem linhas, mas a pesquisa não confirmou a compatibilidade
de suas máscaras com a união de linhas e círculo central usada aqui. Confirme a
taxonomia e as regiões ignoradas antes de adicioná-los ao benchmark.
LARI permanece pendente de acesso e confirmação das categorias.
