# Datasets de visão para Booster T1 e RoboCup HSL

Pesquisa e inspeção em **2026-10-10**. Objetivo: bola distante, marcações do campo e implantação em tempo real no Booster T1 do RoboCIn, usando RealSense 1280 × 720.

## 1. Recomendação executiva

**Não foi identificado um dataset público, exclusivo da câmera do Booster T1, que cubra as nove classes de detecção e `CircleLine/Line` com qualidade operacional já comprovada.** Isso descreve o resultado desta pesquisa, não uma prova de inexistência.

A combinação mais promissora a validar é:

1. **HSLVision filtrado por origem T1 + dados próprios nativos:** preservar perspectiva, domínio e detalhe. Quatro grupos correspondentes ao T1 já estão na cópia local, totalizando 1.439 imagens 1280 × 720.
2. **B-Human BPRPerceptor 2026:** complemento para bola muito pequena, robôs e marcas de pênalti. O README declara imagens da HTWK com Booster K1/T1; a contagem por plataforma não está identificada nos JSONs inspecionados.
3. **TORSO-21 real:** fonte complementar prioritária para segmentação de linhas/campo, traves e interseções. Não é um dataset da câmera do T1 e exige adaptação de anotações.

**MergedDataset**, também do B-Human, é uma alternativa conveniente de detecção de três classes, mas já incorpora HSLVision e TORSO-21. Não somar seus tamanhos como dados novos nem usá-lo para treinar e depois testar nas fontes originais sem deduplicação e rastreamento de proveniência.

“Melhor” significa melhor recall/estabilidade na validação independente do nosso T1, não mais imagens ou mais downloads. Nenhum desses complementos foi treinado ou avaliado no robô nesta pesquisa.

## 2. O que foi verificado

Legenda:

- **Medido:** calculado a partir de anotações ou arquivos efetivamente inspecionados.
- **Publicado:** declarado pelos autores; não revalidado em toda a coleção.
- **Hipótese:** adequação esperada a testar no T1.
- **Pendente:** informação ausente ou ainda não auditada.

Foram consultados índices oficiais da RoboCup/SPL, páginas dos times, GitHub, Hugging Face e fontes acadêmicas. Para os candidatos principais:

- HSLVision: auditoria local dos Parquets e cabeçalhos das 6.068 imagens; ver [auditoria](auditoria_2026-10-10.md).
- BPR e MergedDataset: leitura remota dos diretórios centrais dos ZIPs e dos quatro JSONs COCO via HTTP Range. Não foi necessário baixar os ZIPs de 881 MiB e 6,3 GiB integralmente.
- TORSO-21: leitura completa dos YAMLs de anotações reais train/test, disponíveis separadamente. Imagens e máscaras completas não foram baixadas.
- Tese da whIRLwind e survey HTWK: texto extraído dos PDFs públicos; a tese foi acessada pelo espelho da Universidade de Amsterdam, pois o endereço da equipe retornou HTTP 429.

As contagens de tamanho de BPR/Merged/TORSO usam dimensões e caixas das anotações; seus cabeçalhos de imagens não foram verificados em massa. Não há avaliação de qualidade de modelos, distância física ou recall nesses números.

## 3. Comparação dos candidatos principais

Tamanho da bola = `min(largura, altura)` da caixa no espaço original declarado, antes de resize/clipping. Faixas `[0,8)`, `[8,16)`, `[16,32)`, `[32,+inf)`; não são as faixas de área COCO.

| Fonte inspecionada | Imagens anotadas | Caixas Ball | <8 px | 8–<16 px | 16–<32 px | >=32 px | Uso prioritário |
|---|---:|---:|---:|---:|---:|---:|---|
| HSLVision, todos os splits | 6.068 | 3.368 | 5 | 340 | 1.245 | 1.778 | Domínio HSL e marcações |
| Grupos HSLVision correspondentes ao T1 | 1.439 | 607 | 2 | 70 | 282 | 253 | Perspectiva T1 em 1280 × 720 |
| BPR 2026, `merged.json` | 4.873 | 2.168 | 193 | 672 | 656 | 647 | Complemento de bolas pequenas |
| MergedDataset, train+val | 16.610 | 8.020 | 195 | 842 | 1.836 | 5.147 | Pré-treino/ablação de detecção de três classes |
| TORSO-21 real, train+test | 10.464 | 6.081 | 5 | 180 | 1.172 | 4.724 | Linhas/campo, interseções e diversidade |

Essas linhas **não são conjuntos disjuntos**. BPR tem muito mais bolas sub-8 px anotadas que HSLVision, mas quase todas as imagens são de baixa resolução. Isso é evidência de suporte a uma faixa de tamanho, não de maior alcance em metros nem de superioridade de detalhe óptico. Não ampliar uma imagem 544 × 448 e chamar o resultado de dado nativo 1280 × 720.

## 4. HSLVision: manter a parcela T1

Fontes: [página oficial][hsl], [Hugging Face][hsl-hf], [tese Boosting Perception, tabela 4.1, página impressa 23 / PDF 26][thesis].

### Vínculo com o Booster T1

A página oficial declara plataformas Booster K1, T1 e KidSize. A tese anterior da whIRLwind identifica quatro grupos como **“whIRLwind T1 Data (Booster T1)”**. Cruzei esses grupos com os nomes e contagens dos arquivos locais:

| Grupo da tese | Prefixo local | Imagens medidas | Resolução local | Ball | <8 / 8–<16 / 16–<32 / >=32 px |
|---|---|---:|---|---:|---|
| ice-ribbon-1 | `images/ice ribbon 1_` | 264 | 1280 × 720 | 56 | 0 / 1 / 11 / 44 |
| ice-ribbon-2 | `images/ice ribbon 2_` | 341 | 1280 × 720 | 71 | 0 / 0 / 22 / 49 |
| ice-ribbon-3 | `images/ice ribbon 3_` | 497 | 1280 × 720 | 264 | 0 / 1 / 109 / 154 |
| panda-eye-1 | `images/panda eye 1_` | 337 | 1280 × 720 | 216 | 2 / 68 / 140 / 6 |
| Total | | 1.439 | | 607 | 2 / 70 / 282 / 253 |

**Conclusão:** já existe material fortemente relacionado ao T1 na cópia atual. É melhor selecionar e verificar sua proveniência do que descartar todo o HSLVision por inadequação do conjunto agregado.

**Limite:** a correspondência é por origem/nome/contagem e documentação acadêmica, não por um campo `robot_platform` autenticado em cada frame. Confirmar com os autores o modelo da câmera, intrínsecos, resolução nativa, robô que filmou e identidade das sessões. Não inferir câmera RealSense apenas por ser T1; não confundir robô que aparece na cena com robô que capturou a imagem.

A tese usa uma coleção anterior de 2.510 imagens e outra convenção de classe de robô; não transportar automaticamente suas decisões de rótulo para o HSLVision atual de 6.068 imagens.

### Pontos fortes e limitações

- Licença publicada: **CC BY 4.0**.
- Sete classes: Ball, Goalpost, K1/Robot, L/T/X intersections e Penalty Marks; sem máscaras de `CircleLine/Line`.
- Página declara filtragem de blur, oclusão e baixa qualidade: faltam condições operacionais difíceis.
- Grupos locais aparecem nos três splits; refazer isolamento temporal.
- Página declara 770 imagens de Abu Dhabi para holdout; arquivos não encontrados na cópia local. Solicitar pacote e protocolo.
- Depth é estimado por Depth Anything V3, não referência métrica independente.

## 5. B-Human BPRPerceptor 2026: melhor complemento encontrado para bola pequena

Fontes: [catálogo B-Human][bhuman], [diretório público][bpr], [README][bpr-readme]. O diretório lista os ZIPs com data de modificação de 2026-09-28.

### Acesso e relação com o T1

- [BPRDataset_B-Human.zip][bpr-zip]: 923.349.096 bytes, aproximadamente 881 MiB.
- O README declara imagens B-Human e um pequeno subconjunto HTWK de 2025 de seus Booster K1/T1.
- Não há campo de plataforma/câmera por imagem nos exemplos COCO inspecionados. Não é correto chamar as 4.873 imagens de “dataset exclusivo T1”.

### Conteúdo medido

O ZIP tem 4.873 imagens e dois JSONs que **não devem ser concatenados**:

| JSON | Imagens | Caixas | Ball | Robot | Penaltymark | Penaltymarkcross |
|---|---:|---:|---:|---:|---:|---:|
| `merged.json` | 4.873 | 15.236 | 2.168 | 11.864 | 675 | 529 |
| `result.json` | 4.362 | 13.608 | 2.108 | 10.356 | 667 | 477 |

Para a comparação foi usado **somente `merged.json`**, cuja contagem coincide com os arquivos de imagem. Confirmar com o mantenedor qual versão é canônica antes de incorporar.

Resoluções em `merged.json`: 4.722 imagens 544 × 448, 94 imagens 640 × 480 e 57 imagens 1280 × 800. Há 2.725 imagens sem anotação Ball e 335 sem objetos.

IDs de origem: `0 ball`, `1 robot`, `2 penaltymark`, `3 penaltymarkcross`. São caixas COCO; não foram identificados arquivos de máscaras nessa distribuição.

### Decisão

**Prioridade alta para um piloto controlado de bola pequena e negativos.** Ainda requer revisão visual de caixas de poucos pixels, política de `penaltymarkcross`, anotações incompletas de outras classes e agrupamento por gravação.

`penaltymarkcross` não é evidência de suporte a `XCross`, e `robot` não resolve a distinção `Person/Opponent`. Não cobre Goalpost, L/T/X e BRMarker como classes anotadas.

Licença: os JSONs têm `licenses: []` e o README consultado não especifica licença do conjunto. Registrar como **pendente**, mantendo atribuições das fontes; não atribuir automaticamente a licença do código B-Human aos dados.

## 6. B-Human MergedDataset: alternativa combinada, com sobreposição conhecida

Fonte: mesmo [README BPR][bpr-readme]. [Download][merged-zip]: 6.744.327.389 bytes, aproximadamente 6,3 GiB.

**Publicado:** mistura BPR, subconjuntos de TORSO-21 e HSLVision revisados manualmente, usada para treinar o detector no WHRG 2026. A revisão manual é declaração dos autores; não foi reavaliada integralmente aqui.

**Medido:**

| Split | Imagens | Caixas | Ball | Robot | Penaltymark | Sem anotação Ball | Sem objetos |
|---|---:|---:|---:|---:|---:|---:|---:|
| train2017 | 14.347 | 28.219 | 6.892 | 19.820 | 1.507 | 7.898 | 3.340 |
| val2017 | 2.263 | 4.026 | 1.128 | 2.735 | 163 | 1.236 | 597 |

O diretório `test2017` contém outras 172 imagens, mas **não foi encontrado JSON de teste no ZIP**. Não contar esse diretório como benchmark rotulado disponível. O sufixo `2017` é nome de layout COCO, não evidência da data de captura.

Há 12 resoluções, entre 320 × 240 e 1920 × 1080. Somente 550 das imagens anotadas são 1280 × 720; outras podem ter maior resolução, mas não correspondem automaticamente à câmera do T1.

IDs: `0 ball`, `1 robot`, `2 penaltymark`. As classes de linhas/interseções/traves das fontes originais não estão presentes como categorias nesta versão. Licença agregada não esclarecida nos JSONs/README inspecionados.

**Decisão:** candidato condicional para pré-treino/ablação de bola, robô e pênalti. Não é substituto direto do detector de nove classes. Auditar duplicatas e sessões antes de aceitar os splits. Não usar HSLVision/TORSO como teste independente após treinar nesse agregado.

## 7. TORSO-21: melhor complemento encontrado para marcações e máscaras

Fontes: [repositório][torso], [download][torso-data], [licença declarada para o dataset][torso-citation].

### Conteúdo e acesso

- Real: **10.464 imagens**, confirmado pelos YAMLs: 8.894 treino e 1.570 teste.
- Bolas: **6.081**, confirmado; somente cinco têm menor lado <8 px.
- Publicado: 10.464 segmentações de campo e 10.464 de linhas; cobertura de traves, robôs e interseções L/T/X.
- Publicado: coleção sintética adicional de 24.000 imagens; o ZIP sintético tem aproximadamente 235 GB decimais. Não priorizar seu download nesta fase.
- Real ZIP: 5.672.956.815 bytes, aproximadamente 5,28 GiB. Anotações YAML podem ser obtidas separadamente.
- `CITATION.cff` declara `type: dataset`, licença **MIT**; manter citação/proveniência.

### Limitações medidas e semânticas

- 46 prefixos de imageset inferidos no treino, 44 no teste; os 44 aparecem também no treino. Não tratar o split como holdout por sessão.
- O campo `metadata.imageset_id` ilustrado no README não está preenchido nos YAMLs inspecionados; o prefixo numérico do nome serve como pista, a validar.
- Metadados somados: 7.922 imagens com `Robot recorded=True`, 2.542 com `False`; 335 com perspectiva `Outside`.
- Câmeras documentadas: NAO Upper/Lower, Basler e C920, além de desconhecidas. Não foi identificado subconjunto Booster T1 nesses metadados.
- Resoluções variam de 320 × 240 a 1920 × 1080; apenas uma imagem é 1280 × 720.
- Bolas visíveis com flags: 485 `blurred`, 331 `concealed`; flags podem coexistir. A consistência visual precisa ser verificada.
- Máscara semântica de linha não distingue por si só círculo central e linhas retas. Não converter toda linha em `Line` e chamar isso de cobertura de `CircleLine`.
- Caixas de interseções e máscara de linhas são supervisões diferentes. Conversão para instâncias/polígonos YOLOv8-seg requer política explícita e validação de linhas finas.

**Decisão:** boa fonte de pré-treino e diversidade para marcações, com seleção por perspectiva/qualidade; não é a primeira escolha para suprir a falta de bolas sub-8 px no T1.

## 8. Outras opções pesquisadas

| Fonte | Evidência consultada | Decisão para este projeto |
|---|---|---|
| [BoosterRobotics/robocup_demo][booster-demo] | Código oficial K1/T1 com YOLOv8, engines e configurações; README não fornece dataset rotulado de treino | Solicitar ao fabricante os dados e pesos de origem; não confundir repositório de execução com dataset |
| [HTWK software survey 2026][htwk] | Declara dataset interno de classificador de bola ainda não publicado na data do survey | Contato prioritário para originais K1/T1; BPR posterior já publica um subconjunto |
| [Unitree RoboCupFootball_Dataset][unitree] | README declara TORSO-21 + novas cenas em formato YOLO; card Apache-2.0 | Secundário: não há evidência no card de captura T1; verificar classes/proveniência e sobreposição antes de incorporar |
| [B-Human BallAndPenaltyMark 2024][bpm] | 182.229 exemplos, pipeline de patches 32 × 32 grayscale | Não substituir frames RGB de detector; útil apenas para uma futura etapa ROI/classificador |
| [rUNSWift Field Segmentation][runswift] | 20 sequências; imagem superior 1280 × 960, máscara 256 × 192; campo versus fundo, linhas incluídas no campo | Não serve diretamente para segmentar linhas finas ou CircleLine; apoio a campo/iluminação |
| [B-Human sim-to-real segmentation][bh-seg] | 11.821 imagens sintéticas base, variantes traduzidas e fundos COCO, ZIP ~27 GiB | Secundário, domínio NAO/sintético; não resolve proveniência T1 |
| [ROBO][robo] / [RoboCupVision][robo-seg] | Detecção YUV 512 × 384; segmentação sintética/real NAO | Secundário para técnicas/pretraining; prioridade inferior a T1/HSL |
| [SaiResearch/booster_dataset][sai] | Dados `.npz` de movimento/controle, qpos/qvel | Não é dataset de visão com caixas/máscaras |
| [Booster T1 Task Recovery][recovery] | Vídeos T1 de transferência de pratos, LeRobot | Plataforma coincide, tarefa e supervisão não; sem prioridade para futebol |

Datasets de transmissão de futebol, câmera aérea/SSL e conjuntos genéricos Roboflow não foram priorizados: perspectiva, bola, câmera ou proveniência divergem. O nome “Booster” ou “soccer” sozinho não garante utilidade para detecção egocêntrica.

## 9. Experimento recomendado e critérios de aceitação

Ver o [plano experimental](../plano_de_testes.md) para a matriz completa. Antes da busca de arquitetura:

1. Congelar uma validação própria T1 por sessão, com bola pequena e vídeos sem bola. Reservar teste independente.
2. Corrigir o HSLVision, identificar os quatro grupos T1 e revisar amostras visualmente.
3. Comparar D0 (HSL corrigido + treino próprio) e D1 (D0 + BPR deduplicado) com o mesmo modelo, resolução e orçamento de passos.
4. Usar tarefa offline Ball como diagnóstico da contribuição dos dados, sem mudar os IDs de produção; resolver cobertura das nove classes antes do fine-tuning final.
5. Medir recall por tamanho, FP/min e perdas consecutivas no mesmo conjunto do T1. Relatar suporte e incerteza por sessão.
6. Se BPR ajudar, equilibrar a amostragem por domínio para que 544 × 448 não domine o fine-tune final em dados nativos.
7. Avaliar TORSO separadamente para segmentação, comparando com baseline e anotação própria `CircleLine/Line`.

Para aprovar uma fonte, exigir ganho na validação do T1 ou cobertura demonstrável de falhas relevantes, proveniência reconstruível, mapeamento correto e ausência de contaminação do teste. Mais caixas pequenas não é critério suficiente.

### Solicitações objetivas aos mantenedores

- **whIRLwind:** originais dos quatro grupos T1, câmera/intrínsecos, sessões, frames descartados por blur/oclusão, protocolo e acesso ao holdout Abu Dhabi.
- **HTWK:** gravações T1/RealSense sem redução, dados de bola distante e identificação dos frames cedidos a HSLVision/BPR; licença e labels completas.
- **B-Human:** arquivo canônico entre `merged.json`/`result.json`, licença, proveniência por imagem, política de `penaltymarkcross`, split por gravação e anotações do `test2017`.
- **Booster Robotics:** pesos treináveis, dataset e convenções de classes/segmentação associados aos engines usados no T1.

## 10. Rastreabilidade das medições

BPR/Merged: listar diretório central ZIP, ler JSONs por Range, contar `images`, `annotations` e `categories`; calcular `min(bbox[2],bbox[3])` para Ball. TORSO: `yaml.CSafeLoader`, apenas anotações presentes (`in_image=True`), caixas `vector=[[x1,y1],[x2,y2]]`; calcular `min(x2-x1,y2-y1)`. Não contar marcadores `in_image=False` como objetos.

Hashes SHA-256 dos arquivos de anotação lidos, para detectar mudanças futuras:

| Fonte / arquivo | SHA-256 |
|---|---|
| BPR `BHumanBallPenaltyRobotWithCrossesUnified/merged.json` | `f4a2e7f9c23bbf0d445937864c5ec79c7e2d418e1528e55221453008d00fdb60` |
| BPR `BHumanBallPenaltyRobotWithCrossesUnified/result.json` | `b0fa17c1ea2973948ddc08b0a2b22d60549dad681878ed8dff8eca36224c9311` |
| Merged `annotations/instances_train2017.json` | `7e5e8aa48ff7813323c67bef44f764d52c4ff6a79d45c6aa149df8e4dbba08d9` |
| Merged `annotations/instances_val2017.json` | `b88b85e2688ff85c65a4c4729e45eda67f644c91e1be5297dcf24c49c53dccaa` |
| TORSO `reality/train/annotations.yaml` | `d0563655856cab02b65a2d73af33915f74ab709fdef59398ff27cfd08cb333bb` |
| TORSO `reality/test/annotations.yaml` | `1c598f4157aa4b813b6397fdf7af3576fcd84acf97a4faf728ee8a4f78e5dac0` |

As imagens e máscaras dos datasets externos ainda precisam de auditoria visual e hashes após aquisição. Não foram medidos recall, mAP ou latência de modelos treinados com essas fontes.

## 11. Fontes primárias e acesso

Todos os links abaixo foram consultados em 2026-10-10; os arquivos completos de imagem não foram todos baixados. Os índices oficiais ajudaram na descoberta, mas números específicos acima vêm dos artefatos ou páginas dos autores.

- [Índice de datasets RoboCup](https://www.robocup.org/robocup_datasets).
- [Índice de datasets SPL](https://spl.robocup.org/datasets/).
- [HSLVision — página e descrição das plataformas][hsl].
- [HSLVision — dataset e licença][hsl-hf].
- [Boosting Perception — tese, tabela 4.1][thesis].
- [B-Human — catálogo de datasets][bhuman].
- [BPRPerceptor — arquivos][bpr] e [README de proveniência][bpr-readme].
- [TORSO-21 — descrição/schema][torso], [arquivos][torso-data] e [CITATION/licença][torso-citation].
- [HTWK — survey HSL 2026][htwk].
- [Booster Robotics — execução oficial K1/T1][booster-demo].

[hsl]: https://intelligentroboticslab.github.io/hslvision/
[hsl-hf]: https://huggingface.co/datasets/whirlwind-ams/hslvision
[thesis]: https://staff.fnwi.uva.nl/a.visser/education/bachelorAI/BSc_Thesis_Da%CC%81rio_Xavier_Catarrinho-2.pdf
[bhuman]: https://b-human.de/datasets-en.html
[bpr]: https://b-human.informatik.uni-bremen.de/public/datasets/BPRPerceptor/
[bpr-readme]: https://b-human.informatik.uni-bremen.de/public/datasets/BPRPerceptor/readme.md
[bpr-zip]: https://b-human.informatik.uni-bremen.de/public/datasets/BPRPerceptor/BPRDataset_B-Human.zip
[merged-zip]: https://b-human.informatik.uni-bremen.de/public/datasets/BPRPerceptor/MergedDataset.zip
[torso]: https://github.com/bit-bots/TORSO_21_dataset
[torso-data]: https://data.bit-bots.de/TORSO-21/
[torso-citation]: https://raw.githubusercontent.com/bit-bots/TORSO_21_dataset/master/CITATION.cff
[htwk]: https://hsl.robocup.org/wp-content/uploads/2026/03/HTWK-Robots_36.pdf
[booster-demo]: https://github.com/BoosterRobotics/robocup_demo
[unitree]: https://huggingface.co/datasets/unitreerobotics/RoboCupFootball_Dataset
[bpm]: https://docs.b-human.de/coderelease2024/perception/ball-and-penalty-mark-detection/
[runswift]: https://arxiv.org/html/2108.12809v1
[bh-seg]: https://b-human.informatik.uni-bremen.de/public/datasets/semantic_segmentation/readme.txt
[robo]: https://github.com/szemenyeim/ROBO
[robo-seg]: https://github.com/szemenyeim/RoboCupVision
[sai]: https://huggingface.co/datasets/SaiResearch/booster_dataset
[recovery]: https://huggingface.co/datasets/KillzoneCCO/Booster
