# Plano de Testes — Escolha do Modelo de Visão (RoboCIn Humanoid / Booster T1)

**Objetivo:** substituir o YOLOv8 por um detector de menor latência e maior precisão, com decisão baseada em medições no hardware real, não em benchmarks de terceiros.

**Hardware-alvo:** Jetson AGX Orin 32GB (perception board do Booster T1, ~200 TOPS INT8) + câmera Intel RealSense (modelo a confirmar).

**Candidatos:** YOLOv8 (baseline atual), YOLO26 (n/s/m, e variante P2), RF-DETR (candidato de teto de precisão).

**Princípio de decisão:** maximizar precisão nas classes críticas *sujeito a* latência end-to-end ≤ orçamento definido na Fase 0. Nunca o contrário.

---

## Fase 0 — Levantamento de requisitos (no robô, ~1h)

Estas medições definem o "orçamento" que as fases seguintes usam como critério de aprovação. Sem elas, qualquer comparação de modelos é chute.

### T0.1 — Identificar a câmera e seus perfis

Executar na perception board com a RealSense conectada:

```bash
rs-enumerate-devices -s        # modelo, serial, firmware
rs-enumerate-devices           # todos os perfis (resolução × formato × fps)
```

**Registrar:** modelo (D435/D435i/D455/outro), resolução e FPS máximos do stream de **cor**, e se o stream de cor a 30fps está disponível na resolução usada hoje.

**Por que importa:** o FPS de cor da câmera é o teto físico do pipeline. Se a câmera entrega 30fps, um detector a 90fps não traz benefício — a folga deve ser convertida em precisão (modelo maior ou resolução maior).

### T0.2 — Medir o pipeline atual (baseline YOLOv8)

Com o stack hsl-player rodando em condição normal de jogo:

```bash
ros2 topic hz /camera/camera/color/image_raw     # taxa da câmera
ros2 topic hz <topico_de_deteccoes>              # taxa de saída do detector
ros2 topic delay <topico_de_deteccoes>           # latência frame->detecção
```

**Registrar:** as três taxas + uso de GPU/CPU (`tegrastats`) durante a medição.

**Critérios derivados:** se saída < entrada, o detector atual dropa frames (gargalo confirmado); a latência medida do v8 vira o número a bater.

### T0.3 — Definir o requisito de FPS do loop

Duas fontes, registrar ambas:

1. **Consumo downstream:** `ros2 topic hz` na entrada do node de behavior/localização, ou loop rate no código. Visão mais rápida que a decisão é desperdício.
2. **Dinâmica da bola:** `fps ≥ v_max_bola / deslocamento_tolerado`. Ex.: 4 m/s ÷ 0,15 m ≈ 27 → alvo 30fps.

**Saída da Fase 0:** documento de 1 parágrafo com: modelo da câmera, FPS-alvo, orçamento de latência por frame (ex.: 33ms end-to-end para 30fps) e baseline do v8.

---

## Fase 1 — Precisão (no cluster, sem robô)

Treinar todos os candidatos no HSLVision sob protocolo idêntico (mesmos splits, mesma resolução por comparação, mesmas seeds). Usar `train_yolo26.py` deste pacote.

### T1.1 — Matriz de treino

| Exp | Modelo | imgsz | Propósito |
|---|---|---|---|
| A | YOLOv8s (atual) | a de hoje | baseline honesto re-treinado no mesmo dataset |
| B | YOLO26n | 640 | piso de latência |
| C | YOLO26s | 960 | candidato principal |
| D | YOLO26m | 960 | teto YOLO se a latência permitir |
| E | YOLO26s-p2 | 960 | head P2 p/ objetos pequenos (bola distante, pênalti) |
| F | RF-DETR (S ou base) | ~960 | teto de precisão / oclusão |
| G | YOLO26s | 1280 | testa se resolução > variante p/ objetos pequenos |

### T1.2 — Métricas a registrar por experimento

Além do mAP50-95 global, obrigatoriamente o **mAP por classe** — a decisão de jogo depende de bola e trave, não da média. Registrar: mAP50 e mAP50-95 (global e por classe), recall da classe *ball* estratificado por tamanho de caixa (small/medium/large do COCO eval — bola distante é o caso crítico), e curvas P-R das 7 classes.

**Critério:** um candidato só avança à Fase 3 se superar o baseline A em mAP50-95 global **e** não regredir no recall de *ball small*.

### T1.3 — Robustez cross-venue

O HSLVision reserva ~770 imagens de RCAP Abu Dhabi 2025 como venue não visto. Avaliar os 2–3 melhores da T1.1 nesse holdout (se publicado; senão, holdout manual por venue: treinar sem Salvador, testar em Salvador).

**Registrar:** Δ(mAP treino-venues → venue não visto). **Por que importa:** competição = venue novo, iluminação nova. Um modelo 1pt melhor em val mas 5pts pior cross-venue é a escolha errada. É aqui que o RF-DETR pode justificar seu custo.

### T1.4 — Erros qualitativos

Para os finalistas, inspecionar 50 falsos negativos/positivos das classes *ball* e *robot* (confusões típicas: bola vs. marca de pênalti ao longe, robô parcialmente ocluso). Registrar padrões — isso orienta augmentation e coleta futura de dados próprios.

---

## Fase 2 — Latência preliminar (no cluster / máquina com GPU)

Só para eliminar candidatos obviamente inviáveis antes de ocupar o robô. **Números daqui NÃO decidem nada** — GPU de cluster ≠ Orin.

Executar `yolo benchmark` / `trtexec` em FP16 e INT8 para cada finalista, na resolução do experimento. Registrar ms/frame relativo (candidato ÷ baseline v8 na mesma máquina). Eliminar quem for >3× o orçamento relativo.

---

## Fase 3 — Latência real (na Jetson AGX Orin, com robô)

A fase decisiva. Regras: engine TensorRT **gerado na própria Orin** (engines não são portáveis entre GPUs), power mode fixado e anotado (`nvpmodel -q`, usar MAXN ou o modo de jogo real), clocks travados (`jetson_clocks`) e 2 min de warm-up antes de medir.

### T3.1 — Latência isolada do modelo

Para cada finalista × precisão (FP16, INT8):

```bash
trtexec --loadEngine=<engine> --iterations=500 --avgRuns=100
```

**Registrar:** média, p95 e p99 (o p99 importa mais que a média num loop de controle), GPU util e potência (`tegrastats`).

### T3.2 — Verificar precisão pós-quantização

INT8 pode degradar. Rodar `model.val()` com o engine INT8 no split de teste e comparar com o .pt. **Critério:** queda de mAP50-95 ≤ 1pt com INT8 (com calibração adequada, ~500 imagens do próprio HSLVision). Se cair mais, usar FP16 e reavaliar latência.

### T3.3 — Latência end-to-end no pipeline ROS2

Integrar o finalista no node de visão do hsl-player e medir o que a Fase 0 mediu para o v8: `ros2 topic hz` e `ros2 topic delay` na saída de detecções, com o stack completo rodando (locomoção ativa — a motion board é separada, mas medir com o sistema inteiro ligado captura contenção de I/O/USB da RealSense).

**Critério de aprovação:** taxa de saída = taxa da câmera (zero drops) e latência p99 ≤ orçamento da Fase 0.

### T3.4 — Estresse térmico

30 min contínuos de inferência em MAXN, `tegrastats` logando. **Registrar:** throttling e degradação de FPS ao longo do tempo. Um modelo que só cabe no orçamento nos primeiros 5 minutos não cabe num tempo de jogo.

---

## Fase 4 — Validação em campo

Com o vencedor provisório: teste funcional em treino real (perseguir bola, alinhar chute, localizar por interseções de linha). Registrar falhas de percepção observadas (bola perdida em oclusão, trave confundida com fundo) e, idealmente, gravar rosbags para reprodução offline. Comparar taxa de perda de tracking da bola vs. o sistema v8 no mesmo cenário.

---

## Tabela de decisão final

Preencher e decidir em reunião de equipe:

| Critério | Peso | v8 (base) | YOLO26-? | RF-DETR |
|---|---|---|---|---|
| mAP50-95 test (global) | alto | | | |
| Recall ball-small | alto | | | |
| Δ cross-venue | médio | | | |
| Latência p99 INT8 na Orin (ms) | eliminatório | | | |
| Drops end-to-end (T3.3) | eliminatório | | | |
| Estabilidade térmica 30min | eliminatório | | | |
| Facilidade de manutenção/export | baixo | | | |

**Regra:** critérios eliminatórios primeiro; entre os aprovados, vence o de melhor precisão nas classes críticas.

---

## Riscos e observações

O engine TensorRT deve sempre ser gerado na Orin com a mesma versão de JetPack usada em jogo — atualizar JetPack invalida engines. A calibração INT8 deve usar imagens do domínio (HSLVision + frames próprios), nunca COCO. Se a RealSense for D435 (baseline curta), a profundidade a >6m é pouco confiável e a estimativa de distância da bola deve vir da geometria (altura da câmera + pose) mais do que do depth — isso não muda a escolha do detector, mas muda o pós-processamento. Por fim, o HSLVision cobre venues variados mas não o *seu* campo: coletar e anotar ~500 imagens do campo do RoboCIn e adicioná-las ao fine-tune final tende a valer mais que qualquer troca de arquitetura.
