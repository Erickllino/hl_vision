# hl_vision — replay de rosbag no hsl-player

Esta branch publica os tópicos originais da câmera, que o hsl-player consome como
se fossem da câmera física. A interface X11 mostra RAW com máscaras/caixas/segmentos e o
frame pré-processado. O detector usa **K1** por padrão neste replay, sem alterar
o YAML de produção; a segmentação usa o modelo configurado no `vision.yaml`.
A branch `fix/vision-preprocess` do hsl-player recebeu suporte pequeno a ajuste
dinâmico do filtro de segmentação.

## No robô

Pré-requisitos: ROS 2 e workspace do `hsl-player` compilado/sourceado (pacotes
`vision`, `vision_interface`, `image_transport`); `python3` do ROS com
`rclpy`, `PyYAML`, `numpy`, `opencv-python`/`python3-opencv`, `Pillow` e Tkinter.
Para MCAP sem o plugin ROS `rosbag2_storage_mcap`, instale `python3 -m pip
install --user mcap` (fallback Python; aceita um único arquivo MCAP na bag).
Instale esses módulos no **mesmo Python usado pelo ROS**; não use o `uv.lock`
legado de treinamento para executar o replay. Precisa de `ros2 bag play` e
acesso aos engines/ONNX compatíveis com a GPU do robô. Copie a bag inteira
para `data/minha_bag/` (diretório com `metadata.yaml` e arquivos de armazenamento).

Antes, confira os tópicos e tipos reais da bag:

```bash
ros2 bag info data/minha_bag
```

Na estação de trabalho, abra `ssh -X booster@192.168.0.101` (se necessário,
`ssh -Y ...` para X11 confiável). Na sessão remota:

```bash
cd ~/Workspace/hl_vision
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/Workspace/hsl-player/install/setup.bash
export ROS_DOMAIN_ID=87   # domínio LIVRE, diferente do robô em produção
python3 src/replay.py data/minha_bag \
  --config ~/Workspace/hsl-player/src/vision/config/vision.yaml \
  --vision-bin ~/Workspace/hsl-player/install/vision/lib/vision/vision_node \
  --color-topic /boostercamera/head/rgb \
  --intrin-topic /boostercamera/head/rgb/camera_info \
  --pose-topic /head_pose \
  --rate 0.5
```

O tópico `--color-topic` (Image ou CompressedImage; este último deve conter
`compressed` no nome) deve existir na bag. Quando informados, `--intrin-topic`
e `--pose-topic` também devem existir. Para outros
modelos diferentes do K1, use caminhos **absolutos** em `--det-model` e `--seg-model`. Se o
`vision_node` estiver em outro local, localize-o em `install/vision/lib/vision/`.
Use a configuração do mesmo checkout/build do hsl-player. O processo da visão
executa com `cwd` na raiz desse checkout para resolver paths relativos do YAML.

**Importante:** a bag também precisa conter os insumos de pose do nó (por
exemplo `/head_pose` ou `/booster_soccer/t_head2base`, conforme a configuração)
para que as projeções 3D/linhas sejam confiáveis. Sem `CameraInfo`, o nó usa
intrínsecos estáticos do YAML, que podem não corresponder à gravação. O programa
não sintetiza esses dados nem altera os timestamps da bag. Use um domínio ROS
isolado: definir `ROS_DOMAIN_ID` não impede colisões se o mesmo ID estiver em
uso por outros processos. Isso evita injetar imagens da bag no robô em operação.

## Interface e limites

- RAW: caixas de detecção em amarelo; quando o modelo segmenta, máscaras e
  caixas de segmentação desenhadas pelo hsl-player são sobrepostas ao mesmo
  frame. **Segmentos de linha 2D** ajustados geometricamente aparecem em magenta.
  A máscara e os segmentos são saídas diferentes: um contorno pode ser previsto
  sem gerar segmento. Se o modelo retornar zero objetos para um frame, nada será
  sobreposto; na bag maior, a maioria dos frames testados teve zero segmentos.
  O contador «Máscaras previstas» ajuda a distinguir ausência de predições de
  erro de transporte. Uma máscara rara permanece visível por até 1 s de tempo
  da bag para não ser ocultada pelo detector mais rápido.
  O tópico extra `/booster_camera/segmentation_overlay/compressed` só transporta
  JPEGs quando a interface assina e há predições; não altera a inferência normal.
- Segunda imagem: `/booster_camera/preprocess_video`, entrada geométrica e de
  cor do modelo de **segmentação** (não a entrada do detector). A publicação é
  ligada numa cópia temporária do YAML, sem alterar os arquivos do hsl-player.
- Os valores em tela podem ser editados e confirmados com **Aplicar filtros**.
  Na branch `fix/vision-preprocess` atualizada do hsl-player, a interface envia
  um parâmetro ROS: o filtro muda entre frames **sem recompilar nem recarregar
  os modelos**. Com um binário antigo, faz um reinício de compatibilidade. O botão
  fica desabilitado até um frame pré-processado confirmar a mudança.
- A bag roda em loop a meia velocidade por padrão; `--rate` altera a velocidade.
  Em MCAP com fallback Python, a barra mostra posição/duração e número da volta
  (indicador de reprodução, ainda sem seek). Aceita tanto diretório de bag com
  um MCAP quanto caminho direto para um arquivo `.mcap`.
  Se o modelo não acompanhar, mensagens podem ser descartadas (QoS depth 1).
  Somente câmera, CameraInfo e pose são reproduzidos (nunca comandos do robô ou
  detecções já gravadas). Nesta versão do hsl-player `save_data=false` causa
  falha na inicialização; o replay grava aproximadamente 1 frame/s em
  `~/Workspace/vision_log/`. `runs/replay/vision.log`, `bag.log` e
  `vision_replay.yaml` ajudam no diagnóstico.
  Fechar a janela encerra os dois processos iniciados pelo aplicativo.

### Bag maior disponível no robô de testes

`data/alan_teste3_visao.mcap` é um link para a bag já armazenada no robô
(89 s, aproximadamente 1.079 frames RGB). Ela contém `/booster_soccer/t_head2base`
mas não `/head_pose`. Use o comando acima substituindo a bag por esse caminho
e `--pose-topic /booster_soccer/t_head2base`. O replay publica apenas os três
tópicos escolhidos; os demais dados gravados não são enviados para o robô.
Feche a interface anterior e abra outra para carregar o código/binário novo.

```bash
cd ~/Workspace/hl_vision
source /opt/ros/humble/setup.bash
source ~/Workspace/hsl-player/install/setup.bash
export ROS_DOMAIN_ID=87
python3 src/replay.py data/alan_teste3_visao.mcap \
  --config ~/Workspace/hsl-player/src/vision/config/vision.yaml \
  --vision-bin ~/Workspace/hsl-player/install/vision/lib/vision/vision_node \
  --color-topic /boostercamera/head/rgb \
  --intrin-topic /boostercamera/head/rgb/camera_info \
  --pose-topic /booster_soccer/t_head2base --rate 0.5
```
`--intrin-topic` e `--pose-topic` são opcionais para inspecionar imagem e inferência
2D. `CameraInfo` atualiza os intrínsecos; sem ele o hsl-player utiliza os valores
estáticos do YAML. A pose é usada para projeção 3D e geometria dos segmentos:
sem pose correspondente, as linhas/projeções podem ficar incorretas mesmo que
a imagem e a inferência 2D continuem aparecendo.

Estrutura: `src/` implementação, `docs/` relatórios históricos, `data/` bags
(ignoradas pelo Git). Teste sem ROS: `python3 -m unittest discover -s tests -v`.
