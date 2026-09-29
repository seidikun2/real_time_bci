
## Dashboard v9 — estado atual no PCA

A dashboard do experimentador não desenha mais a trajetória histórica no PCA.
O mapa KDE da calibração permanece estático e apenas o ponto `rep1/rep2` atual é
atualizado. Isso remove a aparência de uma trilha que muda entre frames e reduz
o trabalho gráfico. A cada frame da GUI, se houver várias amostras LSL pendentes,
somente a mais recente é usada. Decoder e `GrazMI_Control` não foram alterados.

# BCI — protocolo em duas etapas

Execute sempre pela raiz:

```bash
python main.py
```

O protocolo atual usa três cues experimentais no PsychoPy/Unity:

- `LEFT_MI_STIM` — bola esquerda
- `RIGHT_MI_STIM` — bola direita
- `REST_STIM` — bola central, entre as pernas

`REST_STIM` é diferente de `REST`: `REST` continua sendo apenas a pausa entre trials.

## Etapa 1 — uma perna vs REST

No `config.yaml`:

```yaml
protocol:
  stage         : "single_target"
  start_phase   : "auto"
  online_target : "right"   # left | right
```

Com `start_phase: auto`, o `main.py` executa:

```text
EM_treino
→ IM_treino
→ online target com feedback PCA
→ online target sem feedback
```

O bloco de treino contém LEFT, RIGHT e REST, mas o classificador usa somente:

```text
RIGHT vs REST   # se online_target=right
```

ou

```text
LEFT vs REST    # se online_target=left
```

A outra perna permanece nos dados/cues, mas não entra no ajuste do modelo. No online ela funciona como confusor.

## Etapa 2 — duas pernas + REST

Depois de terminar a etapa 1, altere somente:

```yaml
protocol:
  stage: "two_legs"
```

e execute novamente:

```bash
python main.py
```

Com `start_phase: auto`, a segunda etapa repete exatamente o mesmo fluxo:

```text
EM_treino
→ IM_treino
→ online duas pernas com feedback PCA
→ online duas pernas sem feedback
```

O modelo passa a usar:

```text
LEFT vs RIGHT vs REST
```

Isto corresponde às duas pernas como classes motoras separadas, mantendo REST como estado explícito de não movimento. Não há movimento simultâneo BOTH nesta etapa.

Portanto, as duas etapas diferem somente no modelo:

```text
single_target : TARGET vs REST_STIM
two_legs      : LEFT vs RIGHT vs REST_STIM
```

A sequência experimental continua LEFT + RIGHT + REST_STIM nas duas etapas.

O `online_model_prefix: "latest"` e `model_session_types: ["IM_treino"]` fazem o online usar automaticamente o modelo da IM mais recente, portanto a segunda etapa não reutiliza por engano o modelo da primeira.

## Feedback contínuo e Unity

Os papéis dos streams ficam separados:

```text
Signal / BCI
= [rep1, rep2, p_left, p_rest, p_right, active_left, active_rest, active_right]

GrazMI_Control / BCIControl
= camada de decisão/histerese; left_leg/right_leg movem o avatar
```

`p_left/p_rest/p_right` vêm diretamente de `predict_proba`, antes do controlador. Em single-target, a classe que não participa do modelo fica com probabilidade 0 e `active_* = 0`; não há renormalização incluindo essa classe. `REST_STIM` nunca gera movimento bilateral.

A lógica visual do Unity deve mapear os cues assim:

```text
LEFT_MI_STIM  → bola para esquerda
RIGHT_MI_STIM → bola para direita
REST_STIM     → bola para o centro (trajetória antes usada por BOTH)
```

A alteração dessa trajetória é feita no projeto Unity, não neste pacote Python.

## PsychoPy

`experiment/stims_sequence.csv` contém apenas LEFT, RIGHT e REST_STIM. `BOTH_MI_STIM` continua no mapa de códigos apenas por compatibilidade, mas não participa do protocolo atual.

As imagens devem estar diretamente em `experiment/`:

```text
cross.png
rest.png
left_foot.png
right_foot.png
```

O `.psyexp` lê `stims_sequence.csv` e envia o marcador correspondente no início do cue.


## Dashboard / teste do contrato

Com `debug_plot.enabled: true`, `tools/plot_decoder_realtime.py` mostra simultaneamente rep1/rep2, P(LEFT/REST/RIGHT), classes ativas, estado do controlador e left_leg/right_leg. O dashboard apenas consome LSL; não participa da inferência.

Para validar o contrato sem coleta:

```bash
python tools/simulate_feedback_lsl.py
```

Para também publicar os vetores de exemplo em LSL:

```bash
python tools/simulate_feedback_lsl.py --lsl
```

## Dashboard PCA — mapa KDE estático (v5)

A dashboard em `tools/plot_decoder_realtime.py` agora recebe automaticamente o
`S#/online/pca_map.json` selecionado pelo `main.py`.

- os limites `rep1/rep2` vêm do JSON do modelo ativo;
- o `map_image` associado é desenhado uma única vez como background do PCA,
  preservando as curvas KDE e os rótulos gerados durante a calibração;
- se o PNG não estiver disponível, a dashboard reconstrói o background a partir
  dos polígonos de `density_regions` do próprio JSON;
- somente a trajetória/ponto online é atualizado no loop, portanto o mapa não
  acrescenta custo relevante à atualização em tempo real;
- os eixos foram deslocados para baixo e os textos de fase/classes/estado foram
  separados dos títulos para evitar sobreposição.


## Dashboard v6 — correção do event loop

A dashboard agora usa o timer nativo do canvas + `plt.show(block=True)` em vez de um loop manual com `flush_events()`. Isso evita erros do backend Tk como `RuntimeError: main thread is not in main loop` e `Tcl_AsyncDelete`, mantendo a dashboard apenas como consumidora dos streams LSL. O mapa PCA/KDE continua estático ao fundo.

## v7 — dashboard leve do experimentador

A dashboard Python foi simplificada para ser uma janela secundária ao Unity:

- mostra somente o **mapa PCA/KDE estático**, a posição/trilha curta de `rep1/rep2`,
  `p_left`, `p_rest`, `p_right`, classes ativas e a fase/cue do PsychoPy;
- **não consome nem plota `GrazMI_Control`**; esse stream continua sendo produzido
  normalmente e continua responsável pelo movimento do avatar no Unity;
- atualização visual padrão reduzida para **8 Hz**; isso não altera a taxa do decoder;
- trilha PCA padrão reduzida para **3 s / máximo de 96 pontos desenhados**;
- uso de **blitting** quando suportado pelo backend, mantendo o mapa KDE fora dos
  redraws normais e melhorando a responsividade ao mover a janela;
- janela compacta (`9.2 x 3.9`) para permanecer no canto durante o experimento.

Parâmetros em `config.yaml`:

```yaml
debug_plot:
  plot_hz          : 8.0
  trail_s          : 3.0
  max_trace_points : 96
  show_markers     : true
```

Se ainda quiser uma GUI mais leve, `plot_hz: 5.0` é suficiente para monitoramento
visual sem alterar em nada a inferência ou o controle enviado ao Unity.


## v8 — dashboard aparece imediatamente

A dashboard agora cria a janela **antes** de procurar `Signal / BCI`. A descoberta LSL é feita de forma não bloqueante dentro do timer da GUI. Assim, mesmo enquanto o decoder está inicializando, a janela aparece com `Signal: aguardando`. O blitting só é ativado depois do primeiro `draw_event` do backend, evitando chamadas de `draw()` antes do `Tk` estar no mainloop. Defaults da GUI: 5 Hz, trilha 2.5 s, 64 pontos. Isso não altera a taxa do decoder nem `GrazMI_Control`.
