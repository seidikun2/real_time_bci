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
