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

## Unity

A interface do decoder permanece compatível:

```text
Signal / BCI = [rep1, rep2, left, both, right]
```

`REST_STIM` não cria um canal motor extra. Quando REST é dominante, nenhuma perna deve ser ativada.

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
