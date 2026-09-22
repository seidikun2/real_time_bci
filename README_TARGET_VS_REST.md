# Modo perna-alvo vs REST_STIM

Configuração principal:

```yaml
protocol:
  training_mode : "target_vs_rest"
  online_target : "left"   # left | right | both
```

- `REST` (6): pausa entre trials.
- `REST_STIM` (8): condição experimental usada no modelo.
- `stims_sequence.csv` pode conter LEFT, RIGHT, BOTH e REST_STIM.
- Em `target_vs_rest`, somente a perna escolhida e REST_STIM entram no treino.
- Os demais cues continuam gravados e podem funcionar como confusores no online.
- O decoder continua publicando `[rep1, rep2, left, both, right]`.
- A probabilidade TARGET é colocada apenas no canal da perna escolhida.
- Para voltar ao comportamento anterior, use `training_mode: "motor_multiclass"`.

As imagens do PsychoPy devem ficar diretamente em `experiment/`.
