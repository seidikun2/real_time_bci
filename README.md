# BCI Graz MI — projeto reorganizado

## Como usar

O ponto de entrada continua sendo somente:

```bash
python main.py
```

Antes de rodar, ajuste `config.yaml` (sujeito, sessão, pasta de dados, modo simulado/real e, se necessário, caminho do Python do PsychoPy).

O `main.py` inicia aquisição, logger, PsychoPy e decoder conforme a fase. O PsychoPy é compilado automaticamente a partir de `experiment/Graz_UpperLimb.psyexp`; não é necessário abrir o Builder para cada bloco.

## Estrutura

```text
bci_project/
├── main.py                    # único ponto de entrada
├── config.yaml                # configuração principal
├── bci_core/                  # módulos internos do pipeline
│   ├── class_schema.py        # definição dinâmica LEFT/BOTH/RIGHT
│   ├── config_models.py       # leitura/validação do config
│   ├── realtime_signal_transmit.py  # EEG simulado -> LSL
│   ├── input_hiamp.py         # g.HIamp -> LSL
│   ├── receive_data_log.py    # gravação oficial sinal + marcadores
│   ├── decoder_calibration.py # treino Riemann + PCA + SVM
│   ├── representation_feedback.py # mapa/JSON da representação
│   ├── online_inference.py    # PCA + probabilidades -> LSL
│   ├── intention_control.py   # densidade/probabilidade -> movimento Unity
│   ├── check_data.py          # QC
│   └── psychopy_process.py    # compila/inicia/encerra PsychoPy
├── experiment/
│   ├── Graz_UpperLimb.psyexp  # fonte do experimento PsychoPy
│   └── stims_sequence.csv     # controla quais classes aparecem
├── stims/                     # imagens do protocolo
└── tools/
    └── plot_decoder_realtime.py # janela diagnóstica opcional
```

`Graz_UpperLimb_autorun.py` e `_lastrun.py` deixam de ser arquivos de trabalho do projeto: o autorun é gerado automaticamente quando necessário.

## Duas ou três classes: altere somente `stims_sequence.csv`

O pipeline deriva as classes motoras dos marcadores efetivamente presentes no bloco. A ordem canônica é:

- `LEFT_MI_STIM` — perna esquerda
- `BOTH_MI_STIM` — ambas as pernas
- `RIGHT_MI_STIM` — perna direita

A versão incluída contém as três classes. Para executar um protocolo binário, remova as linhas `BOTH_MI_STIM` de `experiment/stims_sequence.csv`. Não é necessário alterar `config.yaml`, treino, QC ou decoder.

O marcador de BOTH é `7`. O stream online mantém cinco canais na ordem:

```text
rep1, rep2, left, both, right
```

Em um modelo de duas classes, `both` permanece no stream com valor `0`, preservando a interface com o Unity.

## Simulação

O sinal simulado tem três perfis configuráveis:

```yaml
sim_profile_left_mi:  [1.0, 0.3]
sim_profile_both_mi:  [1.0, 1.0]
sim_profile_right_mi: [0.3, 1.0]
```

O perfil BOTH é bilateral. Ele só é acionado quando o PsychoPy emite `BOTH_MI_STIM` seguido de `ATTEMPT`.

A simulação apenas publica EEG em LSL. A gravação oficial é feita exclusivamente por `receive_data_log.py`, assim como no g.HIamp.

## Online: dois modos separados

O `config.yaml` define dois modos online:

1. `IM_online_PCA` — **primeiro**, com feedback da representação PCA no Unity;
2. `IM_online_sem_feedback` — **depois**, sem apresentação do mapa PCA.

Cada modo possui seu próprio contador e controle de repetição. Os dois são gravados na mesma pasta `S<session>/online/`; a condição é identificada no nome do arquivo por `IM_online_PCA` ou `IM_online_sem_feedback`. Ao terminar um bloco é possível repetir aquele mesmo modo, seguir para o próximo ou finalizar a sessão.

O decoder continua publicando `rep1/rep2` nos dois modos para que aquisição e modelo sejam idênticos. O stream LSL recebe metadata `feedback_mode` (`none` ou `pca`) e `feedback_enabled`; a decisão de mostrar ou não a representação fica no Unity.

## Controle motor por densidade + probabilidade

O online agora separa **decodificação** de **controle**. Isso evita usar a
probabilidade do SVM diretamente como altura da perna.

### Stream do decoder

O `online_inference.py` continua publicando:

```text
Signal / BCI
rep1, rep2, left, both, right
```

`left/both/right` são probabilidades das classes quando o modelo possui
`predict_proba`. `rep1/rep2` são as coordenadas no mesmo espaço do mapa PCA.

### Regiões numéricas no `pca_map.json`

Além do PNG visual, o JSON agora contém `density_regions`. Para cada classe,
são guardadas as regiões HDR de 50%, 80% e 95% como polígonos nas coordenadas
`rep1/rep2`. Portanto o controlador não precisa inferir regiões pelos pixels do PNG.

O padrão é:

- **entrar** na classe ao atingir a HDR 50%;
- **manter** a classe enquanto o ponto permanecer na HDR 80%;
- usar `P(left/both/right)` para confirmar a classe e resolver sobreposições;
- exigir persistência temporal antes de ligar/desligar.

Tudo é ajustável em `config.yaml`:

```yaml
control:
  entry_density_mass: 0.50
  hold_density_mass: 0.80
  enter_persist_s: 0.25
  exit_persist_s: 0.20
  min_probability: 0.40
  min_probability_margin: 0.05
```

### Stream de controle para o Unity

O novo `bci_core/intention_control.py` publica:

```text
GrazMI_Control / BCIControl
left_leg, right_leg, rest, left, both, right, confidence, density_gate, state_id
```

`state_id` usa:

```text
0 = REST
1 = LEFT
2 = BOTH
3 = RIGHT
```

`left_leg` e `right_leg` são posições contínuas entre 0 e 1. Quando LEFT é
ativado, por exemplo, `left_leg` continua subindo a partir do valor atual. Ao
sair da região de manutenção, ela passa a cair a partir do ponto em que estava.
BOTH faz as duas pernas subirem.

A curva temporal também é configurável:

```yaml
control:
  movement_rise_s: 1.20
  movement_fall_s: 0.80
  output_rate_hz: 60.0
```

O Unity pode continuar usando `Signal/BCI` para desenhar o ponto sobre o mapa,
e usar `GrazMI_Control/BCIControl` exclusivamente para animar as pernas. Nos
dois modos online o controlador funciona da mesma forma; a condição
`sem feedback` apenas deixa de mostrar o mapa ao participante.

Cada bloco online também grava um arquivo `*_control_<run_id>.csv` com estado,
posição das pernas, probabilidades, `rep1/rep2` e indicação de entrada/hold em
cada região.

## Janela PCA em Python

A antiga janela `plot_decoder_realtime.py` agora é apenas uma ferramenta diagnóstica e vem desligada:

```yaml
debug_plot:
  enabled: false
```

Ative somente quando quiser depurar o decoder. Ela não é necessária para o protocolo com Unity.

## Organização dos dados, modelo e `pca_map.json`

A organização agora é orientada aos arquivos que você realmente consulta. Não existe mais uma pasta global `S#/models/` para novos treinos.

```text
<log_root>/<subject>/S<session>/
├── IM_treino/
│   └── train/
│       ├── ..._markers_<run_id>.csv
│       ├── ..._signal_<run_id>.csv
│       ├── pca_map_<run_id>.json       # uso direto / Unity
│       ├── pca_map_<run_id>.png        # imagem projetada no Unity
│       ├── _model/
│       │   └── <run_id>/
│       │       ├── classifier.pkl
│       │       ├── pca.pkl
│       │       ├── riemann_mean.pkl
│       │       ├── model_meta.json
│       │       └── channels.txt
│       └── _qc/
│           └── <run_id>/
│               ├── stack_raw_markers.png
│               ├── model_windows.png
│               └── pca_diagnostic.png
└── online/
    ├── ..._IM_online_PCA_online_markers_<run_id>.csv
    ├── ..._IM_online_PCA_online_signal_<run_id>.csv
    ├── ..._IM_online_PCA_decoder_<run_id>.csv
    ├── ..._IM_online_sem_feedback_online_markers_<run_id>.csv
    ├── ..._IM_online_sem_feedback_online_signal_<run_id>.csv
    ├── ..._IM_online_sem_feedback_decoder_<run_id>.csv
    ├── pca_map.json                    # configuração do mapa selecionado
    ├── pca_map.png                     # imagem projetada no Unity
    └── _model/
        └── selected_model.json         # rastreabilidade técnica
```

A mesma regra vale para `EM_treino/train/`. Assim, ao abrir uma pasta de treino, o que fica imediatamente visível são os **dados adquiridos** e o **mapa JSON**. Pickles, metadados e figuras diagnósticas ficam em subpastas.

### Qual `pca_map.json` usar no Unity?

Durante o treino, cada bloco gera o par:

```text
IM_treino/train/pca_map_<run_id>.json
IM_treino/train/pca_map_<run_id>.png
```

O JSON contém a geometria/configuração da representação e o PNG é a imagem efetivamente projetada no Unity. Eles devem permanecer juntos.

Quando você escolhe um modelo para o online, o `main.py` copia os dois para caminhos fixos:

```text
S<session>/online/pca_map.json
S<session>/online/pca_map.png
```

**Esse par é o recomendado para o Unity**, porque não muda entre blocos nem depende de descobrir o `run_id`. Em um treino de duas classes, o JSON descreve LEFT/RIGHT; em três classes, LEFT/BOTH/RIGHT. O JSON também carrega os polígonos numéricos das regiões de densidade usados pelo `intention_control.py`.

### Compatibilidade com versões anteriores

Modelos já criados pela v3/v3.1 em:

```text
S<session>/models/<fase>/<run_id>/
```

continuam sendo encontrados pelo `main.py`. Portanto, uma sessão já iniciada não precisa ser treinada novamente apenas por causa desta reorganização. Novos treinos passam a usar a estrutura `_model/` descrita acima.

## Carregamento do PsychoPy

Para evitar a impressão de que o programa travou durante a inicialização do PsychoPy:

```yaml
psychopy:
  pre_launch_pause_s: 2.0
  startup_grace_s: 10.0
  startup_status_every_s: 5.0
  startup_timeout_s: 90.0
```

Durante a abertura, o `main` mostra mensagens periódicas como `PsychoPy carregando...`. O timeout só é aplicado depois do período de tolerância. Se o processo morrer antes de `BLOCK_END` ou deixar de emitir marcadores além do limite configurado, o bloco é marcado como incompleto e não entra no treino.

## Imagens de estímulo

As imagens não foram enviadas junto com os códigos nesta conversa. Copie os arquivos já existentes do seu projeto para `stims/`:

- `left_foot.png`
- `right_foot.png`
- `both_feet.png`
- `cross.png`
- `rest.png`

Os caminhos no `.psyexp` e na sequência já foram convertidos para caminhos relativos, evitando dependência de `C:\\Users\\...`.
