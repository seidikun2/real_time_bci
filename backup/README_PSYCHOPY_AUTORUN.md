# PsychoPy automático a partir do `main.py`

Esta versão remove a necessidade de abrir o PsychoPy Builder/Runner antes de cada bloco.

## Fluxo novo

Ao iniciar um bloco, `main.py`:

1. inicia sinal (simulado ou g.HIamp), logger e decoder;
2. localiza o Python da instalação do PsychoPy;
3. compila `Graz_UpperLimb.psyexp` diretamente com:
   ```bash
   python -m psychopy.scripts.psyexpCompile Graz_UpperLimb.psyexp -o Graz_UpperLimb_autorun.py
   ```
4. ajusta **somente o script gerado** para usar `subject_id`/`session_id` do `config.yaml` e, por padrão, não mostrar a caixa de diálogo inicial;
5. abre `Graz_UpperLimb_autorun.py` em um processo novo;
6. acompanha os marcadores LSL;
7. ao receber `BLOCK_END`, encerra aquisição/logger e espera o PsychoPy salvar e fechar;
8. se o PsychoPy não fechar, usa `terminate()` e depois força o encerramento da árvore do processo;
9. se o PsychoPy morrer, não iniciar, ou parar de emitir marcadores além do timeout configurado, o bloco é marcado como **incompleto** e não é enviado ao treino.

O arquivo `.psyexp` continua sendo o arquivo mestre e pode ser editado normalmente no Builder.

## Arquivos desta atualização

- `main.py` — integração do processo PsychoPy ao ciclo de cada bloco.
- `psychopy_process.py` — novo módulo de compilação, execução e encerramento do PsychoPy.
- `realtime_signal_transmit.py` — pequena correção para o simulador conseguir sair se o PsychoPy falhar antes de criar o stream de marcadores.
- `config.yaml` — nova seção `psychopy:`.

## Configuração principal

```yaml
psychopy:
  enabled: true
  mode: "psyexp"
  experiment_file: "Graz_UpperLimb.psyexp"
  generated_script: "Graz_UpperLimb_autorun.py"
  fallback_script: "Graz_UpperLimb_lastrun.py"

  python_executable: null
  compile_version: null
  compile_if_changed: true

  inject_session_info: true
  hide_info_dialog: true

  new_console: false
  exit_grace_s: 5.0
  startup_timeout_s: 60.0
  marker_stall_timeout_s: 60.0
  max_block_duration_s: null
```

### `python_executable`

Com `null`, o código tenta:

1. o mesmo Python que executa `main.py`, se tiver PsychoPy;
2. instalações Standalone comuns no Windows, como `C:/Program Files/PsychoPy/python.exe`;
3. Python disponível no `PATH`.

Se a autodetecção não funcionar, configure explicitamente, por exemplo:

```yaml
psychopy:
  python_executable: "C:/Program Files/PsychoPy/python.exe"
```

O executável precisa ser o Python do ambiente em que o seu experimento PsychoPy já funciona, incluindo `pylsl` e demais dependências usadas no `.psyexp`.

## Modos

### `mode: "psyexp"` — recomendado

O `.psyexp` é a fonte. Não depende de `_lastrun.py` previamente criado pela GUI.

### `mode: "auto"`

Tenta compilar o `.psyexp`. Se a compilação falhar e existir `fallback_script`, usa o `_lastrun.py`.

### `mode: "script"`

Não compila `.psyexp`; executa diretamente o `fallback_script`.

## Watchdog

Dois timeouts protegem contra travamentos:

- `startup_timeout_s`: tempo máximo, depois de abrir o processo, até aparecer o primeiro marcador;
- `marker_stall_timeout_s`: tempo máximo sem nenhum novo marcador depois que o protocolo começou.

Nesta versão ambos estão em 60 s. No `.psyexp` enviado, os marcadores aparecem em intervalos muito menores. Se no futuro houver uma rotina legítima com mais de 60 s sem marcadores, aumente esse valor ou use `null` para desabilitar esse watchdog.

`max_block_duration_s` é um limite absoluto opcional para o bloco inteiro.

## Caixa de participante

Com:

```yaml
inject_session_info: true
hide_info_dialog: true
```

o script compilado recebe automaticamente:

- `participant = experiment.subject_id`
- `session = experiment.session_id`

por variáveis de ambiente definidas pelo `main.py`. O `.psyexp` original não é alterado.

## Instalação no projeto atual

Coloque/substitua estes arquivos na pasta onde já estão `Graz_UpperLimb.psyexp`, `stims/` e `stims_sequence.csv`:

```text
main.py
psychopy_process.py
realtime_signal_transmit.py
config.yaml
```

Depois rode somente:

```bash
python main.py
```

Não é necessário abrir o PsychoPy GUI antes.

## Primeira execução

Na primeira execução, observe no console algo semelhante a:

```text
[psychopy] Compilando Graz_UpperLimb.psyexp -> Graz_UpperLimb_autorun.py
[psychopy] Iniciando novo processo: Graz_UpperLimb_autorun.py
[main] Aguardando marcador de fim do PsychoPy: ['99', 'BLOCK_END']
```

Se aparecer a mensagem de que não foi encontrado um Python com PsychoPy, preencha `psychopy.python_executable` no `config.yaml`.
