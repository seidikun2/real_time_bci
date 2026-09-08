# Ajuste do fluxo main + g.HIamp

## Regra atual

- `runtime.simulate_signal: true`  → `main.py` inicia `realtime_signal_transmit.py`.
- `runtime.simulate_signal: false` → `main.py` inicia `input_hiamp.py`, que configura o g.HIamp e publica `cfg.lsl.signal_name`/`cfg.lsl.signal_type` em LSL.

A gravação sincronizada de EEG + marcadores continua concentrada em `receive_data_log.py`, tanto no modo simulado quanto no modo real. Isso evita dois formatos de CSV diferentes entrando no treino.

## Arquivos alterados

- `main.py`
  - Importa `run_sim_transmission` de `realtime_signal_transmit.py`.
  - Importa `run_hiamp_transmission` de `input_hiamp.py`.
  - Usa `input_hiamp.py` quando `simulate_signal = false`.
  - Usa `cfg.decoder.outlet_name`/`cfg.decoder.outlet_type` como padrão do plot online.

- `input_hiamp.py`
  - Refeito como transmissor real g.HIamp → LSL.
  - Usa `cfg.fs_hz` e `cfg.n_channels`, não mais parâmetros de simulação.
  - Não grava CSV próprio; a gravação fica no `receive_data_log.py`.
  - Lê `hiamp.gtec_root` do `config.yaml` ou a variável de ambiente `GTEC_ROOT`.

- `receive_data_log.py`
  - Agora espera os streams LSL ficarem disponíveis, em vez de falhar após poucos segundos.

- `online_inference.py`
  - Usa `cfg.decoder.outlet_type` no stream de saída do decoder.

- `config.yaml`
  - Adicionada seção `hiamp` para o caminho da API g.tec e tamanho de bloco.

## Uso

Para simular:

```yaml
runtime:
  simulate_signal: true
```

Para usar o g.HIamp real:

```yaml
runtime:
  simulate_signal: false

hiamp:
  gtec_root: "D:/Documentos/gtec/gNEEDaccessClientAPI"
  chunk: 10
```

## Observação

O `input_hiamp.py` depende do `pygds` e dos arquivos da API g.tec no Windows. Ele não é executável em máquinas sem o driver/API da g.tec, mas o código foi validado por compilação estática (`py_compile`).

## Atualização: CH16 como referência + CAR

Nesta versão, quando `runtime.simulate_signal = false`, o `input_hiamp.py`:

1. adquire 16 canais físicos do g.HIamp (`hiamp.acquisition_channels: 16`);
2. usa o canal 16 como referência de software (`hiamp.reference_channel: 16`);
3. remove o CH16 do stream publicado (`hiamp.drop_reference_channel: true`);
4. aplica Common Average Reference nos canais restantes (`hiamp.reference_mode: "car"`);
5. publica 15 canais EEG no LSL (`n_channels: 15`).

Assim, `receive_data_log.py`, `decoder_calibration.py`, `check_data.py` e `online_inference.py`
recebem diretamente o sinal já em CAR, com CH1..CH15.


## Correção de sincronização markers/signal

Nesta versão, `receive_data_log.py` gera um `run_id` único por bloco e usa o mesmo `run_id` no arquivo de marcadores e no arquivo de sinal:

- `..._markers_YYYYMMDD_HHMMSS.csv`
- `..._signal_YYYYMMDD_HHMMSS.csv`

O `main.py`, `decoder_calibration.py` e `check_data.py` só pareiam arquivos com o mesmo `run_id`. Isso evita o erro em que um marcador antigo era combinado com o sinal mais recente da pasta.

O `decoder_calibration.py` também imprime os ranges temporais de `markers` e `signal` e interrompe o treino com uma mensagem explícita se não houver sobreposição temporal suficiente para janelar.
