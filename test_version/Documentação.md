# Guia rápido - Real Time BCI

Documentação sucinta para uso do código de BCI em tempo real. O objetivo é permitir que o sistema seja utilizado de forma simples, mesmo por pessoas sem conhecimento aprofundado do código.

---

## 1. Visão geral

Este projeto implementa um fluxo de BCI em tempo real baseado em LSL.

O arquivo principal é:

```text
main.py
```

Ele organiza três etapas:

1. Treino de imagética motora
2. Calibração do classificador
3. Inferência em tempo real

O sinal EEG pode vir de duas fontes:

- **Sinal simulado**, gerado pelo próprio código e transmitido via LSL.
- **Sinal real**, vindo de um equipamento ou software externo que transmita EEG via LSL.

As marcações experimentais vêm de um ambiente **PsychoPy**, que também precisa estar rodando para enviar os eventos via LSL.

---

## 2. Arquivos principais

| Arquivo | Função |
|---|---|
| `main.py` | Organiza o fluxo geral: treino, calibração, visualização e tempo real. |
| `config.yaml` | Define sujeito, sessão, caminhos, nomes dos streams LSL, parâmetros do modelo e modo de execução. |
| `receive_data_log.py` | Recebe sinal e marcadores via LSL e salva os arquivos CSV. |
| `realtime_signal_transmit.py` | Gera sinal EEG simulado e transmite via LSL. |
| `decoder_calibration.py` | Lê os dados de treino, extrai épocas e treina o classificador. |
| `online_inference.py` | Carrega o modelo calibrado e roda a inferência em tempo real. |
| `check_data.py` | Visualiza rapidamente os dados gravados para controle de qualidade. |
| `Graz_UpperLimb.py` | Protocolo PsychoPy que envia os marcadores via LSL. |

---

## 3. Fluxo geral do sistema

```text
PsychoPy
  └── envia marcadores via LSL
        ↓
main.py
  ├── recebe marcadores via LSL
  ├── recebe sinal EEG real ou simulado via LSL
  ├── salva CSVs de treino
  ├── calibra classificador
  ├── permite visualizar dados
  └── roda inferência em tempo real
```

---

## 4. Configuração principal

Antes de rodar, edite o arquivo:

```text
config.yaml
```

Confira principalmente:

```yaml
experiment:
  subject_id: "LC101"
  session_id: 2
  log_root: "C:/Users/User/Desktop/Dados"
  exp_name: "TEST"
  session_type: "IM_treino"
```

Os dados serão salvos em:

```text
log_root / subject_id / S<session_id> / session_type / train
```

Exemplo:

```text
C:/Users/User/Desktop/Dados/LC101/S2/IM_treino/train
```

---

## 5. Configuração dos streams LSL

No `config.yaml`, confira:

```yaml
lsl:
  marker_name: "GrazMI_Markers"
  marker_type: "Markers"
  signal_name: "Cognionics Wireless EEG"
  signal_type: "EEG"
```

O stream de marcadores precisa bater com o PsychoPy.

O protocolo PsychoPy cria um stream com:

```text
name = GrazMI_Markers
type = Markers
```

O stream de sinal precisa bater com o equipamento real ou com o simulador.

---

## 6. Escolha entre sinal simulado e sinal real

### Opção A - Sinal simulado

Use esta configuração:

```yaml
runtime:
  simulate_signal: true
```

Nesse modo:

- O próprio código cria um sinal EEG simulado.
- O sinal simulado é enviado via LSL.
- O PsychoPy ainda precisa rodar para enviar os marcadores.

Esse modo é útil para testes, treinamento e validação do fluxo.

### Opção B - Sinal real

Use esta configuração:

```yaml
runtime:
  simulate_signal: false
```

Nesse modo:

- O código não cria sinal simulado.
- O sinal EEG precisa vir de um equipamento ou software externo via LSL.
- O PsychoPy precisa rodar para enviar os marcadores.

Esse modo é o esperado para coleta real.

---

## 7. Modos de execução

O campo `start_mode` define o comportamento do `main.py`.

```yaml
runtime:
  start_mode: "full_loop"
```

### `full_loop`

Roda treino e calibração em loop até o usuário aprovar as métricas. Depois inicia o tempo real.

Use quando quiser repetir a calibração até obter um resultado aceitável.

### `train_once`

Roda treino e calibração uma única vez. Depois pergunta se deve iniciar o tempo real.

Use quando quiser um fluxo mais direto.

### `realtime_only`

Pula treino e calibração e entra direto no tempo real.

Use apenas quando já existir um modelo calibrado salvo na pasta `train`.

---

## 8. Checklist antes de rodar

- [ ] Conferi o `config.yaml`.
- [ ] O `subject_id` está correto.
- [ ] O `session_id` está correto.
- [ ] O `log_root` aponta para a pasta onde quero salvar os dados.
- [ ] O `session_type` está correto.
- [ ] O `simulate_signal` está definido corretamente.
- [ ] O `start_mode` está definido corretamente.
- [ ] O PsychoPy está aberto.
- [ ] O protocolo PsychoPy está pronto para rodar.
- [ ] O stream de marcadores do PsychoPy está ativo ou será iniciado junto com o protocolo.
- [ ] Se for sinal real, o software de aquisição EEG está transmitindo via LSL.
- [ ] Se for sinal simulado, o `simulate_signal` está como `true`.
- [ ] Os nomes dos streams LSL no `config.yaml` batem com os streams reais.
- [ ] O terminal está aberto na pasta correta do projeto.

---

## 9. Como rodar

Abra o PowerShell e entre na pasta do projeto:

```powershell
cd "C:\Users\User\Documents\GitHub\real_time_bci\test_version"
```

Depois rode:

```powershell
C:/Users/User/anaconda3/python.exe main.py
```

Durante o treino, o terminal mostrará mensagens indicando se está em modo simulado ou modo de aquisição real.

Ao final do protocolo de treino no PsychoPy, volte ao terminal e pressione ENTER para encerrar a etapa de treino.

---

## 10. O que deve aparecer no terminal

Durante a execução, procure mensagens parecidas com:

```text
Conectado: name=GrazMI_Markers, type=Markers
Conectado: name=Cognionics Wireless EEG, type=EEG
(acumulado) marcadores=..., amostras_sinal=...
```

Isso indica que o código está recebendo tanto os marcadores quanto o sinal.

Se aparecer:

```text
marcadores=0
```

provavelmente o stream de marcadores do PsychoPy não está chegando.

---

## 11. Arquivos gerados

Durante o treino, o receptor salva arquivos como:

```text
LC101_TEST_S2_IM_treino_train_markers_YYYYMMDD_HHMMSS.csv
LC101_TEST_S2_IM_treino_train_signal_YYYYMMDD_HHMMSS.csv
```

Depois da calibração, também são salvos arquivos do modelo:

```text
*_best_c_mean.pkl
*_dim_red.pkl
*_classifier.pkl
*_meta.json
*_channels.txt
*_pca.png
```

Esses arquivos são usados posteriormente pela inferência em tempo real.

---

## 12. Visualização dos dados

Depois da calibração, o programa pergunta:

```text
Deseja visualizar os dados (check_data)? [s/N]:
```

Se responder `s`, o código gera visualizações para controle de qualidade.

As figuras mostram:

- Sinal contínuo com marcações.
- Épocas alinhadas ao evento `ATTEMPT`.
- Separação entre classes `LEFT_MI_STIM` e `RIGHT_MI_STIM`.

---

## 13. Tempo real

Na etapa de tempo real, o sistema:

1. Recebe sinal via LSL.
2. Carrega o modelo calibrado mais recente.
3. Aplica janelas deslizantes no sinal.
4. Filtra os dados.
5. Projeta no espaço tangente e no PCA.
6. Classifica a janela.
7. Envia a saída por LSL.

A saída é um stream LSL com três canais:

```text
left
both
right
```

Esse stream pode ser usado por outro programa, por exemplo Unity ou outro ambiente de feedback.

---

## 14. Problemas comuns

### Erro: `config.yaml` não encontrado

Rode o script a partir da pasta correta ou garanta que o `main.py` carregue o config pela pasta do próprio arquivo.

Comando recomendado:

```powershell
cd "C:\Users\User\Documents\GitHub\real_time_bci\test_version"
C:/Users/User/anaconda3/python.exe main.py
```

### Nenhum marcador recebido

Verifique:

- [ ] O PsychoPy está rodando?
- [ ] O protocolo realmente iniciou?
- [ ] O stream se chama `GrazMI_Markers`?
- [ ] O tipo do stream é `Markers`?
- [ ] O `config.yaml` tem os mesmos nomes?

### Sinal recebido, mas marcadores zerados

Isso geralmente indica que o stream EEG está ativo, mas o stream de marcadores não está chegando.

O arquivo de sinal será salvo, mas não será possível gerar épocas válidas para calibração.

### Nenhum par `markers/signal` encontrado

A calibração e o `check_data.py` procuram arquivos com nomes no padrão:

```text
*_markers_*.csv
*_signal_*.csv
```

Se existirem arquivos antigos como:

```text
train_markers.csv
train_signal.csv
```

eles podem não ser reconhecidos automaticamente.

---

## 15. Sugestões de melhoria no código

### 1. Remover logs duplicados do simulador

Atualmente, o simulador pode gerar arquivos próprios como:

```text
train_signal.csv
train_markers.csv
```

Isso pode confundir, porque o receptor principal já salva os arquivos no formato correto.

Sugestão:

- Deixar o `realtime_signal_transmit.py` apenas transmitir sinal via LSL.
- Deixar apenas o `receive_data_log.py` responsável por salvar CSVs.

### 2. Fazer o receptor aguardar os streams

Hoje, se o stream LSL não for encontrado, o código pode gerar erro.

Sugestão: fazer o receptor esperar até o stream aparecer, exibindo mensagens como:

```text
Aguardando stream LSL de marcadores...
Aguardando stream LSL de sinal...
```

Isso facilita o uso por pessoas que não conhecem o código.

### 3. Adicionar uma checagem inicial

Antes de iniciar o treino, o `main.py` poderia mostrar:

```text
[OK] Config carregado
[OK] Pasta de saída criada
[OK] Stream de marcadores encontrado
[OK] Stream de sinal encontrado
```

Isso ajudaria a evitar coletas sem marcador ou sem sinal.

### 4. Corrigir a frequência usada na calibração

Se o sinal simulado usa `fs = 250 Hz`, a calibração também deve usar `250 Hz` ou estimar automaticamente a frequência a partir dos timestamps.

### 5. Usar seleção de canais específica no `check_data.py`

O `check_data.py` poderia usar `check_data.select_channels` em vez de sempre usar `model.select_channels`.

Assim, a visualização poderia ter uma seleção de canais independente do modelo.

---

## 16. Ordem recomendada de uso

1. Editar `config.yaml`.
2. Abrir o ambiente PsychoPy.
3. Preparar o protocolo de imagética.
4. Se for sinal real, iniciar o stream EEG via LSL.
5. Rodar `main.py`.
6. Iniciar o protocolo PsychoPy.
7. Ao final do protocolo, pressionar ENTER no terminal.
8. Revisar a acurácia da calibração.
9. Visualizar os dados se necessário.
10. Aprovar ou repetir o treino.
11. Entrar no modo tempo real.

---

## 17. Resumo operacional

Para teste com sinal simulado:

```yaml
runtime:
  simulate_signal: true
  start_mode: "train_once"
```

Depois rode:

```powershell
cd "C:\Users\User\Documents\GitHub\real_time_bci\test_version"
C:/Users/User/anaconda3/python.exe main.py
```

Para coleta com sinal real:

```yaml
runtime:
  simulate_signal: false
  start_mode: "full_loop"
```

Depois:

1. Inicie o EEG via LSL.
2. Inicie o PsychoPy.
3. Rode o `main.py`.
4. Execute o protocolo.
5. Calibre o modelo.
6. Entre no tempo real.

---

## 18. Observação final

O ponto mais importante para o funcionamento do sistema é garantir que **dois streams LSL estejam disponíveis**:

```text
1. Stream de sinal EEG
2. Stream de marcadores PsychoPy
```

Sem o stream de sinal, não há dados para classificar.

Sem o stream de marcadores, não há como segmentar as tentativas para calibração.
