# analyze_bci_session_riemann_potato.py

Script offline para analisar uma pasta de sessão do protocolo BCI Graz MI.

## Estrutura esperada

A pasta passada em `--session-folder` deve ter algo como:

```text
SY001/S3/
├── EM_treino/train/*_markers_<run_id>.csv
├── EM_treino/train/*_signal_<run_id>.csv
├── IM_treino/train/*_markers_<run_id>.csv
├── IM_treino/train/*_signal_<run_id>.csv
└── IM_online/realtime/*_markers_<run_id>.csv
└── IM_online/realtime/*_signal_<run_id>.csv
```

O pareamento é estrito por `run_id`, por exemplo:

```text
..._markers_20260622_153353.csv
..._signal_20260622_153353.csv
```

## Uso recomendado

```bash
python analyze_bci_session_riemann_potato.py \
  --session-folder "C:/Users/Unifesp/Desktop/Dados Seidi/SY001/S3" \
  --config "config.yaml" \
  --mi-model-run-id 20260622_153353
```

Ou selecionando diretamente o arquivo de sinal MI usado como template do online:

```bash
python analyze_bci_session_riemann_potato.py \
  --session-folder "C:/Users/Unifesp/Desktop/Dados Seidi/SY001/S3" \
  --config "config.yaml" \
  --mi-model-signal "C:/.../IM_treino/train/SY001_TEST_S3_IM_treino_train_signal_20260622_153353.csv"
```

Se nenhum modelo MI for informado, o script usa automaticamente o bloco MI com melhor balanced accuracy média na validação cruzada.

## Saídas principais

Dentro das próprias pastas `train` e `realtime`, em `analysis_riemann_potato/<run_id>/`:

- `*_cv_trial_curves.png`: curvas de classificação por tentativa na validação cruzada.
- `*_pca_potato_svm_rbf.png`: PCA após limpeza por Potato + superfície SVM RBF.
- `*_online_decode_timeline.png`: timeline online de P(LEFT) e P(RIGHT).
- `*_online_trial_curves.png`: médias online por trial/classe.
- `*_online_pca_template.png`: online projetado no mesmo PCA/template do MI selecionado.
- CSVs com predições por janela.
- JSONs com resumo por bloco.

No nível da sessão:

- `training_cv_summary.csv`
- `training_cv_summary.png`
- `online_summary.csv`
- `selected_mi_model.pkl`
- `selected_mi_model.json`
- `EM_treino_session_cv/` e `IM_treino_session_cv/` com validação cruzada agregada por fase.

## Dependências

```bash
pip install numpy pandas scipy scikit-learn matplotlib pyyaml pyriemann
```

`pyriemann` é recomendado. Se ele não estiver instalado, o script usa fallback OAS + log-Euclidean/tangent space, mas mantém a lógica de Potato por distância Riemanniana robusta.
