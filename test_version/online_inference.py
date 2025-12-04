# realtime_inference.py
# -*- coding: utf-8 -*-
"""
Inferência em tempo real:

- Lê um stream LSL de sinal (EEG) pelo nome/tipo definidos no config.
- Carrega artefatos do modelo (cmean, PCA, classificador) da pasta de treino.
- Faz janelas deslizantes (epoch_s, step_s) + bandpass.
- Projeta no espaço tangente + PCA e aplica o classificador.
- Exporta:
    - Stream LSL unificado (3 canais: left, both, right)
    - CSV com tempo, PCA, saída do classificador.

Uso típico (Etapa 3, no main):

    from realtime_inference import run_realtime_decoder
    run_realtime_decoder(cfg, mode="realtime")

O script não se importa se o sinal vem de simulador ou hardware;
ele apenas consome o stream LSL de sinal.
"""

import os
import re
import glob
import csv
import time
import pickle
import datetime as dt
from collections import deque
from typing import List, Optional, Dict, Any

import numpy as np
from scipy import signal
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
from pylsl import (
    StreamInfo,
    StreamOutlet,
    StreamInlet,
    resolve_byprop,
    local_clock,
)

from config_models import AppConfig


# =================== UTIL DE LOG ===================
def log(msg: str) -> None:
    print(f"[decoder] {msg}")


# =============== DIRETÓRIOS E MODELOS ===============
def _make_session_dir(cfg: AppConfig, mode: str) -> str:
    """
    log_root / subject / S{session} / session_type / mode
    """
    return os.path.join(
        cfg.experiment.log_root,
        cfg.experiment.subject_id,
        f"S{cfg.experiment.session_id}",
        cfg.experiment.session_type,
        mode,
    )


def _find_model_prefix(cfg: AppConfig, explicit_prefix: Optional[str] = None) -> str:
    """
    Se `explicit_prefix` for fornecido:
        - se for diretório: usa diretório
        - senão: usa como prefixo direto (base path)

    Caso contrário:
        - procura *_classifier.pkl na pasta de treino da sessão (mode='train')
        - escolhe o mais recente
        - retorna o prefixo SEM o sufixo '_classifier.pkl'
    """
    if explicit_prefix is not None:
        if os.path.isdir(explicit_prefix):
            # diretório com modelos
            cands = glob.glob(os.path.join(explicit_prefix, "*_classifier.pkl"))
            if not cands:
                raise FileNotFoundError(
                    f"Não encontrei *_classifier.pkl em {explicit_prefix}"
                )
            clf_p = max(cands, key=os.path.getmtime)
            prefix = re.sub(r"_classifier\.pkl$", "", clf_p)
            log(f"Modelo selecionado (via diretório): {clf_p}")
            return prefix
        else:
            # prefixo direto
            return explicit_prefix

    train_dir = _make_session_dir(cfg, mode="train")
    pattern   = os.path.join(train_dir, "*_classifier.pkl")
    cands     = glob.glob(pattern)
    if not cands:
        raise FileNotFoundError(
            f"Não encontrei nenhum *_classifier.pkl com padrão:\n  {pattern}"
        )
    clf_p  = max(cands, key=os.path.getmtime)
    prefix = re.sub(r"_classifier\.pkl$", "", clf_p)
    log(f"Modelo selecionado (mais recente em train/): {clf_p}")
    return prefix


def load_artifacts(prefix_or_dir: str):
    """
    Espera encontrar: *_best_c_mean.pkl, *_dim_red.pkl, *_classifier.pkl
    (ou um diretório contendo esses arquivos; escolhe o *_classifier.pkl mais recente).
    """
    if os.path.isdir(prefix_or_dir):
        cands = glob.glob(os.path.join(prefix_or_dir, "*_classifier.pkl"))
        if not cands:
            raise FileNotFoundError("Não encontrei *_classifier.pkl na pasta do modelo.")
        clf_p = max(cands, key=os.path.getmtime)
        base  = re.sub(r"_classifier\.pkl$", "", clf_p)
        cmean_p = base + "_best_c_mean.pkl"
        pca_p   = base + "_dim_red.pkl"
    else:
        base    = prefix_or_dir
        cmean_p = base + "_best_c_mean.pkl"
        pca_p   = base + "_dim_red.pkl"
        clf_p   = base + "_classifier.pkl"

    for p in [cmean_p, pca_p, clf_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Artefato não encontrado: {p}")

    with open(cmean_p, "rb") as f:
        cmean = pickle.load(f)
    with open(pca_p, "rb") as f:
        pca = pickle.load(f)
    with open(clf_p, "rb") as f:
        clf = pickle.load(f)

    log("Artefatos carregados:")
    log(f"  cmean : {cmean_p}")
    log(f"  pca   : {pca_p}")
    log(f"  clf   : {clf_p}")
    return cmean, pca, clf


# ==================== LSL UTILS =====================
def resolve_signal_inlet(
    name: str, stype: str = "EEG", timeout: float = 3.0
) -> StreamInlet:
    label = f"name='{name}'" if name else f"type='{stype}'"
    log(f"Procurando stream LSL do sinal ({label}) ...")
    while True:
        streams = resolve_byprop("name", name, timeout=timeout) if name else []
        if not streams and stype:
            streams = resolve_byprop("type", stype, timeout=timeout)
        if streams:
            si = streams[0]
            log(
                "Conectado ao sinal: "
                f"name={si.name()}, type={si.type()}, "
                f"ch={si.channel_count()}, fs={si.nominal_srate():.2f}"
            )
            return StreamInlet(si, recover=True)
        log("  Sinal não encontrado; tentando novamente em 1s ...")
        time.sleep(1.0)


def make_outlet_unified(name: str, stype: str = "EEG", srate: float = 0.0) -> StreamOutlet:
    """
    Outlet LSL de 3 canais:
      0: left   (probabilidade [0,1] ou valor negativo clippado)
      1: both   (=0)
      2: right  (probabilidade [0,1] ou valor positivo clippado)
    """
    info = StreamInfo(name, stype, 3, srate, "float32", "graz_unified_v2")
    desc = info.desc().append_child("channels")
    for lab, unit in [("left", "a.u."), ("both", "a.u."), ("right", "a.u.")]:
        ch = desc.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "BCI")
    return StreamOutlet(info)


def get_lsl_channel_names(info: StreamInfo) -> List[str]:
    """
    Retorna lista de nomes de canais do stream LSL.
    Se não houver labels no desc(), usa fallback ['ch1', 'ch2', ...].
    """
    C = int(info.channel_count())
    try:
        chs   = info.desc().child("channels")
        names = []
        if chs is not None:
            ch = chs.child("channel")
            while ch is not None and ch.name() == "channel":
                lab = ch.child_value("label") or ""
                lab = lab.strip()
                names.append(lab if lab else None)
                ch = ch.next_sibling()
        if not names or len([n for n in names if n]) < C:
            names = [names[i] if i < len(names) and names[i] else None for i in range(C)]
            names = [n if n else f"ch{i+1}" for i, n in enumerate(names)]
        if len(names) != C:
            if len(names) > C:
                names = names[:C]
            else:
                names = names + [f"ch{i+1}" for i in range(len(names), C)]
        return names
    except Exception:
        return [f"ch{i+1}" for i in range(C)]


def select_channel_indices(
    selection: List[int], ch_names: List[str], index_base: int = 1
) -> List[int]:
    """
    Seleção por índice base-1 (= mesma convenção do cfg.model.select_channels).
    selection vazio/None -> todos os canais.
    """
    C = len(ch_names)
    if not selection:
        return list(range(C))
    idx = [int(v) - (1 if index_base == 1 else 0) for v in selection]
    if any((i < 0 or i >= C) for i in idx):
        raise ValueError(
            f"Algum índice está fora do range [0, {C-1}] (após base). Seleção={idx}"
        )
    return idx


# ==================== PRÉ-PROCESSAMENTO ====================
def design_bandpass(fs: float, order: int, band) -> np.ndarray:
    low, high = float(band[0]), float(band[1])
    if not (0 < low < high < fs / 2):
        raise ValueError(f"Banda inválida {band} para fs={fs}")
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def bp_filtfilt_window(X_win: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(sos, X_win, axis=0, padlen=0)


def window_to_feature(X_win_CxT: np.ndarray, cmean, pca) -> np.ndarray:
    cov = Covariances("oas").transform(X_win_CxT[None, ...])  # (1, C, C)
    ts  = tangent_space(cov, cmean)                           # (1, D)
    Xp  = pca.transform(ts)                                   # (1, p)
    return Xp[0]


# ===================== CSV LOGGER =====================
def timestamp_iso_from_lsl(lsl_time_s: float, unix_offset: float) -> str:
    return dt.datetime.fromtimestamp(lsl_time_s + unix_offset).isoformat(
        timespec="microseconds"
    )


def open_csv_logger(cfg: AppConfig, mode: str):
    """
    Cria arquivo CSV de log com identificadores no nome, se disponíveis:
    ex.: SY100_TEST_S1_IM_treino_decoder_YYYYMMDD_HHMMSS.csv
    """
    out_dir = _make_session_dir(cfg, mode)
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    core = (
        f"{cfg.experiment.subject_id}_"
        f"{cfg.experiment.exp_name}_"
        f"S{cfg.experiment.session_id}_"
        f"{cfg.experiment.session_type}_decoder"
    )

    path = os.path.join(out_dir, f"{core}_{stamp}.csv")
    f    = open(path, "w", newline="", encoding="utf-8")
    w    = csv.writer(f)
    w.writerow(
        [
            "iso_time",
            "lsl_time_s",
            "recv_time_s",
            "pca1",
            "pca2",
            "left",
            "both",
            "right",
        ]
    )
    log(f"Log de inferência: {path}")
    return f, w, path


# ===================== LOOP PRINCIPAL =====================
def run_realtime_decoder(
    cfg: AppConfig,
    mode: str = "realtime",
    model_prefix: Optional[str] = None,
):
    """
    Etapa 3 - Inferência em tempo-real.

    - `mode`: subpasta de sessão (tipicamente "realtime")
    - `model_prefix`: opcional. Se None, procura o classificador mais recente
      na pasta 'train' da mesma sessão.
    """
    decfg = cfg.decoder
    mcfg  = cfg.model

    # 1) Resolve modelo
    prefix = _find_model_prefix(cfg, explicit_prefix=model_prefix)
    cmean, pca, clf = load_artifacts(prefix)

    pca_dim   = getattr(pca, "n_components_", None) or getattr(pca, "n_components", 2)
    pca_dim   = int(pca_dim) if pca_dim else 2
    has_proba = hasattr(clf, "predict_proba")
    log(f"Saída do classificador: {'predict_proba' if has_proba else 'decision_function'}")

    # 2) Lê stream LSL de sinal (tanto faz se é simulado ou real)
    inlet = resolve_signal_inlet(
        name=cfg.lsl.signal_name,
        stype=cfg.lsl.signal_type,
    )
    info  = inlet.info()
    fs    = float(info.nominal_srate())
    C     = int(info.channel_count())
    if fs <= 0:
        raise RuntimeError("Fs nominal inválida no stream LSL.")
    ch_all = get_lsl_channel_names(info)
    log(f"Fs={fs:.2f} Hz, Canais={C}")
    log(f"Canais LSL: {ch_all}")

    # 3) Seleção de canais (mesmo critério do treino: índices base-1)
    sel_idx = select_channel_indices(mcfg.select_channels or [], ch_all, index_base=1)
    ch_sel  = [ch_all[i] for i in sel_idx]
    C_sel   = len(sel_idx)
    log(f"Seleção de canais: {C_sel} -> {ch_sel}")

    # 4) Filtro de banda + parâmetros de janela
    sos = design_bandpass(fs, decfg.filter_order, decfg.band_hz)

    win_n = int(round(decfg.epoch_s * fs))
    hop_n = int(round(decfg.step_s * fs))
    if hop_n <= 0 or win_n <= max(8, 2 * decfg.filter_order):
        raise ValueError("Parâmetros de janela/step inválidos (aumente epoch/step).")
    log(
        f"Janela={win_n} amostras ({decfg.epoch_s:.2f}s), "
        f"Step={hop_n} amostras ({decfg.step_s:.2f}s)"
    )

    # Buffers deslizantes
    buf_X          = deque(maxlen=win_n + 8 * hop_n)
    buf_t          = deque(maxlen=win_n + 8 * hop_n)
    next_compute_at= win_n
    n_samples      = 0

    # 5) Stream LSL de saída
    srate_out           = max(0.0, float(decfg.lsl_rate_hz))
    outlet              = make_outlet_unified(name=decfg.outlet_name, srate=srate_out)
    send_interval       = (1.0 / srate_out) if srate_out > 0 else None
    next_send_monotonic = time.monotonic() if send_interval else None
    latest_vec          = None

    # 6) CSV de inferência
    fcsv, wcsv, csv_path = open_csv_logger(cfg, mode)
    unix_offset = time.time() - local_clock()

    log(
        f"Envio LSL: "
        f"{'desacoplado a ' + str(srate_out) + ' Hz' if send_interval else 'a cada inferência (lsl_rate=0)'}"
    )
    log(f"Nome do stream de saída: {decfg.outlet_name}")
    log("Rodando (Ctrl+C para sair) ...")

    try:
        while True:
            data, ts = inlet.pull_chunk(timeout=0.2, max_samples=8 * hop_n)

            # envia o último vetor na taxa desejada (quando desacoplado)
            if send_interval and latest_vec is not None:
                now_mono = time.monotonic()
                if now_mono >= next_send_monotonic:
                    outlet.push_sample(latest_vec, timestamp=local_clock())
                    next_send_monotonic = now_mono + send_interval

            if not ts:
                continue

            for x, t_lsl in zip(data, ts):
                x = np.asarray(x, dtype=float)
                if x.size < max(sel_idx) + 1:
                    continue
                x_sel = x[sel_idx]
                buf_X.append(x_sel)
                buf_t.append(float(t_lsl))
                n_samples += 1

                # Janela cheia -> inferência
                if n_samples >= next_compute_at and len(buf_X) >= win_n:
                    X_win = np.vstack(list(buf_X)[-win_n:])
                    t_win = np.asarray(list(buf_t)[-win_n:], dtype=float)

                    try:
                        Xf = bp_filtfilt_window(X_win, sos)
                    except Exception:
                        Xf = X_win

                    feat = window_to_feature(Xf.T, cmean, pca)
                    p1   = float(feat[0]) if pca_dim >= 1 else 0.0
                    p2   = float(feat[1]) if pca_dim >= 2 else 0.0

                    # Saída do classificador -> left/right
                    if has_proba:
                        proba   = clf.predict_proba(feat.reshape(1, -1))[0]
                        classes = list(getattr(clf, "classes_", []))

                        left_label  = decfg.left_label
                        right_label = decfg.right_label

                        if (left_label in classes) and (right_label in classes):
                            li, ri = classes.index(left_label), classes.index(right_label)
                            left, right = float(proba[li]), float(proba[ri])
                        else:
                            # Heurísticas
                            u = [str(c).upper() for c in classes]
                            if "LEFT" in u and "RIGHT" in u:
                                left = float(proba[u.index("LEFT")])
                                right = float(proba[u.index("RIGHT")])
                            elif -1 in classes and 1 in classes:
                                left = float(proba[classes.index(-1)])
                                right = float(proba[classes.index(1)])
                            elif 0 in classes and 1 in classes:
                                left = float(proba[classes.index(0)])
                                right = float(proba[classes.index(1)])
                            else:
                                # fallback: assume binário [0,1]
                                left, right = float(proba[0]), float(proba[1])
                        both = 0.0
                    else:
                        raw     = float(clf.decision_function(feat.reshape(1, -1))[0])
                        out_val = float(np.clip(raw, -1.0, 1.0))
                        if out_val < 0:
                            left, right = out_val, 0.0
                        elif out_val > 0:
                            left, right = 0.0, out_val
                        else:
                            left, right = 0.0, 0.0
                        both = 0.0

                    # Vetor unificado para o outlet
                    latest_vec  = [-right, -right, left]
                    print(latest_vec)

                    t_out       = float(t_win[-1])
                    recv_time_s = time.time()
                    iso         = timestamp_iso_from_lsl(t_out, unix_offset)

                    # Log CSV
                    wcsv.writerow(
                        [
                            iso,
                            f"{t_out:.9f}",
                            f"{recv_time_s:.6f}",
                            f"{p1:.6f}",
                            f"{p2:.6f}",
                            f"{left:.6f}",
                            f"{both:.6f}",
                            f"{right:.6f}",
                        ]
                    )

                    # Se não tiver taxa fixa, envia a cada inferência
                    if not send_interval and latest_vec is not None:
                        outlet.push_sample(latest_vec, timestamp=local_clock())

                    next_compute_at += hop_n

    except KeyboardInterrupt:
        log("Interrompido pelo usuário.")
    finally:
        try:
            fcsv.close()
        except Exception:
            pass
        log(f"Arquivo salvo: {csv_path}")


# Execução standalone (debug isolado)
if __name__ == "__main__":
    from config_models import load_config

    cfg = load_config("config.yaml")
    run_realtime_decoder(cfg, mode="realtime", model_prefix=None)
