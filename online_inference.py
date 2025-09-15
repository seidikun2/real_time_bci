# -*- coding: utf-8 -*-
"""
graz_realtime_decoder.py
Decodificador em tempo real para MI (Graz), com saída LSL (3 canais):
  [left, both(=0), right]

Envio LSL com taxa configurável (--lsl_rate, ex.: 30 Hz):
- O pipeline calcula inferências no passo definido por --step.
- O envio LSL é desacoplado: só envia na frequência pedida e SEMPRE usa a
  classificação mais recente (descarta frames atrasados para não entupir).

Saída do modelo:
- Se o classificador tiver predict_proba (binário): usa probabilidades (0..1).
- Caso contrário: usa decision_function, clipe em [-1,1] e envia valor negativo no left,
  positivo no right (ambos em "a.u."), como no seu exemplo.
"""

import os
import re
import csv
import glob
import time
import pickle
import argparse
import datetime as dt
from collections import deque

import numpy as np
from scipy import signal
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

# ===================== Configs padrão =====================
DEFAULT_OUT_DIR       = r"C:\Users\Unifesp\Desktop\Dados Seidi"
DEFAULT_SIGNAL_NAME   = "Cognionics Wireless EEG"
# ==========================================================

def log(msg: str):
    print(f"[decoder] {msg}")

# ------------------------ LSL utils ------------------------
def resolve_signal_inlet(name: str = None, stype: str = "EEG", timeout: float = 3.0) -> StreamInlet:
    label = f"name='{name}'" if name else f"type='{stype}'"
    log(f"Procurando stream LSL do sinal ({label}) ...")
    while True:
        streams = resolve_byprop('name', name, timeout=timeout) if name else []
        if not streams and stype:
            streams = resolve_byprop('type', stype, timeout=timeout)
        if streams:
            si = streams[0]
            log(f"Conectado ao sinal: name={si.name()}, type={si.type()}, ch={si.channel_count()}, fs={si.nominal_srate():.2f}")
            return StreamInlet(si, recover=True)
        log("  Sinal não encontrado; tentando novamente em 1s ...")
        time.sleep(1.0)

def make_outlet_unified(name="Signal", stype="EEG", srate: float = 0.0) -> StreamOutlet:
    """
    Outlet LSL de 3 canais:
      0: left   (probabilidade [0,1] ou valor negativo clippado)
      1: both   (=0)
      2: right  (probabilidade [0,1] ou valor positivo clippado)
    """
    info = StreamInfo(name, stype, 3, srate, 'float32', 'graz_unified_v2')
    desc = info.desc().append_child("channels")
    for lab, unit in [("left", "a.u."), ("both", "a.u."), ("right", "a.u.")]:
        ch = desc.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "BCI")
    return StreamOutlet(info)

# ------------------------ Modelo ------------------------
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

    with open(cmean_p, "rb") as f: cmean = pickle.load(f)
    with open(pca_p,   "rb") as f: pca   = pickle.load(f)
    with open(clf_p,   "rb") as f: clf   = pickle.load(f)

    log("Artefatos carregados:")
    log(f"  cmean : {cmean_p}")
    log(f"  pca   : {pca_p}")
    log(f"  clf   : {clf_p}")
    return cmean, pca, clf

# ------------------------ Pré-processamento ------------------------
def design_bandpass(fs: float, order: int, band):
    low, high = float(band[0]), float(band[1])
    if not (0 < low < high < fs/2):
        raise ValueError(f"Banda inválida {band} para fs={fs}")
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

def bp_filtfilt_window(X_win: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(sos, X_win, axis=0, padlen=0)

def window_to_feature(X_win_CxT: np.ndarray, cmean, pca):
    cov = Covariances("oas").transform(X_win_CxT[None, ...])  # (1, C, C)
    ts  = tangent_space(cov, cmean)                           # (1, D)
    Xp  = pca.transform(ts)                                   # (1, p)
    return Xp[0]

# ------------------------ CSV logger ------------------------
def timestamp_iso_from_lsl(lsl_time_s: float, unix_offset: float) -> str:
    return dt.datetime.fromtimestamp(lsl_time_s + unix_offset).isoformat(timespec="microseconds")

def open_csv_logger(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"graz_decoder_{stamp}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow([
        "iso_time","lsl_time_s","recv_time_s",
        "pca1","pca2",
        "left","both","right"   # o que foi publicado no LSL
    ])
    log(f"Log de inferência: {path}")
    return f, w, path

# ------------------------ Loop principal ------------------------
def run(signal_name: str,
        model_prefix_or_dir: str,
        out_dir: str,
        epoch_s: float,
        step_s: float,
        band,
        order: int,
        outlet_name: str,
        lsl_rate: float,
        left_label=None,
        right_label=None):

    # Conecta ao sinal
    inlet = resolve_signal_inlet(name=signal_name, stype="EEG")
    info = inlet.info()
    fs = float(info.nominal_srate()); C = int(info.channel_count())
    if fs <= 0: raise RuntimeError("Fs nominal inválida no stream LSL.")
    log(f"Fs={fs:.2f} Hz, Canais={C}")

    # Carrega modelo
    cmean, pca, clf = load_artifacts(model_prefix_or_dir)
    pca_dim = getattr(pca, "n_components_", None) or getattr(pca, "n_components", 2)
    pca_dim = int(pca_dim) if pca_dim else 2
    has_proba = hasattr(clf, "predict_proba")
    log(f"Saída do classificador: {'predict_proba' if has_proba else 'decision_function'}")

    # Filtro
    sos = design_bandpass(fs, order, band)

    # Janela e hop (taxa de inferência ~ 1/step_s)
    win_n = int(round(epoch_s * fs))
    hop_n = int(round(step_s * fs))
    if hop_n <= 0 or win_n <= max(8, 2*order):
        raise ValueError("Parâmetros de janela/step inválidos (aumente --epoch e/ou --step).")
    log(f"Janela={win_n} amostras ({epoch_s:.2f}s), Step={hop_n} amostras ({step_s:.2f}s)")

    # Buffers
    buf_X = deque(maxlen=win_n + 8*hop_n)
    buf_t = deque(maxlen=win_n + 8*hop_n)
    next_compute_at = win_n
    n_samples = 0

    # Saída LSL (taxa de envio configurável)
    srate_out = max(0.0, float(lsl_rate))
    outlet = make_outlet_unified(name=outlet_name, srate=srate_out)
    send_interval = (1.0 / srate_out) if srate_out > 0 else None
    next_send_monotonic = time.monotonic() if send_interval else None
    latest_vec = None  # última classificação [left,both,right]

    # CSV
    fcsv, wcsv, csv_path = open_csv_logger(out_dir)
    unix_offset = time.time() - local_clock()

    log(f"Envio LSL: {'desacoplado a ' + str(lsl_rate) + ' Hz' if send_interval else 'a cada inferência (lsl_rate=0)'}")
    log("Rodando (Ctrl+C para sair) ...")
    try:
        while True:
            # ===== 1) Receber chunk do LSL de entrada =====
            data, ts = inlet.pull_chunk(timeout=0.2, max_samples=8*hop_n)

            # ===== 2) Atualizar scheduler de envio (não bloquear) =====
            if send_interval and latest_vec is not None:
                now_mono = time.monotonic()
                if now_mono >= next_send_monotonic:
                    # Envia SOMENTE a última classificação disponível (descarta atrasados)
                    outlet.push_sample(latest_vec, timestamp=local_clock())
                    # agenda o próximo envio sem "catch-up" para não entupir
                    next_send_monotonic = now_mono + send_interval

            if not ts:
                continue

            # ===== 3) Alimentar buffers e, quando for hora, inferir =====
            for x, t_lsl in zip(data, ts):
                buf_X.append(np.asarray(x, dtype=float))
                buf_t.append(float(t_lsl))
                n_samples += 1

                if n_samples >= next_compute_at and len(buf_X) >= win_n:
                    # --- janela (T,C) ---
                    X_win = np.vstack(list(buf_X)[-win_n:])
                    t_win = np.asarray(list(buf_t)[-win_n:], dtype=float)

                    # --- filtra ---
                    try:
                        Xf = bp_filtfilt_window(X_win, sos)
                    except Exception:
                        Xf = X_win  # fallback seguro

                    # --- features: Cov -> TS -> PCA ---
                    feat = window_to_feature(Xf.T, cmean, pca)   # (pca_dim,)
                    p1 = float(feat[0]) if pca_dim >= 1 else 0.0
                    p2 = float(feat[1]) if pca_dim >= 2 else 0.0

                    # --- saída do modelo ---
                    if has_proba:
                        proba = clf.predict_proba(feat.reshape(1, -1))[0]   # (2,)
                        classes = list(getattr(clf, "classes_", []))
                        # mapeia classes -> (left,right)
                        if (left_label in classes) and (right_label in classes):
                            li, ri = classes.index(left_label), classes.index(right_label)
                            left, right = float(proba[li]), float(proba[ri])
                        else:
                            u = [str(c).upper() for c in classes]
                            if "LEFT" in u and "RIGHT" in u:
                                left, right = float(proba[u.index("LEFT")]), float(proba[u.index("RIGHT")])
                            elif -1 in classes and 1 in classes:
                                left, right = float(proba[classes.index(-1)]), float(proba[classes.index(1)])
                            elif 0 in classes and 1 in classes:
                                left, right = float(proba[classes.index(0)]), float(proba[classes.index(1)])
                            else:
                                left, right = float(proba[0]), float(proba[1])
                        both = 0.0
                    else:
                        # decision_function -> lógica idêntica ao seu exemplo
                        raw = float(clf.decision_function(feat.reshape(1, -1))[0])
                        out_val = float(np.clip(raw, -1.0, 1.0))
                        if out_val < 0:
                            left, right = out_val, 0.0
                        elif out_val > 0:
                            left, right = 0.0, out_val
                        else:
                            left, right = 0.0, 0.0
                        both = 0.0

                    # --- registra a classificação mais recente para o scheduler de envio ---
                    latest_vec = [-right, both, -left]

                    # --- tempos (para o CSV apenas quando há nova inferência) ---
                    t_out = float(t_win[-1])
                    recv_time_s = time.time()
                    iso = timestamp_iso_from_lsl(t_out, unix_offset)

                    # --- log CSV (1 linha por inferência, não por envio) ---
                    wcsv.writerow([
                        iso, f"{t_out:.9f}", f"{recv_time_s:.6f}",
                        f"{p1:.6f}", f"{p2:.6f}",
                        f"{left:.6f}", f"{both:.6f}", f"{right:.6f}"
                    ])

                    # --- se não houver throttling, envia já (timestamp no clock local) ---
                    if not send_interval:
                        outlet.push_sample(latest_vec, timestamp=local_clock())

                    # próxima janela de inferência
                    next_compute_at += hop_n

    except KeyboardInterrupt:
        log("Interrompido pelo usuário.")
    finally:
        try: fcsv.close()
        except Exception: pass
        log(f"Arquivo salvo: {csv_path}")

def main():
    ap = argparse.ArgumentParser(
        description=("Decodificador MI (Graz) em tempo real. "
                     "Saída LSL 3c (left, both=0, right). "
                     "Envio desacoplado por --lsl_rate usando a última classificação (drop de frames).")
    )
    ap.add_argument("--signal_name", default=DEFAULT_SIGNAL_NAME, help="Nome do stream LSL do EEG (ex.: 'GrazMI_SimEEG').")
    ap.add_argument("--model_prefix", default=None,
                    help=("Prefixo dos arquivos do modelo (ex: C:\\..\\graz_model_PREFIX) "
                          "ou pasta contendo *_classifier.pkl/_dim_red.pkl/_best_c_mean.pkl. "
                          "Se None, busca o mais recente em --out_dir."))
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Pasta para salvar o CSV.")
    ap.add_argument("--epoch", type=float, default=2.0, help="Janela (s).")
    ap.add_argument("--step",  type=float, default=0.05, help="Passo entre inferências (s).")
    ap.add_argument("--band",  type=float, nargs=2, default=(8.0, 30.0), help="Bandpass (low high) em Hz.")
    ap.add_argument("--order", type=int, default=4, help="Ordem do filtro Butter.")
    ap.add_argument("--outlet_name", default="Signal", help="Nome do stream LSL de saída.")
    ap.add_argument("--lsl_rate", type=float, default=30.0,
                    help="Frequência de envio LSL em Hz. Use 0 para 'enviar a cada inferência'.")
    ap.add_argument("--left_label", default=None, help="Rótulo da classe LEFT (se usar predict_proba).")
    ap.add_argument("--right_label", default=None, help="Rótulo da classe RIGHT (se usar predict_proba).")
    args = ap.parse_args()

    # Resolve modelo (prefixo ou diretório). Se None, pega o mais recente em out_dir.
    model_arg = args.model_prefix
    if model_arg is None:
        cands = glob.glob(os.path.join(args.out_dir, "*_classifier.pkl"))
        if not cands:
            raise FileNotFoundError("Não encontrei *_classifier.pkl em --out_dir. Informe --model_prefix.")
        clf_p = max(cands, key=os.path.getmtime)
        model_arg = re.sub(r"_classifier\.pkl$", "", clf_p)

    run(signal_name=args.signal_name,
        model_prefix_or_dir=model_arg,
        out_dir=args.out_dir,
        epoch_s=args.epoch,
        step_s=args.step,
        band=tuple(args.band),
        order=args.order,
        outlet_name=args.outlet_name,
        lsl_rate=args.lsl_rate,
        left_label=args.left_label,
        right_label=args.right_label)

if __name__ == "__main__":
    main()
