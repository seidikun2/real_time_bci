# graz_realtime_decoder.py
import os
import re
import time
import glob
import pickle
import argparse
import datetime as dt
from collections import deque

import numpy as np
from scipy import signal
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

# ===================== Config Padrão =====================
DEFAULT_OUT_DIR     = r"C:\Users\User\Desktop\Dados"
DEFAULT_SIGNAL_NAME = "GrazMI_SimEEG"   # stream de sinal
# ========================================================

def log(msg): print(f"[decoder] {msg}")

# ------------------------ LSL utils ------------------------
def resolve_signal_inlet(name: str, stype: str = "", timeout: float = 3.0) -> StreamInlet:
    log(f"Procurando stream de sinal LSL (name='{name}' ou type='{stype}') ...")
    while True:
        streams = resolve_byprop('name', name, timeout=timeout) if name else []
        if not streams and stype:
            streams = resolve_byprop('type', stype, timeout=timeout)
        if streams:
            si = streams[0]
            log(f"Conectado sinal: name={si.name()}, type={si.type()}, chn={si.channel_count()}, fs={si.nominal_srate():.2f}")
            return StreamInlet(si, recover=True)
        log("  sinal não encontrado; tentando novamente ...")
        time.sleep(1.0)

def make_outlet_pca(name="GrazMI_PCA", stype="BCI"):
    info = StreamInfo(name, stype, 2, 0, 'float32', 'graz_pca')
    desc = info.desc().append_child("channels")
    for lab in ["pca1", "pca2"]:
        ch = desc.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("unit", "a.u.")
        ch.append_child_value("type", "BCI")
    return StreamOutlet(info)

def make_outlet_score(name="GrazMI_Decision", stype="BCI"):
    info = StreamInfo(name, stype, 1, 0, 'float32', 'graz_decision')
    return StreamOutlet(info)

def make_outlet_score_raw(name="GrazMI_DecisionRaw", stype="BCI"):
    info = StreamInfo(name, stype, 1, 0, 'float32', 'graz_decision_raw')
    return StreamOutlet(info)

def make_outlet_state(name="GrazMI_OutputState", stype="Markers"):
    info = StreamInfo(name, stype, 1, 0, 'int32', 'graz_state')
    return StreamOutlet(info)

def make_outlet_soft3(name="GrazMI_Output3", stype="BCI"):
    # 3 saídas: left, both(=0), right
    info = StreamInfo(name, stype, 3, 0, 'float32', 'graz_soft3')
    desc = info.desc().append_child("channels")
    for lab in ["left", "both", "right"]:
        ch = desc.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("unit", "a.u.")
        ch.append_child_value("type", "BCI")
    return StreamOutlet(info)

# ------------------------ Modelo ------------------------
def load_artifacts(prefix_or_dir: str):
    if os.path.isdir(prefix_or_dir):
        cands   = glob.glob(os.path.join(prefix_or_dir, "*_classifier.pkl"))
        if not cands:
            raise FileNotFoundError("Não encontrei *_classifier.pkl na pasta do modelo.")
        clf_p   = max(cands, key=os.path.getmtime)
        base    = re.sub(r"_classifier\.pkl$", "", clf_p)
        cmean_p = base + "_best_c_mean.pkl"
        pca_p   = base + "_dim_red.pkl"
    else:
        base = prefix_or_dir
        cmean_p = base + "_best_c_mean.pkl"
        pca_p   = base + "_dim_red.pkl"
        clf_p   = base + "_classifier.pkl"

    for p in [cmean_p, pca_p, clf_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Artefato não encontrado: {p}")

    with open(cmean_p, "rb") as f: cmean = pickle.load(f)
    with open(pca_p,   "rb") as f: pca   = pickle.load(f)
    with open(clf_p,   "rb") as f: clf   = pickle.load(f)

    log(f"Artefatos: \n  {cmean_p}\n  {pca_p}\n  {clf_p}")
    return cmean, pca, clf, base

# ------------------------ Filtro/feature ------------------------
def design_bandpass(fs: float, order: int, band):
    return signal.butter(order, band, btype="bandpass", fs=fs, output="sos")

def bp_filtfilt_window(X_win, sos):
    # X_win: (T, C)
    return signal.sosfiltfilt(sos, X_win, axis=0, padlen=0)

def window_to_feature(X_win_CxT, cmean, pca):
    # X_win_CxT: (C, T) → cov → TS → PCA
    cov         = Covariances("oas").transform(X_win_CxT[None, ...])
    ts          = tangent_space(cov, cmean)        # (1, D)
    Xp          = pca.transform(ts)                # (1, pca_dim)
    return Xp[0]

# ------------------------ Normalização/Histerese ------------------------
class DecisionNormalizer:
    """
    Mantém estatísticas robustas (mediana/MAD) de 'raw' e fornece normalização.
    Agora com 2 métodos:
      - normalize(raw): usa stats correntes (NÃO atualiza)
      - update(raw): atualiza o buffer (só chamamos quando estado ∈ {-1, +1})
    """
    def __init__(self, history_len=300):
        self.buf = deque(maxlen=history_len)

    def _stats(self):
        if len(self.buf) == 0:
            return 0.0, 1.0
        arr = np.asarray(self.buf, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = max(1e-6, 1.4826 * mad)
        return med, scale

    def normalize(self, raw):
        med, scale = self._stats()
        return float(np.tanh((float(raw) - med) / (3.0 * scale)))

    def update(self, raw):
        self.buf.append(float(raw))

def discretize(norm_score: float, th: float = 0.25, hysteresis: float = 0.05, prev_state: int = 0) -> int:
    """
    Retorna -1 (esquerda), 0 (ambos), +1 (direita) com histerese.
    """
    thL = -(th + (hysteresis if prev_state == -1 else 0.0))
    thR = +(th + (hysteresis if prev_state == +1 else 0.0))
    if norm_score <= thL: return -1
    if norm_score >= thR: return +1
    return 0

def soft3_left0right(score_norm: float):
    """
    Saída com três valores: [left, both, right],
    onde both é SEMPRE 0.0 e left/right ∈ [0,1] conforme o sinal do score.
    (sem normalização para somar 1 — preserva amplitude)
    """
    left  = max(0.0, -score_norm)
    right = max(0.0,  score_norm)
    both  = 0.0
    return left, both, right

# ------------------------ CSV logger ------------------------
def make_stamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def open_csv_logger(out_dir: str, stamp: str):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"graz_decoder_{stamp}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["iso_time","lsl_time_s","pca1","pca2","score_raw","score_norm","state","left","both","right"])
    log(f"Log inferência: {path}")
    return f, w, path

# ------------------------ Main loop ------------------------
def run(signal_name: str,
        model_prefix_or_dir: str,
        out_dir: str,
        epoch_s: float,
        step_s: float,
        band,
        order: int,
        th: float,
        hyst: float):

    # Conecta sinal
    inlet = resolve_signal_inlet(name=signal_name)
    fs = float(inlet.info().nominal_srate())
    if fs <= 0:
        raise RuntimeError("Fs nominal inválida no stream LSL.")
    C = inlet.info().channel_count()
    log(f"Fs={fs:.2f} Hz, Canais={C}")

    # Carrega artefatos
    cmean, pca, clf, _ = load_artifacts(model_prefix_or_dir)
    pca_dim = getattr(pca, "n_components_", None) or getattr(pca, "n_components", 2)
    pca_dim = int(pca_dim) if pca_dim else 2

    # Filtro
    sos = design_bandpass(fs, order, band)

    # Janela/hop
    win_n = int(round(epoch_s * fs))
    hop_n = int(round(step_s * fs))
    if hop_n <= 0 or win_n <= 2*order:
        raise ValueError("Parâmetros de janela/step inválidos.")
    log(f"Janela={win_n} amostras ({epoch_s:.2f}s), Step={hop_n} amostras ({step_s:.2f}s)")

    # Buffers
    buf_X = deque(maxlen=win_n + 8*hop_n)
    buf_t = deque(maxlen=win_n + 8*hop_n)
    next_compute_at = win_n
    n_samples = 0

    # Normalizador e histerese
    normer = DecisionNormalizer(history_len=max(30, int(10.0 / step_s)))
    prev_state = 0

    # Saídas LSL
    out_pca   = make_outlet_pca()
    out_raw   = make_outlet_score_raw()
    out_score = make_outlet_score()
    out_state = make_outlet_state()
    out_soft3 = make_outlet_soft3()

    # Logger
    stamp = make_stamp()
    fcsv, wcsv, csv_path = open_csv_logger(out_dir, stamp)
    unix_offset = time.time() - local_clock()

    log("Rodando (Ctrl+C para sair) ...")
    try:
        while True:
            samples, ts = inlet.pull_chunk(timeout=0.2, max_samples=8*hop_n)
            if not ts:
                continue
            for x, t_lsl in zip(samples, ts):
                buf_X.append(np.asarray(x, dtype=float))
                buf_t.append(float(t_lsl))
                n_samples += 1

                if n_samples >= next_compute_at and len(buf_X) >= win_n:
                    # janela (T,C) e tempos
                    X_win = np.vstack(list(buf_X)[-win_n:])
                    t_win = np.array(list(buf_t)[-win_n:], dtype=float)

                    # filtra
                    try:
                        Xf = bp_filtfilt_window(X_win, sos)
                    except Exception:
                        Xf = X_win

                    # features → TS → PCA
                    feat_pca = window_to_feature(Xf.T, cmean, pca)
                    pca1 = float(feat_pca[0]) if pca_dim >= 1 else 0.0
                    pca2 = float(feat_pca[1]) if pca_dim >= 2 else 0.0

                    # classificador
                    raw = float(clf.decision_function(feat_pca.reshape(1, -1))[0])

                    # normaliza usando stats correntes (sem atualizar ainda)
                    score = normer.normalize(raw)           # [-1,1] (ambos não interfere)
                    state = discretize(score, th=th, hysteresis=hyst, prev_state=prev_state)

                    # se for LEFT/RIGHT, atualiza stats; se BOTH (0), não atualiza
                    if state != 0:
                        normer.update(raw)
                    prev_state = state

                    # vetor de saída (both = 0 sempre)
                    left, both, right = soft3_left0right(score)

                    # envia (timestamp = última amostra da janela)
                    out_pca.push_sample([pca1, pca2], timestamp=t_win[-1])
                    out_raw.push_sample([raw],         timestamp=t_win[-1])
                    out_score.push_sample([score],     timestamp=t_win[-1])
                    out_state.push_sample([int(state)],timestamp=t_win[-1])
                    out_soft3.push_sample([left, both, right], timestamp=t_win[-1])

                    # log CSV
                    iso = dt.datetime.fromtimestamp(t_win[-1] + unix_offset).isoformat(timespec="microseconds")
                    wcsv.writerow([iso, f"{t_win[-1]:.9f}", f"{pca1:.6f}", f"{pca2:.6f}",
                                   f"{raw:.6f}", f"{score:.6f}", int(state),
                                   f"{left:.6f}", f"{both:.6f}", f"{right:.6f}"])

                    # próxima inferência
                    next_compute_at += hop_n
    except KeyboardInterrupt:
        log("Interrompido pelo usuário.")
    finally:
        try: fcsv.close()
        except: pass
        log(f"Arquivo salvo: {csv_path}")

def main():
    ap = argparse.ArgumentParser(description="Decodificador Graz em tempo-real (LSL) com PCA + score + estado 3 vias (both zerado e fora da normalização).")
    ap.add_argument("--signal_name", default=DEFAULT_SIGNAL_NAME, help="Nome do stream LSL do sinal (EEG).")
    ap.add_argument("--model_prefix", default=None,
                    help="Prefixo dos arquivos do modelo (ex: C:\\..\\graz_model_PREFIX). "
                         "Se None, procura o conjunto mais recente em --out_dir.")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Pasta para salvar o CSV de inferência.")
    ap.add_argument("--epoch", type=float, default=2.0, help="Janela (s).")
    ap.add_argument("--step",  type=float, default=0.05, help="Passo entre inferências (s).")
    ap.add_argument("--band",  type=float, nargs=2, default=(8.0, 30.0), help="Bandpass (low high) Hz.")
    ap.add_argument("--order", type=int, default=4, help="Ordem do Butter bandpass.")
    ap.add_argument("--thr",   type=float, default=0.25, help="Limiar da histerese (|score_norm| >= thr ativa esquerda/direita).")
    ap.add_argument("--hyst",  type=float, default=0.05, help="Margem extra para histerese do último estado.")
    args = ap.parse_args()

    # modelo
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
        th=args.thr,
        hyst=args.hyst)

if __name__ == "__main__":
    main()