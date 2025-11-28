import os
import re
import csv
import glob
import time
import pickle
import datetime as dt
from collections import deque

import numpy as np
from scipy import signal
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

# ===================== CONFIG (editar no Spyder) =====================
OUT_DIR             = r"C:\Users\User\Desktop\Dados"         # pasta para salvar o CSV
SIGNAL_NAME         = "Cognionics Wireless EEG"             # nome do stream LSL de EEG

# Identificadores do modelo (DEVE bater com os arquivos de treino)
SUBJECT_ID          = "SY100"                               # ex.: "SY100"
EXPERIMENT_ID       = "TEST"                                # ex.: "TEST"
SESSION_ID          = 1                                     # ex.: 1 (vira S1)
PROTOCOL_TYPE       = "IM_treino"                           # ex.: "IM_treino"

MODEL_PREFIX        = None                                  # se None, usa as 4 variáveis acima
EPOCH_S             = 2.5                                   # janela (s)
STEP_S              = 0.05                                  # passo entre inferências (s)
BAND_HZ             = (5.0, 50.0)                           # bandpass (Hz)
FILTER_ORDER        = 4                                     # ordem do filtro
OUTLET_NAME         = "Signal"                              # nome base do stream LSL de saída
LSL_RATE_HZ         = 50.0                                  # 0 => enviar a cada inferência
LEFT_LABEL          = None                                  # rótulo explícito da classe LEFT (se usar proba)
RIGHT_LABEL         = None                                  # rótulo explícito da classe RIGHT (se usar proba)

# ===== Seleção de canais =====
# SELECT_BY: "index"  -> usa números (com base escolhida por INDEX_BASE)
#            "name"   -> usa nomes exatamente como no stream LSL (desc/channels/channel/label)
SELECT_BY           = "index"          # "index" ou "name"
INDEX_BASE          = 0                # 0 => Python (0,1,2,...); 1 => 1-based
SELECT_CHANNELS     = []               # [] => usa todos
# ====================================================================

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

# --------- nomes e seleção de canais vindos do LSL ----------
def get_lsl_channel_names(info: StreamInfo):
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
            names = names[:C] if len(names) > C else names + [f"ch{i+1}" for i in range(len(names), C)]
        return names
    except Exception:
        return [f"ch{i+1}" for i in range(C)]

def select_channel_indices(select_by: str, selection, ch_names, index_base: int = 0):
    """
    - select_by='name': 'selection' é lista de nomes exatamente como em ch_names.
    - select_by='index': 'selection' é lista de inteiros; se index_base=1, converte p/ 0-based.
    - selection vazio/None -> todos os canais.
    """
    C = len(ch_names)
    if not selection:
        return list(range(C))
    if str(select_by).lower() == "name":
        name_to_idx = {nm: i for i, nm in enumerate(ch_names)}
        missing     = [nm for nm in selection if nm not in name_to_idx]
        if missing:
            raise ValueError(f"Canais não encontrados (por nome): {missing}\nDisponíveis: {ch_names}")
        return [name_to_idx[nm] for nm in selection]
    idx = [int(v) - (1 if index_base == 1 else 0) for v in selection]
    if any((i < 0 or i >= C) for i in idx):
        raise ValueError(f"Algum índice está fora do range [0, {C-1}] (após base). Seleção={idx}")
    return idx

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
    """
    Cria arquivo CSV de log com identificadores no nome, se disponíveis:
    ex.: SY100_TEST_S1_IM_treino_decoder_YYYYMMDD_HHMMSS.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Se todos os IDs estão definidos, usa padrão com IDs
    if all(v not in (None, "") for v in [SUBJECT_ID, EXPERIMENT_ID, SESSION_ID, PROTOCOL_TYPE]):
        core = f"{SUBJECT_ID}_{EXPERIMENT_ID}_S{SESSION_ID}_{PROTOCOL_TYPE}_decoder"
    else:
        core = "graz_decoder"

    path = os.path.join(out_dir, f"{core}_{stamp}.csv")
    f    = open(path, "w", newline="", encoding="utf-8")
    w    = csv.writer(f)
    w.writerow([
        "iso_time","lsl_time_s","recv_time_s",
        "pca1","pca2",
        "left","both","right"
    ])
    log(f"Log de inferência: {path}")
    return f, w, path

# ------------------------ Loop principal ------------------------
def run(signal_name: str, model_prefix_or_dir: str, out_dir: str, epoch_s: float,
        step_s: float, band, order: int, outlet_name: str, lsl_rate: float,
        left_label=None, right_label=None):

    inlet = resolve_signal_inlet(name=signal_name, stype="EEG")
    info  = inlet.info()
    fs    = float(info.nominal_srate())
    C     = int(info.channel_count())
    if fs <= 0:
        raise RuntimeError("Fs nominal inválida no stream LSL.")
    ch_all = get_lsl_channel_names(info)
    log(f"Fs={fs:.2f} Hz, Canais={C}")
    log(f"Canais LSL: {ch_all}")

    sel_idx = select_channel_indices(SELECT_BY, SELECT_CHANNELS, ch_all, INDEX_BASE)
    ch_sel  = [ch_all[i] for i in sel_idx]
    C_sel   = len(sel_idx)
    log(f"Seleção de canais: {C_sel} -> {ch_sel}")

    cmean, pca, clf = load_artifacts(model_prefix_or_dir)
    pca_dim   = getattr(pca, "n_components_", None) or getattr(pca, "n_components", 2)
    pca_dim   = int(pca_dim) if pca_dim else 2
    has_proba = hasattr(clf, "predict_proba")
    log(f"Saída do classificador: {'predict_proba' if has_proba else 'decision_function'}")

    sos = design_bandpass(fs, order, band)

    win_n = int(round(epoch_s * fs))
    hop_n = int(round(step_s * fs))
    if hop_n <= 0 or win_n <= max(8, 2*order):
        raise ValueError("Parâmetros de janela/step inválidos (aumente epoch/step).")
    log(f"Janela={win_n} amostras ({epoch_s:.2f}s), Step={hop_n} amostras ({step_s:.2f}s)")

    buf_X          = deque(maxlen=win_n + 8*hop_n)
    buf_t          = deque(maxlen=win_n + 8*hop_n)
    next_compute_at= win_n
    n_samples      = 0

    srate_out = max(0.0, float(lsl_rate))

    outlet             = make_outlet_unified(name=outlet_name, srate=srate_out)
    send_interval      = (1.0 / srate_out) if srate_out > 0 else None
    next_send_monotonic= time.monotonic() if send_interval else None
    latest_vec         = None

    fcsv, wcsv, csv_path = open_csv_logger(out_dir)
    unix_offset = time.time() - local_clock()

    log(f"Envio LSL: {'desacoplado a ' + str(lsl_rate) + ' Hz' if send_interval else 'a cada inferência (lsl_rate=0)'}")
    log(f"Nome do stream de saída: {outlet_name}")
    log("Rodando (Ctrl+C para sair) ...")
    try:
        while True:
            data, ts = inlet.pull_chunk(timeout=0.2, max_samples=8*hop_n)

            if send_interval and latest_vec is not None:
                now_mono = time.monotonic()
                if now_mono >= next_send_monotonic:
                    outlet.push_sample(latest_vec, timestamp=local_clock())
                    next_send_monotonic = now_mono + send_interval

            if not ts:
                continue

            for x, t_lsl in zip(data, ts):
                x = np.asarray(x, dtype=float)
                if x.size < max(sel_idx)+1:
                    continue
                x_sel = x[sel_idx]
                buf_X.append(x_sel)
                buf_t.append(float(t_lsl))
                n_samples += 1

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

                    if has_proba:
                        proba   = clf.predict_proba(feat.reshape(1, -1))[0]
                        classes = list(getattr(clf, "classes_", []))
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
                        raw     = float(clf.decision_function(feat.reshape(1, -1))[0])
                        out_val = float(np.clip(raw, -1.0, 1.0))
                        if out_val < 0:
                            left, right = out_val, 0.0
                        elif out_val > 0:
                            left, right = 0.0, out_val
                        else:
                            left, right = 0.0, 0.0
                        both = 0.0

                    latest_vec  = [-right, -right, left]
                    print(latest_vec)
                    t_out       = float(t_win[-1])
                    recv_time_s = time.time()
                    iso         = timestamp_iso_from_lsl(t_out, unix_offset)

                    wcsv.writerow([
                        iso, f"{t_out:.9f}", f"{recv_time_s:.6f}",
                        f"{p1:.6f}", f"{p2:.6f}",
                        f"{left:.6f}", f"{both:.6f}", f"{right:.6f}"
                    ])

                    if not send_interval:
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

# ===================== Execução direta no Spyder =====================
if __name__ == "__main__":
    # Resolve modelo (prefixo ou diretório).
    # Se MODEL_PREFIX for None, usa SUBJECT_ID, EXPERIMENT_ID, SESSION_ID, PROTOCOL_TYPE.
    if MODEL_PREFIX is not None:
        model_arg = MODEL_PREFIX
        log(f"Usando MODEL_PREFIX explícito: {model_arg}")
    else:
        if any(v in (None, "") for v in [SUBJECT_ID, EXPERIMENT_ID, SESSION_ID, PROTOCOL_TYPE]):
            raise ValueError(
                "MODEL_PREFIX está None e algum identificador (SUBJECT_ID, EXPERIMENT_ID, "
                "SESSION_ID, PROTOCOL_TYPE) não foi definido."
            )
        core    = f"{SUBJECT_ID}_{EXPERIMENT_ID}_S{SESSION_ID}_{PROTOCOL_TYPE}_signal"
        pattern = os.path.join(OUT_DIR, f"{core}_*_classifier.pkl")
        cands   = glob.glob(pattern)
        if not cands:
            raise FileNotFoundError(
                f"Não encontrei nenhum *_classifier.pkl com padrão:\n  {pattern}"
            )
        clf_p     = max(cands, key=os.path.getmtime)
        model_arg = re.sub(r"_classifier\.pkl$", "", clf_p)
        log(f"Classificador selecionado (via IDs): {clf_p}")

    run(signal_name=SIGNAL_NAME,
        model_prefix_or_dir=model_arg,
        out_dir=OUT_DIR,
        epoch_s=EPOCH_S,
        step_s=STEP_S,
        band=BAND_HZ,
        order=FILTER_ORDER,
        outlet_name=OUTLET_NAME,
        lsl_rate=LSL_RATE_HZ,
        left_label=LEFT_LABEL,
        right_label=RIGHT_LABEL)
