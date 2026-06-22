# -*- coding: utf-8 -*-
"""
online_inference.py

Inferência contínua com a mesma lógica de feature do treino:
- carrega cmean, PCA, classificador e meta do modelo;
- usa window_s/epoch_s do modelo para a janela contínua;
- usa step_s do modelo/config para a frequência de inferência;
- aplica o mesmo filtro causal contínuo usado no treino;
- publica [pca1, pca2, left, both, right] no LSL.
"""

import os, re, glob, csv, json, time, threading, pickle, datetime as dt
from collections import deque
from typing import List, Optional, Dict, Any

import numpy as np
from scipy import signal
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

from config_models import AppConfig


def log(msg: str) -> None:
    print(f"[decoder] {msg}")


def _make_session_dir(cfg: AppConfig, mode: str) -> str:
    return os.path.join(cfg.experiment.log_root, cfg.experiment.subject_id, f"S{cfg.experiment.session_id}", cfg.experiment.session_type, mode)


def _find_model_prefix(cfg: AppConfig, explicit_prefix: Optional[str] = None) -> str:
    if explicit_prefix is not None:
        if os.path.isdir(explicit_prefix):
            cands = glob.glob(os.path.join(explicit_prefix, "*_classifier.pkl"))
            if not cands:
                raise FileNotFoundError(f"Não encontrei *_classifier.pkl em {explicit_prefix}")
            return re.sub(r"_classifier\.pkl$", "", max(cands, key=os.path.getmtime))
        return explicit_prefix

    train_dir = _make_session_dir(cfg, mode="train")
    cands = glob.glob(os.path.join(train_dir, "*_classifier.pkl"))
    if not cands:
        raise FileNotFoundError(f"Não encontrei nenhum *_classifier.pkl em {train_dir}")
    return re.sub(r"_classifier\.pkl$", "", max(cands, key=os.path.getmtime))


def load_artifacts(prefix_or_dir: str):
    if os.path.isdir(prefix_or_dir):
        cands = glob.glob(os.path.join(prefix_or_dir, "*_classifier.pkl"))
        if not cands:
            raise FileNotFoundError("Não encontrei *_classifier.pkl na pasta do modelo.")
        base = re.sub(r"_classifier\.pkl$", "", max(cands, key=os.path.getmtime))
    else:
        base = prefix_or_dir

    cmean_p = base + "_best_c_mean.pkl"
    pca_p   = base + "_dim_red.pkl"
    clf_p   = base + "_classifier.pkl"
    for p in [cmean_p, pca_p, clf_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Artefato não encontrado: {p}")

    with open(cmean_p, "rb") as f: cmean = pickle.load(f)
    with open(pca_p, "rb") as f: pca = pickle.load(f)
    with open(clf_p, "rb") as f: clf = pickle.load(f)

    log("Artefatos carregados:")
    log(f"  cmean: {cmean_p}")
    log(f"  pca  : {pca_p}")
    log(f"  clf  : {clf_p}")
    return cmean, pca, clf


def read_model_meta(prefix_or_dir: str) -> Dict[str, Any]:
    if os.path.isdir(prefix_or_dir):
        cands = glob.glob(os.path.join(prefix_or_dir, "*_classifier.pkl"))
        if not cands:
            return {}
        base = re.sub(r"_classifier\.pkl$", "", max(cands, key=os.path.getmtime))
    else:
        base = prefix_or_dir
    meta_p = base + "_meta.json"
    try:
        with open(meta_p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def resolve_signal_inlet(name: str, stype: str = "EEG", timeout: float = 3.0) -> StreamInlet:
    label = f"name='{name}'" if name else f"type='{stype}'"
    log(f"Procurando stream LSL do sinal ({label}) ...")
    while True:
        streams = resolve_byprop("name", name, timeout=timeout) if name else []
        if not streams and stype:
            streams = resolve_byprop("type", stype, timeout=timeout)
        if streams:
            si = streams[0]
            log(f"Conectado ao sinal: name={si.name()}, type={si.type()}, ch={si.channel_count()}, fs={si.nominal_srate():.2f}")
            return StreamInlet(si, recover=True)
        log("  Sinal não encontrado; tentando novamente em 1s ...")
        time.sleep(1.0)


def make_outlet_unified(name: str, stype: str = "BCI", srate: float = 0.0) -> StreamOutlet:
    info = StreamInfo(name, stype, 5, srate, "float32")
    desc = info.desc().append_child("channels")
    for lab in ["pca1", "pca2", "left", "both", "right"]:
        ch = desc.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("unit", "a.u.")
        ch.append_child_value("type", "BCI")
    return StreamOutlet(info)


def get_lsl_channel_names(info) -> List[str]:
    C = int(info.channel_count())
    try:
        chs = info.desc().child("channels")
        names = []
        ch = chs.child("channel") if chs is not None else None
        while ch is not None and ch.name() == "channel":
            lab = (ch.child_value("label") or "").strip()
            names.append(lab if lab else None)
            ch = ch.next_sibling()
        names = [names[i] if i < len(names) and names[i] else f"ch{i+1}" for i in range(C)]
        return names
    except Exception:
        return [f"ch{i+1}" for i in range(C)]


def select_channel_indices(selection: List[int], ch_names: List[str], index_base: int = 1) -> List[int]:
    if not selection:
        return list(range(len(ch_names)))
    idx = [int(v) - (1 if index_base == 1 else 0) for v in selection]
    if any(i < 0 or i >= len(ch_names) for i in idx):
        raise ValueError(f"Índice de canal fora do range após base={index_base}: {selection}")
    return idx


def design_bandpass(fs: float, order: int, band) -> np.ndarray:
    low, high = float(band[0]), float(band[1])
    if not (0 < low < high < fs / 2):
        raise ValueError(f"Banda inválida {band} para fs={fs}")
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def window_to_feature(X_win_CxT: np.ndarray, cmean, pca) -> np.ndarray:
    cov = Covariances("oas").transform(X_win_CxT[None, ...])
    ts  = tangent_space(cov, cmean)
    return pca.transform(ts)[0]


def timestamp_iso_from_lsl(lsl_time_s: float, unix_offset: float) -> str:
    return dt.datetime.fromtimestamp(lsl_time_s + unix_offset).isoformat(timespec="microseconds")


def open_csv_logger(cfg: AppConfig, mode: str):
    out_dir = _make_session_dir(cfg, mode)
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    core = f"{cfg.experiment.subject_id}_{cfg.experiment.exp_name}_S{cfg.experiment.session_id}_{cfg.experiment.session_type}_decoder"
    path = os.path.join(out_dir, f"{core}_{stamp}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["iso_time", "lsl_time_s", "recv_time_s", "pca1", "pca2", "left", "both", "right"])
    log(f"Log de inferência: {path}")
    return f, w, path


def class_outputs(clf, feat: np.ndarray, decfg) -> tuple[float, float, float]:
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(feat.reshape(1, -1))[0]
        classes = list(getattr(clf, "classes_", []))
        left_label = getattr(decfg, "left_label", None)
        right_label = getattr(decfg, "right_label", None)
        if (left_label in classes) and (right_label in classes):
            return float(proba[classes.index(left_label)]), 0.0, float(proba[classes.index(right_label)])
        if 0 in classes and 1 in classes:
            return float(proba[classes.index(0)]), 0.0, float(proba[classes.index(1)])
        return float(proba[0]), 0.0, float(proba[-1])

    raw = float(clf.decision_function(feat.reshape(1, -1))[0])
    val = float(np.clip(raw, -1.0, 1.0))
    return (-val, 0.0, 0.0) if val < 0 else (0.0, 0.0, val)


def run_realtime_decoder(cfg: AppConfig, mode: str = "realtime", model_prefix: Optional[str] = None, stop_event: Optional[threading.Event] = None):
    if stop_event is None:
        stop_event = threading.Event()

    decfg = cfg.decoder
    mcfg  = cfg.model

    prefix = _find_model_prefix(cfg, explicit_prefix=model_prefix)
    cmean, pca, clf = load_artifacts(prefix)
    meta = read_model_meta(prefix)

    inlet = resolve_signal_inlet(name=cfg.lsl.signal_name, stype=cfg.lsl.signal_type)
    info = inlet.info()
    fs   = float(info.nominal_srate())
    if fs <= 0:
        raise RuntimeError("Fs nominal inválida no stream LSL.")
    ch_all = get_lsl_channel_names(info)
    log(f"Fs stream={fs:.2f} Hz | canais={ch_all}")

    train_ch_names = meta.get("channels_selected", []) if isinstance(meta, dict) else []
    if train_ch_names:
        name_to_idx = {str(n).strip().lower(): i for i, n in enumerate(ch_all)}
        missing = [ch for ch in train_ch_names if str(ch).strip().lower() not in name_to_idx]
        if missing:
            log(f"AVISO: canais do modelo não encontrados no stream: {missing}. Usando cfg.model.select_channels.")
            sel_idx = select_channel_indices(mcfg.select_channels or [], ch_all, index_base=1)
        else:
            sel_idx = [name_to_idx[str(ch).strip().lower()] for ch in train_ch_names]
    else:
        sel_idx = select_channel_indices(mcfg.select_channels or [], ch_all, index_base=1)
    ch_sel = [ch_all[i] for i in sel_idx]
    log(f"Canais usados: {ch_sel}")

    filt_meta = meta.get("filter", {}) if isinstance(meta, dict) else {}
    band_hz   = filt_meta.get("band_hz", getattr(mcfg, "bp_band", decfg.band_hz))
    order     = int(filt_meta.get("order", getattr(mcfg, "bp_order", decfg.filter_order)))
    window_s  = float(meta.get("window_s", meta.get("epoch_s", getattr(mcfg, "epoch_s", decfg.epoch_s))))
    step_s    = float(meta.get("step_s", getattr(decfg, "step_s", 0.05)))
    train_fs  = float(meta.get("fs_hz", fs)) if isinstance(meta, dict) else fs
    if abs(fs - train_fs) > 1e-3:
        log(f"AVISO: fs stream={fs:.2f} Hz diferente do fs do modelo={train_fs:.2f} Hz.")

    sos = design_bandpass(fs, order, band_hz)
    zi = signal.sosfilt_zi(sos)
    zf = np.repeat(zi[:, :, None], len(sel_idx), axis=2)

    win_n = int(round(window_s * fs))
    hop_n = max(1, int(round(step_s * fs)))
    if win_n < 8:
        raise ValueError("Janela muito curta.")
    log(f"Online contínuo: janela={window_s:.2f}s ({win_n} samples), step={step_s:.3f}s ({hop_n} samples), filtro causal")

    outlet = make_outlet_unified(name=decfg.outlet_name, stype="BCI", srate=max(0.0, float(decfg.lsl_rate_hz)))
    fcsv, wcsv, _ = open_csv_logger(cfg, mode)
    unix_offset = time.time() - local_clock()

    buf_X = deque(maxlen=win_n)
    buf_t = deque(maxlen=win_n)
    n_samples = 0
    next_compute = win_n

    try:
        while not stop_event.is_set():
            data, ts = inlet.pull_chunk(timeout=0.2, max_samples=max(32, 4 * hop_n))
            if not ts:
                continue

            X = np.asarray(data, dtype=float)
            if X.ndim != 2 or X.shape[1] <= max(sel_idx):
                continue
            X_sel = X[:, sel_idx]

            # Mesmo tipo de filtro do treino: causal contínuo, preservando estado.
            Xf, zf = signal.sosfilt(sos, X_sel, axis=0, zi=zf)

            for x_f, t_lsl in zip(Xf, ts):
                buf_X.append(x_f)
                buf_t.append(float(t_lsl))
                n_samples += 1

                if n_samples >= next_compute and len(buf_X) >= win_n:
                    X_win = np.vstack(buf_X)  # (T,C), já filtrado
                    feat = window_to_feature(X_win.T, cmean, pca)
                    pca_dim = int(getattr(pca, "n_components_", getattr(pca, "n_components", 2)) or 2)
                    p1 = float(feat[0]) if pca_dim >= 1 else 0.0
                    p2 = float(feat[1]) if pca_dim >= 2 else 0.0
                    left, both, right = class_outputs(clf, feat, decfg)
                    vec = [p1, p2, left, both, right]
                    outlet.push_sample(vec, timestamp=local_clock())

                    t_out = float(buf_t[-1])
                    recv_time_s = time.time()
                    iso = timestamp_iso_from_lsl(t_out, unix_offset)
                    wcsv.writerow([iso, f"{t_out:.9f}", f"{recv_time_s:.6f}", f"{p1:.6f}", f"{p2:.6f}", f"{left:.6f}", f"{both:.6f}", f"{right:.6f}"])
                    next_compute += hop_n
    except KeyboardInterrupt:
        log("Ctrl+C recebido.")
    finally:
        try:
            fcsv.close()
        except Exception:
            pass
        log("Decoder encerrado.")
