# -*- coding: utf-8 -*-
"""
check_data.py

QC do mesmo dado usado para o treino:
- RAW/stack com marcadores;
- janelas de treino dentro do ATTEMPT;
- usa os mesmos canais e o mesmo filtro causal/banda do modelo para a figura de epochs.
"""

import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from collections import Counter
from typing import List, Optional, Dict
from scipy import signal

from config_models import AppConfig

COLOR = {
    "BASELINE"     : "#7f7f7f",
    "ATTENTION"    : "#ff7f0e",
    "LEFT_MI_STIM" : "#4169e1",
    "RIGHT_MI_STIM": "#dc143c",
    "ATTEMPT"      : "#2e8b57",
    "REST"         : "#800080",
}


def _split_marker_name(fname: str):
    import re
    m = re.match(r"^(?P<prefix>.+)_markers_(?P<run_id>\d{8}_\d{6})\.csv$", fname)
    if not m:
        return None
    return m.group("prefix"), m.group("run_id")


def find_marker_signal_pairs(folder: str):
    markers      = glob.glob(os.path.join(folder, "*markers_*.csv"))
    exact_pairs  = []
    legacy_pairs = []

    for m in markers:
        base_m = os.path.basename(m)
        parsed = _split_marker_name(base_m)

        if parsed is not None:
            prefix, run_id = parsed
            s = os.path.join(os.path.dirname(m), f"{prefix}_signal_{run_id}.csv")
            if os.path.exists(s):
                exact_pairs.append((m, s))
            continue

        if "_markers_" not in base_m:
            continue
        prefix, _ = base_m.split("_markers_", 1)
        sig_files = glob.glob(os.path.join(folder, prefix + "_signal_*.csv"))
        if sig_files:
            legacy_pairs.append((m, max(sig_files, key=os.path.getmtime)))

    pairs = exact_pairs if exact_pairs else legacy_pairs
    return sorted(pairs, key=lambda p: os.path.getmtime(p[1]), reverse=True)

def choose_pair(folder: str):
    pairs = find_marker_signal_pairs(folder)
    if not pairs:
        raise FileNotFoundError(f"Nenhum par *_markers_*.csv / *_signal_*.csv em {folder}.")
    for i, (m, s) in enumerate(pairs, start=1):
        mt = dt.datetime.fromtimestamp(os.path.getmtime(m)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{i}] {os.path.basename(m)} | {os.path.basename(s)} ({mt})")
    while True:
        ans = input("Selecione o número do par [1 = mais recente]: ").strip()
        idx = 1 if ans == "" else int(ans) if ans.isdigit() else -1
        if 1 <= idx <= len(pairs):
            return pairs[idx-1]
        print("Número inválido.")


def resolve_pair(folder: str, mark_explicit: Optional[str] = None, sig_explicit: Optional[str] = None):
    if mark_explicit and sig_explicit:
        m = mark_explicit if os.path.isabs(mark_explicit) else os.path.join(folder, mark_explicit)
        s = sig_explicit if os.path.isabs(sig_explicit) else os.path.join(folder, sig_explicit)
        if not os.path.exists(m):
            raise FileNotFoundError(f"Arquivo de marcadores não existe: {m}")
        if not os.path.exists(s):
            raise FileNotFoundError(f"Arquivo de sinal não existe: {s}")
        return m, s
    return choose_pair(folder)


def read_markers_csv(path: str, code_map: Dict[int, str]):
    t, labels, codes = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = None
            if row.get("lsl_time_s"):
                try: ts = float(row["lsl_time_s"])
                except Exception: ts = None
            if ts is None and row.get("time_s"):
                try: ts = float(row["time_s"])
                except Exception: ts = None
            if ts is None and row.get("iso_time"):
                try: ts = dt.datetime.fromisoformat(row["iso_time"]).timestamp()
                except Exception: ts = None
            if ts is None:
                continue
            lab = (row.get("label") or row.get("event") or "").strip()
            c = None
            raw_code = (row.get("code") or "").strip()
            if raw_code:
                try: c = int(float(raw_code))
                except Exception: c = None
            if not lab and c is not None and c in code_map:
                lab = code_map[c]
            t.append(ts); labels.append(lab); codes.append(c)
    return np.asarray(t, float), labels, codes


def read_signal_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if "lsl_time_s" not in header:
            raise RuntimeError("CSV do sinal precisa ter coluna 'lsl_time_s'.")
        idx_ts = header.index("lsl_time_s")
        ch_start = header.index("local_recv_s") + 1 if "local_recv_s" in header else idx_ts + 1
        ch_names = header[ch_start:]
        t, X = [], []
        for row in r:
            try:
                t.append(float(row[idx_ts]))
                X.append([float(v) for v in row[ch_start:]])
            except Exception:
                continue
    return np.asarray(t, float), np.asarray(X, float), ch_names


def estimate_fs(t):
    d = np.diff(t)
    d = d[(d > 0) & np.isfinite(d)]
    return 1 / np.median(d) if d.size else 0.0


def bandpass_causal(data, fs, order, band):
    sos = signal.butter(int(order), [float(band[0]), float(band[1])], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, data, axis=0)


def nearest_index(t, x):
    i = np.searchsorted(t, x)
    if i <= 0: return 0
    if i >= len(t): return len(t)-1
    return i-1 if abs(t[i-1]-x) <= abs(t[i]-x) else i


def attempts_by_class(t_mark, labels, codes, code_map):
    out = {"LEFT_MI_STIM": [], "RIGHT_MI_STIM": []}
    last = None
    for ts, lab, c in zip(t_mark, labels, codes):
        lab_eff = lab if lab else code_map.get(c, "")
        if lab_eff in out:
            last = lab_eff
        elif lab_eff == "ATTEMPT" and last in out:
            out[last].append(ts)
    return out


def select_channel_indices(selection: List, ch_names: List[str], select_by: str = "index", index_base: int = 1):
    if not selection:
        return list(range(len(ch_names)))
    if str(select_by).lower() == "name":
        name_to_idx = {str(n).strip().lower(): i for i, n in enumerate(ch_names)}
        missing = [ch for ch in selection if str(ch).strip().lower() not in name_to_idx]
        if missing:
            raise ValueError(f"Canais não encontrados: {missing}")
        return [name_to_idx[str(ch).strip().lower()] for ch in selection]
    idx = [int(v) - (1 if index_base == 1 else 0) for v in selection]
    if any(i < 0 or i >= len(ch_names) for i in idx):
        raise ValueError(f"Índice de canal fora do range: {selection}")
    return idx


def stack_plot(ax, t, X, ch_names):
    stds = X.std(axis=0, ddof=1)
    med = float(np.median(stds[stds > 0])) if np.any(stds > 0) else 1.0
    offset = 4.0 * med
    for ci in range(X.shape[1]):
        ax.plot(t, X[:, ci] + ci * offset, color="k", lw=0.8)
    ax.set_yticks([i * offset for i in range(len(ch_names))], ch_names)
    ax.set_ylabel("Canais")
    ax.grid(True, axis="x", ls="--", alpha=0.3)


def plot_window_examples(ax, t_sig, Xf, events, fs, window_s, trial_duration_s, ch_names, color, title):
    win_n = int(round(window_s * fs))
    # mostra uma janela por tentativa, centrada aproximadamente no começo útil do ATTEMPT
    windows = []
    for ev in events:
        i0 = nearest_index(t_sig, ev)
        i1 = i0 + win_n
        if i1 <= len(t_sig):
            windows.append((t_sig[i0:i1] - ev, Xf[i0:i1]))
    n_show = min(len(windows), 10)
    if n_show == 0:
        ax.axis("off")
        ax.set_title(title + " — sem janelas")
        return
    stds = np.concatenate([w[1] for w in windows[:n_show]], axis=0).std(axis=0, ddof=1)
    med = float(np.median(stds[stds > 0])) if np.any(stds > 0) else 1.0
    offset = 4.0 * med
    for k, (tt, xx) in enumerate(windows[:n_show]):
        alpha = 0.35 if k else 0.9
        for ci in range(xx.shape[1]):
            ax.plot(tt, xx[:, ci] + ci * offset, color=color, lw=0.8, alpha=alpha)
    ax.axvspan(0, trial_duration_s, color=color, alpha=0.06)
    ax.axvline(0, color="#555", ls="--", lw=1.0)
    ax.set_yticks([i * offset for i in range(len(ch_names))], ch_names)
    ax.set_xlim(-0.05, max(window_s, 1.0))
    ax.set_title(f"{title} — janelas de {window_s:.1f}s")
    ax.grid(True, axis="x", ls="--", alpha=0.3)


def run_check_data(cfg: AppConfig, mode: str = "train", markers_file: Optional[str] = None, signal_file: Optional[str] = None, save_png: Optional[bool] = None):
    cdcfg = cfg.check_data
    mcfg = cfg.model
    data_dir = os.path.join(cfg.experiment.log_root, cfg.experiment.subject_id, f"S{cfg.experiment.session_id}", cfg.experiment.session_type, mode)
    print(f"\n[check] Procurando dados em: {data_dir}")
    mark_path, sig_path = resolve_pair(data_dir, markers_file, signal_file)
    print("[check] Marcadores:", mark_path)
    print("[check] Sinal:     ", sig_path)

    if save_png is None:
        save_png = bool(cdcfg.save_png)

    code_map = cfg.codes.code_map
    t_mark, labels, codes = read_markers_csv(mark_path, code_map)
    t_sig, X_full, ch_all = read_signal_csv(sig_path)
    fs = estimate_fs(t_sig)
    if fs <= 0:
        fs = float(mcfg.fs_hz)
    print(f"[check] Fs estimado ~ {fs:.2f} Hz | canais={len(ch_all)}")
    print("[check] Labels:", dict(Counter(labels)))

    select_channels = mcfg.select_channels or []
    sel_idx = select_channel_indices(select_channels, ch_all, select_by=getattr(mcfg, "select_by", "index"), index_base=int(getattr(mcfg, "index_base", 1)))
    ch_names = [ch_all[i] for i in sel_idx]
    X = X_full[:, sel_idx]

    # Figura 1: stack RAW sem alterar o sinal, só para identificar artefatos e marcadores.
    t0 = min(t_sig[0], t_mark[0]) if len(t_mark) else t_sig[0]
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    stack_plot(ax1, t_sig - t0, X, ch_names)
    for tm, lab in zip(t_mark, labels):
        ax1.axvline(tm - t0, color=COLOR.get(lab, "k"), lw=1.2, alpha=0.9)
    ax1.set_xlabel("Tempo (s)")
    ax1.set_title(f"RAW + marcadores — {os.path.basename(sig_path)}")

    # Figura 2: exemplos de janelas após o MESMO filtro do modelo.
    band = tuple(mcfg.bp_band)
    order = int(mcfg.bp_order)
    window_s = float(mcfg.epoch_s)
    trial_duration_s = float(getattr(cdcfg, "tmax", 3.75))
    Xf = bandpass_causal(X, float(mcfg.fs_hz), order, band)
    print(f"[check] Filtro igual ao modelo: causal bandpass {band}, ordem {order}")

    ev_by_cls = attempts_by_class(t_mark, labels, codes, code_map)
    fig2, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    plot_window_examples(axes[0], t_sig, Xf, ev_by_cls.get("LEFT_MI_STIM", []), float(mcfg.fs_hz), window_s, trial_duration_s, ch_names, COLOR["LEFT_MI_STIM"], "LEFT")
    plot_window_examples(axes[1], t_sig, Xf, ev_by_cls.get("RIGHT_MI_STIM", []), float(mcfg.fs_hz), window_s, trial_duration_s, ch_names, COLOR["RIGHT_MI_STIM"], "RIGHT")
    axes[-1].set_xlabel("Tempo relativo ao ATTEMPT (s)")
    fig2.suptitle("Janelas usadas pelo modelo — mesmo filtro do decoder", y=0.98)
    plt.tight_layout()

    if save_png:
        base = os.path.splitext(sig_path)[0]
        fig1.savefig(base + "_stack_raw_markers.png", dpi=150)
        fig2.savefig(base + "_model_windows.png", dpi=150)
        print(f"[check] Figuras salvas em:\n  {base + '_stack_raw_markers.png'}\n  {base + '_model_windows.png'}")
    else:
        plt.show()
    plt.close(fig1); plt.close(fig2)


if __name__ == "__main__":
    from config_models import load_config
    cfg = load_config("config.yaml")
    run_check_data(cfg)
