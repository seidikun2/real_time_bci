# -*- coding: utf-8 -*-
"""
plot_graz_stack_and_epochs_sel_hp.py

- Lê *markers* e *signal* (pega os mais recentes se não forem informados).
- Seleciona canais por índice ou nome.
- Aplica filtro passa-altas Butterworth (0.5 Hz) com filtfilt.
- Plota stack RAW + marcações e epochs alinhados no ATTEMPT.
"""

import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.signal import butter, filtfilt

# ======== CONFIG RÁPIDA ========
DATA_DIR      = r"C:\Users\User\Desktop\Dados"
MARKERS_FILE  = None   # None => pega o mais recente (*markers_*.csv)
SIGNAL_FILE   = None   # None => pega o mais recente (*signal_*.csv)

# Seleção de canais
SELECT_BY     = "index"   # "index" ou "name"
INDEX_BASE    = 0         # 0 => índices Python (0,1,2,...); 1 => índices 1-based
SELECT_CHANNELS = [1,2,4,5,7,9,10]  # default pedido

# Config do filtro passa-altas
HP_CUTOFF_HZ  = 0.5
HP_ORDER      = 4

# Classes mostradas
CLASSES       = ("LEFT_MI_STIM", "RIGHT_MI_STIM")

TMIN, TMAX    = -0.5, 3.75
BASELINE_S    = 0.0
SAVE_PNG      = True

COLOR         = {"BASELINE": "#7f7f7f", "ATTENTION": "#ff7f0e", "LEFT_MI_STIM": "#4169e1",
                 "RIGHT_MI_STIM":"#dc143c", "ATTEMPT":"#2e8b57", "REST": "#800080",}
# ===============================

def find_latest(folder: str, pattern: str) -> str:
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo: {os.path.join(folder, pattern)}")
    return max(files, key=os.path.getmtime)

def read_markers_csv(path: str):
    t, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = None
            if "lsl_time_s" in row and row["lsl_time_s"]:
                try: ts = float(row["lsl_time_s"])
                except: ts = None
            if ts is None and row.get("iso_time"):
                try: ts = dt.datetime.fromisoformat(row["iso_time"]).timestamp()
                except: ts = None
            if ts is None: continue
            lab = (row.get("label") or "").strip()
            t.append(ts); labels.append(lab)
    return np.asarray(t, float), labels

def read_signal_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f); header = next(r)
        if "lsl_time_s" not in header:
            raise RuntimeError("CSV do sinal precisa ter 'lsl_time_s'.")
        idx_ts = header.index("lsl_time_s")
        ch_start = header.index("local_recv_s")+1 if "local_recv_s" in header else idx_ts+1
        ch_names = header[ch_start:]
        t, X = [], []
        for row in r:
            try:
                t.append(float(row[idx_ts]))
                X.append([float(v) for v in row[ch_start:]])
            except: continue
    return np.asarray(t, float), np.asarray(X, float), ch_names

def estimate_fs(t):
    d = np.diff(t); d = d[(d > 0) & np.isfinite(d)]
    return 1/np.median(d) if d.size else 0.0

def butter_highpass(cut_hz, fs, order=4):
    nyq = fs/2.0
    wn = cut_hz/nyq
    b,a = butter(order, wn, btype='high')
    return b,a

def nearest_index(t, x):
    i = np.searchsorted(t, x)
    if i <= 0: return 0
    if i >= len(t): return len(t)-1
    return i-1 if abs(t[i-1]-x) <= abs(t[i]-x) else i

def epoch_list(t_sig, X, events, fs, tmin, tmax):
    n_times = int(round((tmax - tmin) * fs)) + 1
    time_rel = np.arange(n_times)/fs + tmin
    s_ofs = int(round(tmin * fs)); e_ofs = int(round(tmax * fs))
    trials = []
    for ev in events:
        i0 = nearest_index(t_sig, ev); s, e = i0+s_ofs, i0+e_ofs
        if s < 0 or e >= len(t_sig): continue
        seg = X[s:e+1, :]
        if seg.shape[0] == n_times: trials.append(seg)
    return trials, time_rel

def baseline_correct(trials, fs, tmin, baseline_s):
    if baseline_s <= 0: return trials
    i_end = int(round((0 - tmin) * fs))
    i_sta = max(0, i_end - int(round(baseline_s * fs)))
    out = []
    for seg in trials:
        if 0 <= i_sta < i_end <= seg.shape[0]:
            bl = seg[i_sta:i_end,:].mean(axis=0, keepdims=True)
            out.append(seg - bl)
        else: out.append(seg)
    return out

def stack_plot(ax, t, X, ch_names):
    stds = X.std(axis=0, ddof=1)
    med = float(np.median(stds[stds>0])) if np.any(stds>0) else 1.0
    offset = 4.0*med
    for ci in range(X.shape[1]):
        ax.plot(t, X[:,ci]+ci*offset, color="k", lw=0.8)
    ax.set_yticks([i*offset for i in range(len(ch_names))], ch_names)
    ax.set_ylabel("Canais"); ax.grid(True, axis="x", ls="--", alpha=0.3)

def compute_offset_for_trials(trials):
    if not trials: return 1.0
    Xcat = np.concatenate(trials, axis=0)
    stds = Xcat.std(axis=0, ddof=1)
    med = float(np.median(stds[stds>0])) if np.any(stds>0) else 1.0
    return 4.0*med

def plot_trial_stack(ax, time_rel, seg, ch_names, offset, color):
    C = seg.shape[1]
    for ci in range(C):
        ax.plot(time_rel, seg[:,ci]+ci*offset, color=color, lw=0.8)
    if ax.get_subplotspec().is_first_col():
        ax.set_yticks([i*offset for i in range(C)], ch_names)
    else: ax.set_yticks([])
    ax.axvline(0, color="#555", ls="--", lw=1.0, alpha=0.7)
    ax.grid(True, axis="x", ls="--", alpha=0.3)

def attempts_by_class(t_mark, labels):
    out = {"LEFT_MI_STIM": [], "RIGHT_MI_STIM": []}
    last_cue = None
    for ts,lab in zip(t_mark, labels):
        if lab in out: last_cue = lab
        elif lab=="ATTEMPT" and last_cue in out: out[last_cue].append(ts)
    return out

def select_channel_indices(select_by, selection, ch_names, index_base=0):
    if not selection: raise ValueError("SELECT_CHANNELS vazio.")
    if select_by.lower()=="name":
        m = {nm:i for i,nm in enumerate(ch_names)}
        return [m[nm] for nm in selection]
    idx = [int(v)-(1 if index_base==1 else 0) for v in selection]
    return idx

def main():
    mark_path = MARKERS_FILE or find_latest(DATA_DIR, "*markers_*.csv")
    sig_path  = SIGNAL_FILE  or find_latest(DATA_DIR, "*signal_*.csv")
    print("Marcadores:", mark_path)
    t_mark, labels = read_markers_csv(mark_path)
    print("Sinal:", sig_path)
    t_sig, X_full, ch_all = read_signal_csv(sig_path)
    fs = estimate_fs(t_sig)
    print(f"Fs ~ {fs:.2f} Hz | canais: {len(ch_all)}")
    sel_idx  = select_channel_indices(SELECT_BY, SELECT_CHANNELS, ch_all, INDEX_BASE)
    ch_names = [ch_all[i] for i in sel_idx]
    X        = X_full[:,sel_idx]

    # aplica filtro passa-altas
    b,a = butter_highpass(HP_CUTOFF_HZ, fs, HP_ORDER)
    X   = filtfilt(b,a,X,axis=0)
    print(f"Filtro passa-altas aplicado: {HP_CUTOFF_HZ} Hz, ordem {HP_ORDER}")

    # FIGURA 1: stack
    t0 = min(t_sig[0], t_mark[0]) if (t_sig.size and t_mark.size) else t_sig[0]
    t_rel = t_sig - t0
    fig1, ax1 = plt.subplots(figsize=(12,6))
    stack_plot(ax1, t_rel, X, ch_names)
    for tm,lab in zip(t_mark, labels):
        ax1.axvline(tm-t0, color=COLOR.get(lab,"k"), lw=1.2, alpha=0.9)
    ax1.set_xlabel("Tempo (s)")
    ax1.set_title(f"Stack RAW (filtro HP {HP_CUTOFF_HZ} Hz) — {os.path.basename(sig_path)}")

    # FIGURA 2: epochs
    ev_by_cls = attempts_by_class(t_mark, labels)
    trials_by_cls = {}
    for cls in CLASSES:
        ev = np.asarray(ev_by_cls.get(cls,[]), float)
        trials,time_rel = epoch_list(t_sig,X,ev,fs,TMIN,TMAX)
        trials = baseline_correct(trials,fs,TMIN,BASELINE_S)
        trials_by_cls[cls] = trials
        print(f"{cls}: {len(trials)} trials")
    ncols = max(len(trials_by_cls[c]) for c in CLASSES) or 1
    fig2,axes = plt.subplots(2,ncols,figsize=(3.2*ncols,6.4),sharex=True)
    if ncols==1: axes = np.array(axes).reshape(2,1)
    for row,cls in enumerate(CLASSES):
        trials = trials_by_cls[cls]; color=COLOR.get(cls,"k")
        offset=compute_offset_for_trials(trials)
        for col in range(ncols):
            ax=axes[row,col]
            if col<len(trials):
                plot_trial_stack(ax,time_rel,trials[col],ch_names,offset,color)
                ax.set_title(f"{cls} — trial {col+1}",fontsize=9,color=color)
            else: ax.axis("off")
    axes[1,0].set_xlabel("Tempo relativo ao ATTEMPT (s)")
    fig2.suptitle(f"Epochs (HP {HP_CUTOFF_HZ} Hz) — canais selecionados",y=0.98)

    if SAVE_PNG:
        base = os.path.splitext(sig_path)[0]
        fig1.savefig(base+"_stack_hp.png",dpi=150)
        fig2.savefig(base+"_epochs_hp.png",dpi=150)
    else: plt.show()

if __name__=="__main__":
    main()
