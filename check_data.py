# plot_graz_stack_and_epochs.py
import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# ======== CONFIG RÁPIDA ========
DATA_DIR      = r"C:\Users\User\Desktop\Dados"
MARKERS_FILE  = DATA_DIR + r'\graz_sim_markers_20250828_174406.csv'                 # None => pega o mais recente (*markers_*.csv)
SIGNAL_FILE   = None #DATA_DIR + r'\graz_sim_signal_20250828_161404.csv'                 # None => pega o mais recente (*signal_*.csv)

# Classes mostradas nas linhas de subplots (derivadas do cue anterior ao ATTEMPT)
CLASSES       = ("LEFT_MI_STIM", "RIGHT_MI_STIM")

TMIN, TMAX    = -0.5, 3.75          # janela do recorte (s) relativa ao ATTEMPT
BASELINE_S    = 0.0                 # 0.0 => sem correção de baseline
SAVE_PNG      = True               # True => salva figuras ao lado do CSV do sinal

# Cores para linhas de marcação / epochs
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
            if ts is None:
                continue
            lab = (row.get("label") or "").strip()
            t.append(ts); labels.append(lab)
    return np.asarray(t, float), labels

def read_signal_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if "lsl_time_s" not in header:
            raise RuntimeError("CSV do sinal precisa ter 'lsl_time_s'.")
        idx_ts = header.index("lsl_time_s")
        # canais começam após 'local_recv_s' (se existir) ou após lsl_time_s
        ch_start = header.index("local_recv_s")+1 if "local_recv_s" in header else idx_ts+1
        ch_names = header[ch_start:]
        t, X = [], []
        for row in r:
            try:
                ts = float(row[idx_ts])
                vals = [float(v) for v in row[ch_start:]]
            except:
                continue
            t.append(ts); X.append(vals)
    return np.asarray(t, float), np.asarray(X, float), ch_names

def estimate_fs(t):
    d = np.diff(t)
    d = d[(d > 0) & np.isfinite(d)]
    return 1/np.median(d) if d.size else 0.0

def nearest_index(t, x):
    i = np.searchsorted(t, x)
    if i <= 0: return 0
    if i >= len(t): return len(t)-1
    return i-1 if abs(t[i-1]-x) <= abs(t[i]-x) else i

def epoch_list(t_sig, X, events, fs, tmin, tmax):
    """Retorna lista de trials (cada trial = (n_times, C)) e time_rel (centrado no ATTEMPT)."""
    n_times = int(round((tmax - tmin) * fs)) + 1
    time_rel = np.arange(n_times)/fs + tmin
    s_ofs = int(round(tmin * fs))
    e_ofs = int(round(tmax * fs))
    trials = []
    for ev in events:
        i0 = nearest_index(t_sig, ev)
        s, e = i0 + s_ofs, i0 + e_ofs
        if s < 0 or e >= len(t_sig):
            continue
        seg = X[s:e+1, :]
        if seg.shape[0] == n_times:
            trials.append(seg)
    return trials, time_rel

def baseline_correct(trials, fs, tmin, baseline_s):
    if baseline_s <= 0: return trials
    i_end = int(round((0 - tmin) * fs))
    i_sta = max(0, i_end - int(round(baseline_s * fs)))
    out = []
    for seg in trials:
        if 0 <= i_sta < i_end <= seg.shape[0]:
            bl = seg[i_sta:i_end, :].mean(axis=0, keepdims=True)
            out.append(seg - bl)
        else:
            out.append(seg)
    return out

def stack_plot(ax, t, X, ch_names):
    """Stack RAW com offset automático (sem normalizar)."""
    stds = X.std(axis=0, ddof=1)
    med = float(np.median(stds[stds>0])) if np.any(stds>0) else 1.0
    offset = 4.0 * med
    for ci in range(X.shape[1]):
        ax.plot(t, X[:, ci] + ci*offset, color="k", lw=0.8)
    ax.set_yticks([i*offset for i in range(len(ch_names))], ch_names)
    ax.set_ylabel("Canais")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

def compute_offset_for_trials(trials):
    if not trials:
        return 1.0
    Xcat = np.concatenate(trials, axis=0)
    stds = Xcat.std(axis=0, ddof=1)
    med = float(np.median(stds[stds > 0])) if np.any(stds > 0) else 1.0
    return 4.0 * med

def plot_trial_stack(ax, time_rel, seg, ch_names, offset, color):
    C = seg.shape[1]
    for ci in range(C):
        ax.plot(time_rel, seg[:, ci] + ci*offset, color=color, lw=0.8)
    if ax.get_subplotspec().is_first_col():
        ax.set_yticks([i*offset for i in range(C)], ch_names)
    else:
        ax.set_yticks([])
    ax.axvline(0, color="#555", ls="--", lw=1.0, alpha=0.7)
    ax.grid(True, axis="x", ls="--", alpha=0.3)

def attempts_by_class(t_mark, labels):
    """
    Retorna dict:
      {'LEFT_MI_STIM': [t_attempts...], 'RIGHT_MI_STIM': [t_attempts...]}
    Cada ATTEMPT é associado à última pista anterior (LEFT/RIGHT).
    """
    out = {"LEFT_MI_STIM": [], "RIGHT_MI_STIM": []}
    last_cue = None
    for ts, lab in zip(t_mark, labels):
        if lab == "LEFT_MI_STIM":
            last_cue = "LEFT_MI_STIM"
        elif lab == "RIGHT_MI_STIM":
            last_cue = "RIGHT_MI_STIM"
        elif lab == "ATTEMPT":
            if last_cue in out:
                out[last_cue].append(ts)  # 0 s = ATTEMPT
            # se não houver pista anterior, ignora
    return out

def main():
    # ---- arquivos
    mark_path = MARKERS_FILE or find_latest(DATA_DIR, "*markers_*.csv")
    sig_path  = SIGNAL_FILE  or find_latest(DATA_DIR, "*signal_*.csv")

    print("Marcadores:", mark_path)
    t_mark, labels = read_markers_csv(mark_path)

    print("Sinal:", sig_path)
    t_sig, X, ch_names = read_signal_csv(sig_path)
    fs = estimate_fs(t_sig)
    print(f"Fs ~ {fs:.2f} Hz | canais: {len(ch_names)}")

    # ---- FIGURA 1: STACK RAW + linhas de marcação (todas as labels)
    t0 = min(t_sig[0], t_mark[0]) if (t_sig.size and t_mark.size) else (t_sig[0] if t_sig.size else 0.0)
    t_rel = t_sig - t0

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    stack_plot(ax1, t_rel, X, ch_names)
    for tm, lab in zip(t_mark, labels):
        ax1.axvline(tm - t0, color=COLOR.get(lab, "k"), lw=1.2, alpha=0.9)
    ax1.set_xlabel("Tempo (s)")
    ax1.set_title(f"Stack RAW + marcações  —  {os.path.basename(sig_path)}")
    plt.tight_layout()

    # ---- FIGURA 2: Epochs (todos os canais por subplot), 2 linhas (LEFT / RIGHT)
    # Alinhamento ao ATTEMPT (0 s) e classe definida pela última pista
    ev_by_cls = attempts_by_class(t_mark, labels)

    trials_by_cls = {}
    for cls in CLASSES:
        ev = np.asarray(ev_by_cls.get(cls, []), float)
        trials, time_rel = epoch_list(t_sig, X, ev, fs, TMIN, TMAX)
        trials = baseline_correct(trials, fs, TMIN, BASELINE_S)
        trials_by_cls[cls] = trials
        print(f"{cls} (ATTEMPT-alinhado): {len(trials)} trials")

    n_left  = len(trials_by_cls.get(CLASSES[0], []))
    n_right = len(trials_by_cls.get(CLASSES[1], []))
    ncols = max(n_left, n_right) if max(n_left, n_right) > 0 else 1

    fig2, axes = plt.subplots(2, ncols, figsize=(3.2*ncols, 6.4), sharex=True)
    if ncols == 1:
        axes = np.array(axes).reshape(2,1)

    for row, cls in enumerate(CLASSES):
        trials = trials_by_cls.get(cls, [])
        color = COLOR.get(cls, "k")
        offset = compute_offset_for_trials(trials)  # mesmo offset em toda a linha
        for col in range(ncols):
            ax = axes[row, col]
            if col < len(trials):
                plot_trial_stack(ax, time_rel, trials[col], ch_names, offset, color)
                ax.set_title(f"{cls} — trial {col+1} (0=ATTEMPT)", fontsize=9, color=color)
            else:
                ax.axis("off")
    axes[1,0].set_xlabel("Tempo relativo ao ATTEMPT (s)")
    fig2.suptitle("Epochs por tentativa — todos os canais empilhados (0 s = ATTEMPT)", y=0.98)
    plt.tight_layout(rect=[0,0,1,0.96])

    if SAVE_PNG:
        base = os.path.splitext(sig_path)[0]
        f1   = base + "_stack_raw_marks.png"
        f2   = base + "_epochs_trials_ALLCHANNELS_ATTEMPT0.png"
        fig1.savefig(f1, dpi=150); fig2.savefig(f2, dpi=150)
        print("Salvo:\n ", f1, "\n ", f2)
    else:
        plt.show()

if __name__ == "__main__":
    main()
