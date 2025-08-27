# plot_graz_data.py
import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# ===================== CONFIGURAÇÃO RÁPIDA =====================
DEFAULT_DIR   = r"C:\Users\User\Desktop\Dados"  # pasta dos CSVs
MARKERS_FILE  = None   # se None, pega o mais recente (padrão: graz_markers_*.csv)
SIGNAL_FILE   = None   # se None, pega o mais recente (padrão: graz_signal_*.csv)

# Plot inicial (sinal empilhado + marcadores)
NORMALIZE_STACK = False   # normalizar cada canal antes de empilhar (z-score)
SAVE_PNG        = False   # salvar PNGs (na mesma pasta do sinal) em vez de só mostrar

# Epochs / tentativas
TMIN      = -0.5   # início do recorte (s) relativo ao marcador
TMAX      =  2.0   # fim do recorte (s) relativo ao marcador
BASELINE  =  0.2   # segundos de baseline ([-BASELINE, 0]); 0.0 desliga correção
CLASSES   = ["LEFT_MI_STIM", "RIGHT_MI_STIM"]  # rótulos que entram na análise
SHOW_TRIALS = False  # sobrepor traçados de cada tentativa (pode ficar pesado)

# Cores por rótulo (ajuste à vontade)
COLOR_MAP = {
    "BASELINE":      "#7f7f7f",  # gray
    "ATTENTION":     "#ff7f0e",  # orange
    "LEFT_MI_STIM":  "#4169e1",  # royalblue
    "RIGHT_MI_STIM": "#dc143c",  # crimson
    "ATTEMPT":       "#2e8b57",  # seagreen
    "REST":          "#800080",  # purple
}
# ===============================================================


def find_latest(folder: str, pattern: str) -> str:
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em: {os.path.join(folder, pattern)}")
    return max(files, key=os.path.getmtime)

def read_markers_csv(path: str):
    """
    Lê CSV de marcadores (graz_markers_*.csv) e retorna:
      - t (np.array de tempos LSL, s)
      - labels (lista de rótulos)
    """
    t, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # preferir LSL
            ts = None
            try:
                ts = float(row["lsl_time_s"])
            except Exception:
                iso = row.get("iso_time", "")
                if iso:
                    try:
                        ts = dt.datetime.fromisoformat(iso).timestamp()
                    except Exception:
                        ts = None
            if ts is None:
                continue
            lab = (row.get("label") or "UNKNOWN").strip()
            t.append(ts); labels.append(lab)
    return np.asarray(t, dtype=float), labels

def read_signal_csv(path: str, downsample: int = 1):
    """
    Lê CSV de sinal (graz_signal_*.csv) e retorna:
      - t (np.array tempo LSL, s)
      - X (np.array [N amostras, C canais])
      - ch_names (lista)
    """
    t = []; data = []; ch_names = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        try:
            idx_ts = header.index("lsl_time_s")
            idx_local = header.index("local_recv_s")
        except ValueError:
            raise RuntimeError("Cabeçalho do sinal precisa conter 'lsl_time_s' e 'local_recv_s'.")
        ch_names = header[idx_local + 1 :]
        for i, row in enumerate(r):
            if downsample > 1 and (i % downsample) != 0:
                continue
            try:
                ts = float(row[idx_ts])
                ch_vals = [float(v) for v in row[idx_local + 1 :]]
            except Exception:
                continue
            t.append(ts); data.append(ch_vals)
    if not t:
        raise RuntimeError(f"Nenhuma amostra lida de {path}")
    X = np.asarray(data, dtype=float)
    return np.asarray(t, dtype=float), X, ch_names

def estimate_fs(t_sig: np.ndarray) -> float:
    dtv = np.diff(t_sig)
    dtv = dtv[(dtv > 0) & np.isfinite(dtv)]
    if len(dtv) == 0:
        return 0.0
    med = np.median(dtv)
    return float(1.0 / med) if med > 0 else 0.0

def stack_plot(ax, t, X, ch_names, offset=None, normalize=False, line_kw=None):
    """
    Empilha canais com offset vertical.
    """
    if normalize:
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        Xp = (X - X.mean(axis=0)) / std
    else:
        Xp = X
    if offset is None:
        if normalize:
            offset = 2.0
        else:
            med = np.median(Xp.std(axis=0, ddof=1))
            offset = 4.0 * (med if med > 0 else 1.0)
    if line_kw is None:
        line_kw = dict(color="black", linewidth=0.8, alpha=0.8)
    for ci in range(Xp.shape[1]):
        y = Xp[:, ci] + ci * offset
        ax.plot(t, y, **line_kw)
    ax.set_yticks([i * offset for i in range(len(ch_names))], ch_names)
    ax.set_ylabel("Canais")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

def overlay_markers(ax, t_mark, labels, t0=0.0, colors=COLOR_MAP, add_legend=True):
    used = {}
    for tm, lab in zip(t_mark, labels):
        x = tm - t0
        c = colors.get(lab, "k")
        ln = ax.axvline(x, color=c, linestyle='-', linewidth=1.5, alpha=0.9)
        if add_legend and lab not in used:
            used[lab] = ln
    if add_legend and used:
        ax.legend(used.values(), used.keys(), title="Marcadores", loc="upper right", frameon=False)

def nearest_index(t: np.ndarray, target: float) -> int:
    i = np.searchsorted(t, target)
    if i <= 0: return 0
    if i >= len(t): return len(t) - 1
    return i-1 if abs(t[i-1] - target) <= abs(t[i] - target) else i

def epoch_data(t_sig: np.ndarray, X: np.ndarray, events: np.ndarray,
               fs: float, tmin: float, tmax: float):
    """
    Retorna epochs shape: (n_trials, n_times, n_channels), time_rel (n_times,)
    """
    n_times = int(round((tmax - tmin) * fs)) + 1
    time_rel = np.arange(n_times) / fs + tmin
    epochs = []
    kept = 0; dropped = 0
    step_start = int(round(tmin * fs))
    step_end   = int(round(tmax * fs))
    for ev in events:
        i0 = nearest_index(t_sig, ev)
        s = i0 + step_start
        e = i0 + step_end
        if s < 0 or e >= len(t_sig):
            dropped += 1
            continue
        seg = X[s:e+1, :]
        if seg.shape[0] != n_times:
            dropped += 1
            continue
        epochs.append(seg); kept += 1
    if kept == 0:
        return None, time_rel, kept, dropped
    return np.stack(epochs, axis=0), time_rel, kept, dropped

def baseline_correct(epochs: np.ndarray, fs: float, tmin: float, baseline: float):
    """
    Subtrai média do intervalo [-baseline, 0] por trial/canal.
    """
    n_times = epochs.shape[1]
    i_end = int(round((0 - tmin) * fs))
    i_start = max(0, i_end - int(round(baseline * fs)))
    if i_end <= i_start or i_start < 0 or i_end > n_times:
        return epochs
    bl = epochs[:, i_start:i_end, :].mean(axis=1, keepdims=True)
    return epochs - bl

def plot_class_averages(fig_title, classes_info, ch_names, normalize=False):
    """
    classes_info: lista de dicts:
        {'label': 'LEFT_MI_STIM', 'time': time_rel, 'mean': (n_times, C),
         'n': int, 'color': '#...'}
    """
    n_classes = len(classes_info)
    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 3.5*n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]
    for ax, info in zip(axes, classes_info):
        t = info['time']; M = info['mean']; color = info['color']
        line_kw = dict(color=color, linewidth=1.2, alpha=0.95)
        stack_plot(ax, t, M, ch_names, normalize=normalize, line_kw=line_kw)
        ax.axvline(0.0, color="#555", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"Média por canal — {info['label']} (n={info.get('n','?')})")
    axes[-1].set_xlabel("Tempo relativo ao marcador (s)")
    fig.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def main():
    # Seleciona arquivos (ou usa os mais recentes)
    mark_path = MARKERS_FILE or find_latest(DEFAULT_DIR, "graz_markers_*.csv")
    sig_path  = SIGNAL_FILE  or find_latest(DEFAULT_DIR, "graz_signal_*.csv")

    print(f"Lendo marcadores: {mark_path}")
    t_mark, labels = read_markers_csv(mark_path)

    print(f"Lendo sinal: {sig_path}")
    t_sig, X, ch_names = read_signal_csv(sig_path, downsample=1)

    # Figura 1 — stack do sinal + linhas de marcadores
    t0 = min(t_sig[0], t_mark[0]) if (len(t_mark) and len(t_sig)) else (t_sig[0] if len(t_sig) else 0.0)
    t_sig_rel = t_sig - t0
    title1 = f"Graz — {os.path.basename(sig_path)}  |  {os.path.basename(mark_path)}"

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    stack_plot(ax1, t_sig_rel, X, ch_names, normalize=NORMALIZE_STACK)
    overlay_markers(ax1, t_mark, labels, t0=t0, colors=COLOR_MAP, add_legend=True)
    ax1.set_xlabel("Tempo (s)")
    ax1.set_title(title1)
    plt.tight_layout()

    # Epochs + médias por classe
    fs = estimate_fs(t_sig)
    print(f"Fs estimada: {fs:.3f} Hz")

    classes_info = []
    for cls in CLASSES:
        ev_times = np.asarray([tm for tm, lab in zip(t_mark, labels) if lab == cls], dtype=float)
        if ev_times.size == 0:
            print(f"Aviso: sem eventos para '{cls}'.")
            continue
        E, time_rel, kept, dropped = epoch_data(t_sig, X, ev_times, fs, TMIN, TMAX)
        if E is None:
            print(f"Aviso: nenhum recorte válido para '{cls}'.")
            continue
        print(f"Classe {cls}: recortes ok={kept}, descartados={dropped}, shape={E.shape}")

        if BASELINE > 0:
            E = baseline_correct(E, fs, TMIN, BASELINE)

        M = E.mean(axis=0)  # (n_times, C)

        if SHOW_TRIALS:
            fig_t, ax_t = plt.subplots(figsize=(12, 6))
            # single trials
            for k in range(E.shape[0]):
                stack_plot(ax_t, time_rel, E[k], ch_names,
                           normalize=NORMALIZE_STACK,
                           line_kw=dict(color=COLOR_MAP.get(cls, "black"), linewidth=0.5, alpha=0.12))
            # média por cima
            stack_plot(ax_t, time_rel, M, ch_names,
                       normalize=NORMALIZE_STACK,
                       line_kw=dict(color=COLOR_MAP.get(cls, "black"), linewidth=1.6, alpha=0.95))
            ax_t.axvline(0.0, color="#555", linestyle="--", linewidth=1.0, alpha=0.7)
            ax_t.set_title(f"Trials + média — {cls} (n={E.shape[0]})")
            ax_t.set_xlabel("Tempo relativo ao marcador (s)")
            plt.tight_layout()
            if SAVE_PNG:
                outp = os.path.splitext(sig_path)[0] + f"_{cls}_trials.png"
                plt.savefig(outp, dpi=150)
                print(f"Figura (trials) salva em: {outp}")

        classes_info.append({
            'label': cls,
            'time': time_rel,
            'mean': M,
            'n': E.shape[0],
            'color': COLOR_MAP.get(cls, "black"),
        })

    if classes_info:
        title2 = f"Epochs — médias por classe ({os.path.basename(sig_path)})"
        fig2 = plot_class_averages(title2, classes_info, ch_names, normalize=NORMALIZE_STACK)
        if SAVE_PNG:
            outp = os.path.splitext(sig_path)[0] + "_epochs_mean.png"
            plt.savefig(outp, dpi=150)
            print(f"Figura (médias) salva em: {outp}")

    if SAVE_PNG:
        outp = os.path.splitext(sig_path)[0] + "_stack_plot.png"
        plt.savefig(outp, dpi=150)
        print(f"Figura (stack) salva em: {outp}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
