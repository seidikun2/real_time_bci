# -*- coding: utf-8 -*-
"""
plot_bandpass_stack_simple.py
- Lê o CSV de sinal (*signal_*.csv); se SIGNAL_FILE=None, pega o mais recente em DATA_DIR
- Aplica bandpass Butterworth [LOWCUT_HZ, HIGHCUT_HZ] com filtfilt (zero-fase)
- Plota SOMENTE os canais selecionados, empilhados com OFFSET manual
"""

import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============ CONFIG RÁPIDA ============
DATA_DIR      = r"C:\Users\User\Desktop\Dados"
SIGNAL_FILE   = None                 # ou caminho direto para um *_signal_*.csv
LOWCUT_HZ     = 0.5                 # borda inferior do bandpass (Hz)
HIGHCUT_HZ    = 500.0                 # borda superior do bandpass (Hz)
BP_ORDER      = 2                 # ordem do Butterworth
OFFSET        = 100.0                 # deslocamento vertical FIXO entre canais (mesmas unidades do sinal)
SAVE_PNG      = False

# Escolha dos canais (mude e rode de novo):
# SELECT_CHANNELS = ['ch1','ch2','ch3', 'ch20']  # por nome...
SELECT_CHANNELS = [1,2,4,5,7,9,10]            # ...ou por índice
# ======================================

def find_latest(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo {pattern} em {folder}")
    return max(files, key=os.path.getmtime)

def read_signal_csv(path):
    """Retorna t (s), X (N x C), ch_names. Requer coluna 'lsl_time_s'."""
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if "lsl_time_s" not in header:
            raise RuntimeError("CSV precisa ter coluna 'lsl_time_s'.")
        idx_ts   = header.index("lsl_time_s")
        ch_start = header.index("local_recv_s")+1 if "local_recv_s" in header else idx_ts+1
        ch_names = header[ch_start:]
        t, X = [], []
        for row in r:
            try:
                t.append(float(row[idx_ts]))
                X.append([float(v) for v in row[ch_start:]])
            except:
                continue
    return np.asarray(t, float), np.asarray(X, float), ch_names

def estimate_fs(t):
    d = np.diff(t)
    d = d[(d > 0) & np.isfinite(d)]
    return 1.0/np.median(d) if d.size else 0.0

def butter_bandpass_ba(low, high, fs, order=4):
    if fs <= 0: raise ValueError("Fs inválida.")
    nyq = fs/2.0
    wn  = [low/nyq, high/nyq]
    if not (0 < wn[0] < wn[1] < 1):
        raise ValueError(f"Banda inválida: [{low}, {high}] Hz para Fs={fs:.2f} Hz")
    b, a = butter(order, wn, btype='bandpass')
    return b, a

def select_indices(sel, names):
    if not sel: raise ValueError("SELECT_CHANNELS vazio.")
    if isinstance(sel[0], str):
        idxmap = {nm: i for i, nm in enumerate(names)}
        miss = [nm for nm in sel if nm not in idxmap]
        if miss: raise ValueError(f"Canais não encontrados: {miss}\nDisponíveis: {names}")
        return [idxmap[nm] for nm in sel]
    return [int(i) for i in sel]

def main():
    sig_path = SIGNAL_FILE or find_latest(DATA_DIR, "*signal_*.csv")
    print("Sinal:", sig_path)

    t, X, ch_names = read_signal_csv(sig_path)
    fs = estimate_fs(t)
    print(f"Fs ≈ {fs:.2f} Hz | canais: {len(ch_names)}")
    print("Canais no arquivo:", ch_names)

    sel_idx   = select_indices(SELECT_CHANNELS, ch_names)
    sel_names = [ch_names[i] for i in sel_idx]
    Xsel      = X[:, sel_idx]

    b, a = butter_bandpass_ba(LOWCUT_HZ, HIGHCUT_HZ, fs, order=BP_ORDER)
    Xf   = filtfilt(b, a, Xsel, axis=0)  # zero-fase

    t_rel = t - t[0]

    plt.figure(figsize=(12, 6))
    for k in range(Xf.shape[1]):
        y0 = k * OFFSET
        plt.hlines(y0, t_rel[0], t_rel[-1], linestyles=":", linewidth=0.6, alpha=0.6)
        plt.plot(t_rel, Xf[:, k] + y0, lw=0.9, label=sel_names[k])
    plt.yticks([k*OFFSET for k in range(len(sel_names))], sel_names)
    plt.xlabel("Tempo (s)")
    plt.ylabel(f"Canais (OFFSET={OFFSET:g})")
    plt.title(f"Bandpass Butterworth {LOWCUT_HZ:.1f}-{HIGHCUT_HZ:.1f} Hz (ordem {BP_ORDER}) — {os.path.basename(sig_path)}")
    plt.grid(True, axis="x", ls="--", alpha=0.3)
    plt.tight_layout()

    if SAVE_PNG:
        base = os.path.splitext(sig_path)[0]
        out = base + f"_BP{int(LOWCUT_HZ)}-{int(HIGHCUT_HZ)}Hz_STACK.png"
        plt.savefig(out, dpi=150)
        print("Figura salva em:", out)
    else:
        plt.show()

if __name__ == "__main__":
    main()
