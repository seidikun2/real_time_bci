# check_data.py
# -*- coding: utf-8 -*-
"""
Visualização rápida dos dados de treino (QC):

- Lê *markers* e *signal* em par (mesmo prefixo antes de '_markers_' / '_signal_').
- Lista pares disponíveis no diretório de treino e permite escolher um.
- Seleciona canais por índice base-1 (cfg.model.select_channels) ou todos.
- Aplica filtro passa-altas Butterworth (cfg.check_data.hp_cutoff_hz) com filtfilt.
- Plota stack RAW + marcações e epochs alinhados no ATTEMPT
  para classes definidas em cfg.check_data.classes.

Uso típico: opcional depois da calibração, chamado a partir do main:

    from check_data import run_check_data
    run_check_data(cfg)
"""

import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from collections import Counter
from typing import List, Tuple, Optional, Dict

from scipy.signal import butter, filtfilt

from config_models import AppConfig

# cores por label (deixo hardcoded mesmo)
COLOR = {
    "BASELINE"     : "#7f7f7f",
    "ATTENTION"    : "#ff7f0e",
    "LEFT_MI_STIM" : "#4169e1",
    "RIGHT_MI_STIM": "#dc143c",
    "ATTEMPT"      : "#2e8b57",
    "REST"         : "#800080",
}


def find_marker_signal_pairs(folder: str):
    """Encontra pares (markers, signal) com mesmo prefixo antes de '_markers_'."""
    markers = glob.glob(os.path.join(folder, "*markers_*.csv"))
    pairs   = []
    for m in markers:
        base_m = os.path.basename(m)
        if "_markers_" not in base_m:
            continue
        prefix, _ = base_m.split("_markers_", 1)
        pattern_s = os.path.join(folder, prefix + "_signal_*.csv")
        sig_files = glob.glob(pattern_s)
        if not sig_files:
            continue
        s = max(sig_files, key=os.path.getmtime)
        pairs.append((m, s))
    pairs = sorted(pairs, key=lambda p: os.path.getmtime(p[1]), reverse=True)
    return pairs


def choose_pair(folder: str):
    pairs = find_marker_signal_pairs(folder)
    if not pairs:
        raise FileNotFoundError(f"Nenhum par *_markers_*.csv / *_signal_*.csv em {folder}.")
    print(f"\nPares de arquivos encontrados em {folder}:")
    for i, (m, s) in enumerate(pairs, start=1):
        mt = dt.datetime.fromtimestamp(os.path.getmtime(m)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{i}] {os.path.basename(m)}  |  {os.path.basename(s)}   ({mt})")
    while True:
        choice = input("Selecione o número do par [1 = mais recente]: ").strip()
        if choice == "":
            idx = 1
        else:
            try:
                idx = int(choice)
            except ValueError:
                print("Entrada inválida, digite um número.")
                continue
        if 1 <= idx <= len(pairs):
            return pairs[idx-1]
        print("Número fora do intervalo, tente novamente.")


def resolve_pair(folder: str,
                 mark_explicit: Optional[str] = None,
                 sig_explicit: Optional[str] = None):
    if mark_explicit and sig_explicit:
        m = mark_explicit if os.path.isabs(mark_explicit) else os.path.join(folder, mark_explicit)
        s = sig_explicit  if os.path.isabs(sig_explicit)  else os.path.join(folder, sig_explicit)
        if not os.path.exists(m):
            raise FileNotFoundError(f"Arquivo de marcadores não existe: {m}")
        if not os.path.exists(s):
            raise FileNotFoundError(f"Arquivo de sinal não existe: {s}")
        return m, s
    return choose_pair(folder)


def read_markers_csv(path: str, code_map: Dict[int, str]):
    """Lê tempos (lsl_time_s), labels e códigos numéricos."""
    t, labels, codes = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = None
            if "lsl_time_s" in row and row["lsl_time_s"]:
                try:
                    ts = float(row["lsl_time_s"])
                except Exception:
                    ts = None
            if ts is None and row.get("iso_time"):
                try:
                    ts = dt.datetime.fromisoformat(row["iso_time"]).timestamp()
                except Exception:
                    ts = None
            if ts is None:
                continue

            lab = (row.get("label") or "").strip()

            c = None
            raw_code = (row.get("code") or "").strip()
            if raw_code != "":
                try:
                    c = int(raw_code)
                except Exception:
                    c = None

            # Se label vier vazio, tenta mapear pelo código
            if not lab and (c is not None) and (c in code_map):
                lab = code_map[c]

            t.append(ts)
            labels.append(lab)
            codes.append(c)
    return np.asarray(t, float), labels, codes


def read_signal_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r      = csv.reader(f)
        header = next(r)
        if "lsl_time_s" not in header:
            raise RuntimeError("CSV do sinal precisa ter coluna 'lsl_time_s'.")
        idx_ts   = header.index("lsl_time_s")
        ch_start = header.index("local_recv_s") + 1 if "local_recv_s" in header else idx_ts + 1
        ch_names = header[ch_start:]
        t, X     = [], []
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
    return 1/np.median(d) if d.size else 0.0


def butter_highpass(cut_hz, fs, order=4):
    nyq = fs/2.0
    wn  = cut_hz/nyq
    b,a = butter(order, wn, btype='high')
    return b,a


def nearest_index(t, x):
    i = np.searchsorted(t, x)
    if i <= 0:
        return 0
    if i >= len(t):
        return len(t)-1
    return i-1 if abs(t[i-1]-x) <= abs(t[i]-x) else i


def epoch_list(t_sig, X, events, fs, tmin, tmax):
    n_times  = int(round((tmax - tmin) * fs)) + 1
    time_rel = np.arange(n_times)/fs + tmin
    s_ofs    = int(round(tmin * fs))
    e_ofs    = int(round(tmax * fs))
    trials   = []
    for ev in events:
        i0 = nearest_index(t_sig, ev)
        s  = i0 + s_ofs
        e  = i0 + e_ofs
        if s < 0 or e >= len(t_sig):
            continue
        seg = X[s:e+1, :]
        if seg.shape[0] == n_times:
            trials.append(seg)
    return trials, time_rel


def baseline_correct(trials, fs, tmin, baseline_s):
    if baseline_s <= 0:
        return trials
    i_end = int(round((0 - tmin) * fs))
    i_sta = max(0, i_end - int(round(baseline_s * fs)))
    out   = []
    for seg in trials:
        if 0 <= i_sta < i_end <= seg.shape[0]:
            bl = seg[i_sta:i_end,:].mean(axis=0, keepdims=True)
            out.append(seg - bl)
        else:
            out.append(seg)
    return out


def stack_plot(ax, t, X, ch_names):
    stds   = X.std(axis=0, ddof=1)
    med    = float(np.median(stds[stds>0])) if np.any(stds>0) else 1.0
    offset = 4.0*med
    for ci in range(X.shape[1]):
        ax.plot(t, X[:,ci] + ci*offset, color="k", lw=0.8)
    ax.set_yticks([i*offset for i in range(len(ch_names))], ch_names)
    ax.set_ylabel("Canais")
    ax.grid(True, axis="x", ls="--", alpha=0.3)


def compute_offset_for_trials(trials):
    if not trials:
        return 1.0
    Xcat = np.concatenate(trials, axis=0)
    stds = Xcat.std(axis=0, ddof=1)
    med  = float(np.median(stds[stds>0])) if np.any(stds>0) else 1.0
    return 4.0*med


def plot_trial_stack(ax, time_rel, seg, ch_names, offset, color):
    C = seg.shape[1]
    for ci in range(C):
        ax.plot(time_rel, seg[:,ci] + ci*offset, color=color, lw=0.8)
    if ax.get_subplotspec().is_first_col():
        ax.set_yticks([i*offset for i in range(C)], ch_names)
    else:
        ax.set_yticks([])
    ax.axvline(0, color="#555", ls="--", lw=1.0, alpha=0.7)
    ax.grid(True, axis="x", ls="--", alpha=0.3)


def attempts_by_class(t_mark, labels, codes, code_map: Dict[int, str]):
    """
    Usa preferencialmente o label textual; se estiver vazio,
    tenta inferir a partir do codigo numérico (code_map).
    """
    out      = {"LEFT_MI_STIM": [], "RIGHT_MI_STIM": []}
    last_cue = None

    for ts, lab, c in zip(t_mark, labels, codes):
        lab_eff = lab
        if (not lab_eff) and (c is not None) and (c in code_map):
            lab_eff = code_map[c]

        if lab_eff in ("LEFT_MI_STIM", "RIGHT_MI_STIM"):
            last_cue = lab_eff
        elif lab_eff == "ATTEMPT" and last_cue in out:
            out[last_cue].append(ts)

    return out


def select_channel_indices(selection: List[int],
                           ch_names: List[str],
                           index_base: int = 1) -> List[int]:
    """
    Seleciona canais por índice base-1.
    selection: lista de índices (ex.: [1,2,4]) ou [] para todos.
    """
    if not selection:
        return list(range(len(ch_names)))
    idx = [int(v) - (1 if index_base == 1 else 0) for v in selection]
    for i in idx:
        if i < 0 or i >= len(ch_names):
            raise ValueError(
                f"Índice de canal {i} fora de [0, {len(ch_names)-1}]. "
                f"Verifique select_channels={selection} e index_base={index_base}."
            )
    return idx


def run_check_data(cfg: AppConfig,
                   mode: str = "train",
                   markers_file: Optional[str] = None,
                   signal_file: Optional[str] = None,
                   save_png: Optional[bool] = None):
    """
    QC/visualização dos dados de uma sessão.

    - `mode`: subpasta dentro de S{session_id}/session_type (ex.: "train")
    - usa cfg.model.select_channels (índices base-1); se lista vazia, usa todos.
    - parâmetros de filtro e epochs vêm de cfg.check_data.
    """
    cdcfg = cfg.check_data

    HP_CUTOFF_HZ = cdcfg.hp_cutoff_hz
    HP_ORDER     = cdcfg.hp_order
    TMIN         = cdcfg.tmin
    TMAX         = cdcfg.tmax
    BASELINE_S   = cdcfg.baseline_s
    CLASSES      = tuple(cdcfg.classes)
    if save_png is None:
        save_png = cdcfg.save_png

    # diretório de dados
    data_dir = os.path.join(
        cfg.experiment.log_root,
        cfg.experiment.subject_id,
        f"S{cfg.experiment.session_id}",
        cfg.experiment.session_type,
        mode,
    )
    print(f"\n[check] Procurando dados em: {data_dir}")

    # escolha de arquivos
    mark_path, sig_path = resolve_pair(data_dir, markers_file, signal_file)

    print("\n[check] Usando arquivos:")
    print("  Marcadores:", mark_path)
    print("  Sinal:     ", sig_path)

    code_map = cfg.codes.code_map

    # leitura
    t_mark, labels, codes   = read_markers_csv(mark_path, code_map)
    t_sig, X_full, ch_all   = read_signal_csv(sig_path)
    fs                      = estimate_fs(t_sig)
    if fs <= 0:
        raise RuntimeError("Fs estimado == 0. Verifique a coluna 'lsl_time_s' do sinal.")
    print(f"[check] Fs estimado ~ {fs:.2f} Hz | canais: {len(ch_all)}")

    # resumo dos labels
    print("\n[check] Resumo de labels lidos:")
    for lab, cnt in Counter(labels).most_common():
        print(f"  {lab or '(vazio)'}: {cnt}")

    print("\n[check] Canais disponíveis:")
    for i, nm in enumerate(ch_all):
        print(f"  {i:2d}: {nm}")

    # seleção de canais por índice base-1 (mesmos índices do cfg.model)
    select_channels = cfg.model.select_channels or []
    print(f"\n[check] select_channels (base-1) = {select_channels or 'todos'}")

    sel_idx  = select_channel_indices(select_channels, ch_all, index_base=1)
    ch_names = [ch_all[i] for i in sel_idx]
    X        = X_full[:, sel_idx]

    # filtro HP para visualização
    b,a = butter_highpass(HP_CUTOFF_HZ, fs, HP_ORDER)
    X   = filtfilt(b, a, X, axis=0)
    print(f"\n[check] Filtro passa-altas aplicado: {HP_CUTOFF_HZ} Hz, ordem {HP_ORDER}")

    # FIGURA 1: stack
    t0    = min(t_sig[0], t_mark[0]) if (t_sig.size and t_mark.size) else t_sig[0]
    t_rel = t_sig - t0
    fig1, ax1 = plt.subplots(figsize=(12,6))
    stack_plot(ax1, t_rel, X, ch_names)
    for tm, lab in zip(t_mark, labels):
        ax1.axvline(tm - t0, color=COLOR.get(lab, "k"), lw=1.2, alpha=0.9)
    ax1.set_xlabel("Tempo (s)")
    ax1.set_title(f"Stack RAW (HP {HP_CUTOFF_HZ} Hz) — {os.path.basename(sig_path)}")

    # FIGURA 2: epochs por classe
    ev_by_cls      = attempts_by_class(t_mark, labels, codes, code_map)
    trials_by_cls  = {}
    for cls in CLASSES:
        ev              = np.asarray(ev_by_cls.get(cls, []), float)
        trials, time_rel= epoch_list(t_sig, X, ev, fs, TMIN, TMAX)
        trials          = baseline_correct(trials, fs, TMIN, BASELINE_S)
        trials_by_cls[cls] = trials
        print(f"[check] {cls}: {len(trials)} trials")

    ncols     = max(len(trials_by_cls[c]) for c in CLASSES) or 1
    fig2,axes = plt.subplots(2, ncols, figsize=(3.2*ncols, 6.4), sharex=True)
    if ncols == 1:
        axes = np.array(axes).reshape(2,1)

    for row, cls in enumerate(CLASSES):
        trials = trials_by_cls[cls]
        color  = COLOR.get(cls, "k")
        offset = compute_offset_for_trials(trials)
        for col in range(ncols):
            ax = axes[row, col]
            if col < len(trials):
                plot_trial_stack(ax, time_rel, trials[col], ch_names, offset, color)
                ax.set_title(f"{cls} — trial {col+1}", fontsize=9, color=color)
            else:
                ax.axis("off")
    axes[1,0].set_xlabel("Tempo relativo ao ATTEMPT (s)")
    fig2.suptitle(f"Epochs (HP {HP_CUTOFF_HZ} Hz) — canais selecionados", y=0.98)

    if save_png:
        base = os.path.splitext(sig_path)[0]
        fig1.savefig(base + "_stack_hp.png", dpi=150)
        fig2.savefig(base + "_epochs_hp.png", dpi=150)
        print(f"[check] Figuras salvas em:\n  {base + '_stack_hp.png'}\n  {base + '_epochs_hp.png'}")
    else:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    from config_models import load_config
    cfg = load_config("config.yaml")
    run_check_data(cfg)
