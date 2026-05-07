# decoder_calibration.py
# -*- coding: utf-8 -*-
import os, glob, re, sys, json, platform
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.tangentspace import tangent_space
import pickle
import sklearn
import pyriemann

from config_models import AppConfig


# ---------- utilitários de arquivos (pares) ----------
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
        s = max(sig_files, key=os.path.getmtime)  # sinal mais recente p/ esse prefixo
        pairs.append((m, s))
    pairs = sorted(pairs, key=lambda p: os.path.getmtime(p[1]), reverse=True)
    return pairs


def choose_pair(folder: str):
    pairs = find_marker_signal_pairs(folder)
    if not pairs:
        raise FileNotFoundError(f"Nenhum par *_markers_*.csv / *_signal_*.csv em {folder}.")
    print(f"\nPares de arquivos encontrados em {folder}:")
    for i, (m, s) in enumerate(pairs, start=1):
        mt = pd.to_datetime(os.path.getmtime(m), unit="s").strftime("%Y-%m-%d %H:%M:%S")
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
    """Se arquivos forem dados, usa; senão escolhe par pelo prefixo."""
    if mark_explicit and sig_explicit:
        m = mark_explicit if os.path.isabs(mark_explicit) else os.path.join(folder, mark_explicit)
        s = sig_explicit  if os.path.isabs(sig_explicit)  else os.path.join(folder, sig_explicit)
        if not os.path.exists(m):
            raise FileNotFoundError(f"Arquivo de marcadores não existe: {m}")
        if not os.path.exists(s):
            raise FileNotFoundError(f"Arquivo de sinal não existe: {s}")
        return m, s
    return choose_pair(folder)


# ---------------- leitura de CSVs ----------------
def read_markers_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if "lsl_time_s" in df.columns:
        t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)
    elif "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
    else:
        raise KeyError("Marcadores precisam de 'lsl_time_s' (ou 'time_s').")

    if "label" in df.columns:
        labels = df["label"].astype(str).str.strip().tolist()
    elif "event" in df.columns:
        labels = df["event"].astype(str).str.strip().tolist()
    elif "code" in df.columns:
        code = pd.to_numeric(df["code"], errors="coerce").fillna(-1).astype(int).to_numpy()
        cmap = {1:"BASELINE",2:"ATTENTION",3:"LEFT_MI_STIM",4:"RIGHT_MI_STIM",5:"ATTEMPT",6:"REST"}
        labels = [cmap.get(int(c), "UNKNOWN") for c in code]
    else:
        raise KeyError("Marcadores precisam de 'label' ou 'code' ou 'event'.")

    ok = np.isfinite(t)
    return t[ok], [labels[i] for i in np.where(ok)[0]]


def read_signal_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path, low_memory=False)
    if "lsl_time_s" not in df.columns:
        raise KeyError("Sinal precisa ter a coluna 'lsl_time_s'.")
    t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)

    # tenta achar ch1..chN
    ch_cols = [c for c in df.columns if re.match(r"(?i)^ch\d+$", c)]
    if not ch_cols:
        cols = df.columns.tolist()
        start = 0
        for meta in ("iso_time", "lsl_time_s", "local_recv_s"):
            if meta in df.columns:
                start += 1
        ch_cols = cols[start:] if len(cols) > start else []
    if not ch_cols:
        raise KeyError("Não encontrei colunas de canais (ex.: ch1..chN).")

    X = df[ch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    return t[ok], X[ok, :], ch_cols


# ----------------- processamento ----------------
def bandpass(data: np.ndarray, fs: float, order: int, band: Tuple[float, float]) -> np.ndarray:
    low, high = band
    if not (0 < low < high < fs/2):
        raise ValueError(f"Banda inválida {band} para Fs={fs:.2f} Hz.")
    b, a = signal.butter(order, band, btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, data.T).T


def nearest_index(t: np.ndarray, x: float) -> int:
    i = np.searchsorted(t, x)
    if i <= 0:
        return 0
    if i >= len(t):
        return len(t)-1
    return i-1 if abs(t[i-1]-x) <= abs(t[i]-x) else i


def attempts_by_class(t_mark: np.ndarray, labels: List[str]) -> Dict[str, List[float]]:
    out = {"LEFT_MI_STIM": [], "RIGHT_MI_STIM": []}
    last = None
    for ts, lab in zip(t_mark, labels):
        if lab in out:
            last = lab
        elif lab == "ATTEMPT" and last in out:
            out[last].append(ts)
    return out


def epoch_trials_simple(t_sig: np.ndarray, X: np.ndarray, attempts: Dict[str, List[float]],
                        label_map: Dict[str, int], fs: float,
                        epoch_s: float, offset_s: float, seed: int = 42):
    """Uma época fixa por tentativa (C x T)."""
    n     = int(round(epoch_s * fs))
    shift = int(round(offset_s * fs))

    epochs, y, trial_id, t_rel = [], [], [], []
    cur   = 0
    for cls, times in attempts.items():
        for t0 in times:
            i0 = nearest_index(t_sig, t0) + shift
            i1 = i0 + n
            if i0 < 0 or i1 > len(t_sig):
                continue
            epochs.append(X[i0:i1, :].T)  # (C,T)
            y.append(label_map[cls])
            trial_id.append(cur)
            t_rel.append(t_sig[i0 + n//2] - t0)
            cur += 1

    if not epochs:
        raise ValueError("Nenhuma época válida — ajuste epoch_s / trial_offset_s.")
    Xw      = np.stack(epochs, axis=0)  # (N,C,T)
    y       = np.asarray(y, int)
    trial_id= np.asarray(trial_id, int)
    t_rel   = np.asarray(t_rel, float)

    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(y))
    return Xw[perm], y[perm], trial_id[perm], t_rel[perm]


def pick_cmean_pca_via_cv(X: np.ndarray, y: np.ndarray, trial_id: np.ndarray,
                          pca_dim: int, svc_c: float, n_rep: int = 20, seed: int = 42):
    rng         = np.random.default_rng(seed)
    uniq_trials = np.unique(trial_id)
    if len(uniq_trials) < 2:
        raise ValueError("Trials insuficientes (>=2).")

    best_acc, best_c, best_pca = -np.inf, None, None
    val_size = min(4, max(1, len(uniq_trials)//5), len(uniq_trials)-1)

    for _ in range(n_rep):
        va_trials = rng.choice(uniq_trials, size=val_size, replace=False)
        tr_mask   = ~np.isin(trial_id, va_trials)
        va_mask   = ~tr_mask
        if not np.any(tr_mask) or not np.any(va_mask):
            continue

        cov_tr = Covariances("oas").transform(X[tr_mask])
        cov_va = Covariances("oas").transform(X[va_mask])
        cmean  = mean_covariance(cov_tr)

        ts_tr  = tangent_space(cov_tr, cmean)
        ts_va  = tangent_space(cov_va, cmean)

        dim    = max(1, min(pca_dim, ts_tr.shape[1]))
        pca    = PCA(n_components=dim).fit(ts_tr)
        Xtr, Xva = pca.transform(ts_tr), pca.transform(ts_va)

        clf    = SVC(kernel="linear", C=svc_c)
        clf.fit(Xtr, y[tr_mask])
        acc    = clf.score(Xva, y[va_mask])

        if acc > best_acc:
            best_acc, best_c, best_pca = acc, cmean, pca

    if best_c is None:
        cov      = Covariances("oas").transform(X)
        best_c   = mean_covariance(cov)
        ts_all   = tangent_space(cov, best_c)
        dim      = max(1, min(pca_dim, ts_all.shape[1]))
        best_pca = PCA(n_components=dim).fit(ts_all)
        best_acc = float("nan")

    return best_c, best_pca, float(best_acc)


def cv_trialwise_accuracies(Xw: np.ndarray, y: np.ndarray, trial_id: np.ndarray,
                            pca_dim: int, svc_c: float, n_splits: int = 5, seed: int = 42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr_idx, va_idx in sgkf.split(Xw, y, groups=trial_id):
        cov_tr = Covariances("oas").transform(Xw[tr_idx])
        cmean  = mean_covariance(cov_tr)
        ts_tr  = tangent_space(cov_tr, cmean)
        ts_va  = tangent_space(Covariances("oas").transform(Xw[va_idx]), cmean)

        dim = max(1, min(pca_dim, ts_tr.shape[1]))
        pca = PCA(n_components=dim).fit(ts_tr)
        Xtr, Xva = pca.transform(ts_tr), pca.transform(ts_va)

        clf = SVC(kernel="linear", C=svc_c)
        clf.fit(Xtr, y[tr_idx])
        acc = clf.score(Xva, y[va_idx])
        accs.append(float(acc))
    return accs


def plot_pca_scatter(X_pca: np.ndarray, y: np.ndarray, clf: SVC, out_png: str):
    Xp = X_pca if X_pca.shape[1] > 1 else np.c_[X_pca[:, 0], np.zeros_like(X_pca[:, 0])]
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in np.unique(y):
        m = (y == c)
        ax.scatter(Xp[m, 0], Xp[m, 1], s=20, alpha=0.8, label=f"class {int(c)}")
        mu = Xp[m].mean(axis=0)
        ax.plot(mu[0], mu[1], marker="x", ms=10, mew=2)
    if hasattr(clf, "coef_") and Xp.shape[1] >= 2:
        w = clf.coef_[0]; b = clf.intercept_[0]
        if abs(w[1]) > 1e-9:
            xs = np.linspace(Xp[:,0].min()-1, Xp[:,0].max()+1, 200)
            ys = -(w[0]*xs + b)/w[1]
            ax.plot(xs, ys, lw=1.2, alpha=0.9, label="SVM boundary")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA (LEFT vs RIGHT)")
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------- seleção de canais ----------
def select_channel_indices(select_by: str,
                           selection: List,
                           ch_names: List[str],
                           index_base: int = 0) -> List[int]:
    """
    selection: lista de índices OU nomes.
      - se select_by == "index": índices inteiros, com base definida por index_base
      - se select_by == "name":  nomes dos canais
    """
    if selection is None or len(selection) == 0:
        return list(range(len(ch_names)))

    if select_by.lower() == "name":
        name_to_idx = {nm: i for i, nm in enumerate(ch_names)}
        missing = [nm for nm in selection if nm not in name_to_idx]
        if missing:
            raise ValueError(f"Canais não encontrados (por nome): {missing}")
        return [name_to_idx[nm] for nm in selection]

    # select_by == "index"
    idx = [int(v) - (1 if index_base == 1 else 0) for v in selection]
    if any((i < 0 or i >= len(ch_names)) for i in idx):
        raise ValueError(f"Algum índice está fora do range [0, {len(ch_names)-1}] (após base).")
    return idx


# ===================== PIPELINE =======================
def run_calibration(cfg: AppConfig,
                    markers_file: Optional[str] = None,
                    signal_file: Optional[str] = None):
    """
    Calibração do decodificador (Etapa 2).

    - Lê logs de treino em:
        log_root / subject_id / S{session_id} / session_type / train
    - Usa parâmetros de modelo em cfg.model
    - Seleção de canais por índice base-1 (cfg.model.select_channels)
    """
    # Diretório dos dados de treino
    data_dir = os.path.join(
        cfg.experiment.log_root,
        cfg.experiment.subject_id,
        f"S{cfg.experiment.session_id}",
        cfg.experiment.session_type,
        "train",
    )
    print(f"[calib] Procurando dados de treino em: {data_dir}")

    # Parâmetros do modelo (vindo do config)
    mcfg = cfg.model
    FS_HZ          = mcfg.fs_hz
    BP_ORDER       = mcfg.bp_order
    BP_BAND        = tuple(mcfg.bp_band)
    EPOCH_S        = mcfg.epoch_s
    TRIAL_OFFSET_S = mcfg.trial_offset_s
    PCA_DIM        = mcfg.pca_dim
    SVC_C          = mcfg.svc_c
    CV_SPLITS      = mcfg.cv_splits
    RNG_SEED       = mcfg.rng_seed
    SELECT_CHANNELS = mcfg.select_channels or []

    # Escolha do par marcador+sinal
    markers_csv, signal_csv = resolve_pair(data_dir, markers_file, signal_file)
    print(f"[calib] Marcadores: {markers_csv}")
    print(f"[calib] Sinal     : {signal_csv}")

    # leitura
    t_mark, labels       = read_markers_csv(markers_csv)
    t_sig, X_all, ch_all = read_signal_csv(signal_csv)
    print(f"[calib] Fs (config)={FS_HZ:.1f} Hz | samples={X_all.shape[0]} | canais={len(ch_all)}")

    # seleção de canais: índice, base 1
    sel_idx = select_channel_indices(
        select_by="index",
        selection=SELECT_CHANNELS,
        ch_names=ch_all,
        index_base=1,   # base-1 fixa
    )
    ch_sel  = [ch_all[i] for i in sel_idx]
    X_sel   = X_all[:, sel_idx]
    print(f"[calib] Seleção de canais (base-1): "
          f"{SELECT_CHANNELS if SELECT_CHANNELS else 'todos'} -> {ch_sel}")

    # filtro
    Xf = bandpass(X_sel, FS_HZ, BP_ORDER, BP_BAND)
    print(f"[calib] Sinal filtrado: {Xf.shape} (samples x canais)")

    # trials (ATTEMPT após LEFT/RIGHT)
    atts = attempts_by_class(t_mark, labels)
    print(f"[calib] ATTEMPTs: LEFT={len(atts['LEFT_MI_STIM'])}, RIGHT={len(atts['RIGHT_MI_STIM'])}")

    label_map = {"LEFT_MI_STIM": 0, "RIGHT_MI_STIM": 1}
    Xw, y, trial_id, t_rel = epoch_trials_simple(
        t_sig, Xf, atts, label_map, FS_HZ, EPOCH_S, TRIAL_OFFSET_S, RNG_SEED
    )
    print(f"[calib] Épocas: N={Xw.shape[0]} | C={Xw.shape[1]} | T={Xw.shape[2]} "
          f"| trials={len(np.unique(trial_id))}")

    # CV externa
    accs = cv_trialwise_accuracies(Xw, y, trial_id, PCA_DIM, SVC_C, CV_SPLITS, RNG_SEED)
    print(f"[calib] Acurácias CV: {[f'{a:.3f}' for a in accs]}")
    print(f"[calib] Média={np.mean(accs):.3f} | DP={np.std(accs, ddof=1) if len(accs)>1 else 0.0:.3f}")

    # Seleção de C_mean e PCA
    best_cmean, best_pca, best_acc = pick_cmean_pca_via_cv(
        Xw, y, trial_id, pca_dim=PCA_DIM, svc_c=SVC_C, n_rep=20, seed=RNG_SEED
    )
    print(f"[calib] melhor acurácia validação interna ~ {best_acc:.3f}")

    # Treino final
    cov = Covariances("oas").transform(Xw)
    ts  = tangent_space(cov, best_cmean)
    Xp  = best_pca.transform(ts)

    clf = SVC(kernel="linear", C=SVC_C, probability=True, random_state=RNG_SEED)
    clf.fit(Xp, y)

    # caminhos
    base       = os.path.splitext(os.path.basename(signal_csv))[0]
    out_prefix = os.path.join(os.path.dirname(signal_csv), base)
    pca_png    = out_prefix + "_pca.png"

    # figura PCA
    plot_pca_scatter(Xp, y, clf, pca_png)
    print(f"[calib] Figura salva: {pca_png}")

    # ---------- salvar artefatos ----------
    cmean_p = out_prefix + "_best_c_mean.pkl"
    pca_p   = out_prefix + "_dim_red.pkl"
    clf_p   = out_prefix + "_classifier.pkl"
    meta_p  = out_prefix + "_meta.json"
    ch_p    = out_prefix + "_channels.txt"

    with open(cmean_p, "wb") as f:
        pickle.dump(best_cmean, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pca_p, "wb") as f:
        pickle.dump(best_pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(clf_p, "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        with open(ch_p, "w", encoding="utf-8") as f:
            f.write("\n".join(ch_sel))
    except Exception:
        pass

    meta = {
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "data_dir": data_dir,
        "markers_file": os.path.basename(markers_csv),
        "signal_file": os.path.basename(signal_csv),
        "fs_hz": FS_HZ,
        "filter": {"type": "butter_bandpass", "order": BP_ORDER, "band_hz": BP_BAND},
        "epoch_s": EPOCH_S,
        "trial_offset_s": TRIAL_OFFSET_S,
        "pca_dim": int(getattr(best_pca, "n_components_", PCA_DIM) or PCA_DIM),
        "svc": {"C": SVC_C, "kernel": "linear", "probability": True, "random_state": RNG_SEED},
        "classes_map": {"LEFT_MI_STIM": 0, "RIGHT_MI_STIM": 1},
        "cv": {
            "splits": CV_SPLITS,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1) if len(accs)>1 else 0.0)
        },
        "channels_selected": ch_sel,
        "select_by": "index",
        "index_base": 1,
        "rng_seed": RNG_SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "scipy": getattr(signal, "__version__", None),
            "sklearn": sklearn.__version__,
            "pyriemann": pyriemann.__version__,
        }
    }
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[calib] Artefatos salvos:")
    print(" ", cmean_p)
    print(" ", pca_p)
    print(" ", clf_p)
    print(" ", meta_p)
    print(" ", ch_p)

    # >>> retorno dos dados para uso posterior <<<
    return {
        "Xw": Xw,
        "y": y,
        "trial_id": trial_id,
        "t_rel": t_rel,
        "accs_cv": accs,
        "Xp": Xp,
        "meta": meta,
        "markers_csv": markers_csv,
        "signal_csv": signal_csv,
        "cmean_path": cmean_p,
        "pca_path": pca_p,
        "clf_path": clf_p,
    }
