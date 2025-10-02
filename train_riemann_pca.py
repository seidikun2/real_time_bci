# -*- coding: utf-8 -*-
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.tangentspace import tangent_space

# ======================= CONFIG =======================
DATA_DIR       = r"C:\Users\User\Desktop\Dados"
MARKERS_FILE   = None  # None -> usa o mais recente 'graz_*markers_*.csv'
SIGNAL_FILE    = None  # None -> usa o mais recente 'graz_*signal_*.csv'

FS_HZ          = 500.0
BP_ORDER       = 4
BP_BAND        = (5.0, 50.0)      # Hz

EPOCH_S        = 2.5              # duração da época por tentativa
TRIAL_OFFSET_S = 0.5              # deslocamento após ATTEMPT

PCA_DIM        = 2                # fixo
SVC_C          = 1.0              # fixo
RNG_SEED       = 42
CV_SPLITS      = 5
# ======================================================

def _find_latest(folder: str, pattern: str) -> str:
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo em {folder} com padrão {pattern}")
    return max(files, key=os.path.getmtime)

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
        raise KeyError("Marcadores precisam de 'label' ou 'code'.")

    ok = np.isfinite(t)
    return t[ok], [labels[i] for i in np.where(ok)[0]]

def read_signal_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path, low_memory=False)
    if "lsl_time_s" not in df.columns:
        raise KeyError("Sinal precisa ter a coluna 'lsl_time_s'.")
    t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)

    ch_cols = [c for c in df.columns if re.match(r"(?i)^ch\d+$", c)]
    if not ch_cols:
        cols = df.columns.tolist()
        start = 0
        for meta in ("iso_time","lsl_time_s","local_recv_s"):
            if meta in df.columns: start += 1
        ch_cols = cols[start:] if len(cols) > start else []
    if not ch_cols:
        raise KeyError("Não encontrei colunas de canais (ex.: ch1..chN).")

    X = df[ch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    return t[ok], X[ok, :], ch_cols

def bandpass(data: np.ndarray, fs: float, order: int, band: Tuple[float, float]) -> np.ndarray:
    low, high = band
    if not (0 < low < high < fs/2):
        raise ValueError(f"Banda inválida {band} para Fs={fs:.2f} Hz.")
    b, a = signal.butter(order, band, btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, data.T).T

def nearest_index(t: np.ndarray, x: float) -> int:
    i = np.searchsorted(t, x)
    if i <= 0: return 0
    if i >= len(t): return len(t)-1
    return i-1 if abs(t[i-1]-x) <= abs(t[i]-x) else i

def attempts_by_class(t_mark: np.ndarray, labels: List[str]) -> Dict[str, List[float]]:
    out = {"LEFT_MI_STIM": [], "RIGHT_MI_STIM": []}
    last = None
    for ts, lab in zip(t_mark, labels):
        if lab in out: last = lab
        elif lab == "ATTEMPT" and last in out:
            out[last].append(ts)
    return out

def epoch_trials_simple(t_sig: np.ndarray, X: np.ndarray, attempts: Dict[str, List[float]],
                        label_map: Dict[str, int],fs: float, epoch_s: float, offset_s: float,  seed: int = 42):
    """Uma época fixa por tentativa (C x T)."""
    n = int(round(epoch_s * fs))
    shift = int(round(offset_s * fs))

    epochs, y, trial_id, t_rel = [], [], [], []
    cur = 0
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
        raise ValueError("Nenhuma época válida gerada — ajuste EPOCH_S / TRIAL_OFFSET_S.")
    Xw = np.stack(epochs, axis=0)               # (N,C,T)
    y = np.asarray(y, int)
    trial_id = np.asarray(trial_id, int)
    t_rel = np.asarray(t_rel, float)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return Xw[perm], y[perm], trial_id[perm], t_rel[perm]

def pick_cmean_pca_via_cv(X: np.ndarray, y: np.ndarray, trial_id: np.ndarray,
                          pca_dim: int, svc_c: float, n_rep: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    uniq_trials = np.unique(trial_id)
    if len(uniq_trials) < 2:
        raise ValueError("Trials insuficientes (>=2).")

    best_acc, best_c, best_pca = -np.inf, None, None
    val_size = min(4, max(1, len(uniq_trials)//5), len(uniq_trials)-1)

    for _ in range(n_rep):
        va_trials = rng.choice(uniq_trials, size=val_size, replace=False)
        tr_mask = ~np.isin(trial_id, va_trials)
        va_mask = ~tr_mask
        if not np.any(tr_mask) or not np.any(va_mask): continue

        cov_tr = Covariances("oas").transform(X[tr_mask])
        cov_va = Covariances("oas").transform(X[va_mask])
        cmean  = mean_covariance(cov_tr)

        ts_tr  = tangent_space(cov_tr, cmean)
        ts_va  = tangent_space(cov_va, cmean)

        dim    = max(1, min(pca_dim, ts_tr.shape[1]))
        pca    = PCA(n_components=dim).fit(ts_tr)
        Xtr, Xva = pca.transform(ts_tr), pca.transform(ts_va)

        clf = SVC(kernel="linear", C=svc_c)
        clf.fit(Xtr, y[tr_mask])
        acc = clf.score(Xva, y[va_mask])

        if acc > best_acc:
            best_acc, best_c, best_pca = acc, cmean, pca

    if best_c is None:  # fallback
        cov = Covariances("oas").transform(X)
        best_c = mean_covariance(cov)
        ts_all = tangent_space(cov, best_c)
        dim    = max(1, min(pca_dim, ts_all.shape[1]))
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
    """Scatter PCA 2D (ou 1D -> completa com zeros) + fronteira linear."""
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

def main():
    # arquivos
    markers_csv = MARKERS_FILE or _find_latest(DATA_DIR, "graz_*markers_*.csv")
    signal_csv  = SIGNAL_FILE  or _find_latest(DATA_DIR, "graz_*signal_*.csv")
    print(f"[train] Marcadores: {markers_csv}")
    print(f"[train] Sinal     : {signal_csv}")

    # leitura
    t_mark, labels = read_markers_csv(markers_csv)
    t_sig, X_all, ch_all = read_signal_csv(signal_csv)
    print(f"[train] Fs fixo={FS_HZ:.1f} Hz | samples={X_all.shape[0]} | canais={len(ch_all)}")

    # filtro
    Xf = bandpass(X_all, FS_HZ, BP_ORDER, BP_BAND)
    print(f"[train] Sinal filtrado: {Xf.shape} (samples x canais)")

    # trials por tentativa
    atts = attempts_by_class(t_mark, labels)
    print(f"[train] ATTEMPTs: LEFT={len(atts['LEFT_MI_STIM'])}, RIGHT={len(atts['RIGHT_MI_STIM'])}")

    label_map = {"LEFT_MI_STIM": 0, "RIGHT_MI_STIM": 1}
    Xw, y, trial_id, t_rel = epoch_trials_simple(
        t_sig, Xf, atts, label_map, FS_HZ, EPOCH_S, TRIAL_OFFSET_S, RNG_SEED
    )
    print(f"[train] Épocas: N={Xw.shape[0]} | C={Xw.shape[1]} | T={Xw.shape[2]} | trials={len(np.unique(trial_id))}")

    # CV externa (informativa)
    accs = cv_trialwise_accuracies(Xw, y, trial_id, PCA_DIM, SVC_C, CV_SPLITS, RNG_SEED)
    print(f"[train] Acurácias CV: {[f'{a:.3f}' for a in accs]}")
    print(f"[train] Média={np.mean(accs):.3f} | DP={np.std(accs, ddof=1) if len(accs)>1 else 0.0:.3f}")

    # Seleção (C_mean, PCA) via validação interna repetida
    best_cmean, best_pca, best_acc = pick_cmean_pca_via_cv(
        Xw, y, trial_id, pca_dim=PCA_DIM, svc_c=SVC_C, n_rep=20, seed=RNG_SEED
    )
    print(f"[train] melhor acurácia validação interna ~ {best_acc:.3f}")

    # Treino final
    cov = Covariances("oas").transform(Xw)
    ts  = tangent_space(cov, best_cmean)
    Xp  = best_pca.transform(ts)

    clf = SVC(kernel="linear", C=SVC_C, probability=True, random_state=RNG_SEED)
    clf.fit(Xp, y)

    # caminhos de saída
    base = os.path.splitext(os.path.basename(signal_csv))[0]
    out_prefix = os.path.join(os.path.dirname(signal_csv), base)
    pca_png   = out_prefix + "_pca.png"

    # salvar figura PCA (sempre)
    plot_pca_scatter(Xp, y, clf, pca_png)
    print(f"[train] Figura salva: {pca_png}")

if __name__ == "__main__":
    main()