# train_graz_riemann_from_logs.py
import os
import re
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.tangentspace import tangent_space
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Config & datatypes
# -----------------------------
@dataclass
class Params:
    epoch_s: float = 2.0       # janela (s)
    step_s: float = 0.1        # passo (s)
    max_trial_s: float = 3.75  # pós-ATTEMPT (s)
    bp_order: int = 4
    bp_band: Tuple[float, float] = (8.0, 30.0)
    n_cv: int = 20
    pca_dim: int = 2
    svc_c: float = 1.0
    rng_seed: int = 42
    trial_offset_s: float = 0.0  # ignorar os primeiros X s depois do ATTEMPT


# -----------------------------
# Logging helper
# -----------------------------
def _log(msg: str) -> None:
    print(f"[train] {msg}")


# -----------------------------
# IO helpers
# -----------------------------
def find_latest(folder: str, pattern: str) -> str:
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em: {os.path.join(folder, pattern)}")
    return max(files, key=os.path.getmtime)


def read_markers_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """CSV de marcadores do simulador: usa lsl_time_s e label (ou code)."""
    df = pd.read_csv(path)
    if "lsl_time_s" in df.columns:
        t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(dtype=float)
    elif "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    else:
        raise KeyError("CSV de marcadores precisa de 'lsl_time_s' (ou 'time_s').")

    if "label" in df.columns:
        labels = df["label"].astype(str).str.strip().tolist()
    elif "event" in df.columns:
        labels = df["event"].astype(str).str.strip().tolist()
    elif "code" in df.columns:
        code = pd.to_numeric(df["code"], errors="coerce").fillna(-1).astype(int).to_numpy()
        code_map = {1:"BASELINE",2:"ATTENTION",3:"LEFT_MI_STIM",4:"RIGHT_MI_STIM",5:"ATTEMPT",6:"REST"}
        labels = [code_map.get(int(c), "UNKNOWN") for c in code]
    else:
        raise KeyError("CSV de marcadores precisa de 'label' ou 'code'.")

    ok = np.isfinite(t)
    return t[ok], [labels[i] for i in np.where(ok)[0]]


def read_signal_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    CSV do sinal do simulador:
    header típico: ['iso_time','lsl_time_s','ch1','ch2',...]
    """
    df = pd.read_csv(path, low_memory=False)
    if "lsl_time_s" not in df.columns:
        raise KeyError("CSV do sinal precisa ter a coluna 'lsl_time_s'.")

    t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(dtype=float)

    ch_cols = [c for c in df.columns if re.match(r"(?i)^ch\d+$", c)]
    if not ch_cols:
        maybe = df.columns.tolist()
        if len(maybe) > 2:
            ch_cols = maybe[2:]
    if not ch_cols:
        raise KeyError("Não encontrei colunas de canais (ex.: ch1, ch2, ...).")

    X = df[ch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    return t[ok], X[ok, :], ch_cols


def estimate_fs(t: np.ndarray) -> float:
    d = np.diff(t)
    d = d[(d > 0) & np.isfinite(d)]
    return float(1.0 / np.median(d)) if d.size else 0.0


# -----------------------------
# Signal helpers
# -----------------------------
def bandpass(data: np.ndarray, fs: float, order: int, band: Tuple[float, float]) -> np.ndarray:
    b, a = signal.butter(order, band, btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, data.T).T


def sliding_windows(length: int, fs: float, epoch_s: float, step_s: float,
                    start: int = 0, stop: Optional[int] = None) -> List[np.ndarray]:
    n = int(round(epoch_s * fs))
    hop = int(round(step_s * fs))
    start = int(start)
    stop = length if stop is None else int(stop)
    if stop - start < n:
        return []
    idx0 = np.arange(start, start + n)
    out = []
    while idx0[-1] < stop:
        out.append(idx0.copy())
        idx0 += hop
    return out


def nearest_index(t: np.ndarray, x: float) -> int:
    i = np.searchsorted(t, x)
    if i <= 0: return 0
    if i >= len(t): return len(t) - 1
    return i - 1 if abs(t[i - 1] - x) <= abs(t[i] - x) else i


# -----------------------------
# Epoching alinhado ao ATTEMPT
# -----------------------------
def attempts_by_class(t_mark: np.ndarray, labels: List[str]) -> Dict[str, List[float]]:
    """Agrupa tempos de ATTEMPT pela última pista anterior (LEFT/RIGHT)."""
    out = {"LEFT_MI_STIM": [], "RIGHT_MI_STIM": []}
    last_cue = None
    for ts, lab in zip(t_mark, labels):
        if lab == "LEFT_MI_STIM":
            last_cue = "LEFT_MI_STIM"
        elif lab == "RIGHT_MI_STIM":
            last_cue = "RIGHT_MI_STIM"
        elif lab == "ATTEMPT":
            if last_cue in out:
                out[last_cue].append(ts)
    return out


def epoch_trials_from_attempts(
    t_sig: np.ndarray,
    X: np.ndarray,
    attempts_by_cls: Dict[str, List[float]],
    label_map: Dict[str, int],
    p: Params,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera janelas deslizantes dentro de cada tentativa (alinhadas ao ATTEMPT)."""
    trial_windows: List[np.ndarray] = []
    y_list: List[int] = []
    trial_id: List[int] = []

    cur_trial = 0
    empty_trials = 0

    for cls_name, att_times in attempts_by_cls.items():
        for t0 in att_times:
            i0 = nearest_index(t_sig, t0)
            start = i0 + int(round(p.trial_offset_s * fs))
            stop  = i0 + int(round(p.max_trial_s * fs))
            idxs = sliding_windows(length=len(t_sig), fs=fs,
                                   epoch_s=p.epoch_s, step_s=p.step_s,
                                   start=start, stop=stop)
            if not idxs:
                empty_trials += 1
                continue
            for idx in idxs:
                trial_windows.append(idx)
                y_list.append(label_map[cls_name])
                trial_id.append(cur_trial)
            cur_trial += 1

    if empty_trials:
        _log(f"{empty_trials} tentativa(s) sem janelas válidas (ajuste epoch/step/max_trial/trial_offset)")

    if not trial_windows:
        raise ValueError("Nenhuma janela gerada. Verifique se há ATTEMPT e pistas LEFT/RIGHT.")

    Xw = np.stack([X[idx, :] for idx in trial_windows], axis=0)  # (N, T, C)
    Xw = np.swapaxes(Xw, 1, 2)  # (N, C, T)

    rng = np.random.default_rng(p.rng_seed)
    perm = rng.permutation(len(y_list))
    return Xw[perm], np.array(y_list)[perm], np.array(trial_id)[perm]


# -----------------------------
# Modelo (Riemann + PCA + SVM)
# -----------------------------
def pick_cmean_via_cv(X: np.ndarray, y: np.ndarray, trial_id: np.ndarray, p: Params) -> np.ndarray:
    rng = np.random.default_rng(p.rng_seed)
    uniq_trials = np.unique(trial_id)
    if len(uniq_trials) < 2:
        raise ValueError(f"Trials insuficientes: {len(uniq_trials)} (precisa >= 2)")

    best_acc, best_c = -np.inf, None
    val_size = min(4, max(1, len(uniq_trials) // 5), len(uniq_trials) - 1)

    for _ in range(p.n_cv):
        val_trials = rng.choice(uniq_trials, size=val_size, replace=False)
        tr_mask = ~np.isin(trial_id, val_trials)
        va_mask = ~tr_mask
        if not np.any(tr_mask) or not np.any(va_mask):
            continue

        cov_tr = Covariances("oas").transform(X[tr_mask])
        cov_va = Covariances("oas").transform(X[va_mask])
        cmean = mean_covariance(cov_tr)
        ts_tr = tangent_space(cov_tr, cmean)
        ts_va = tangent_space(cov_va, cmean)

        clf = SVC(kernel="linear", C=p.svc_c)
        clf.fit(ts_tr, y[tr_mask])
        acc = clf.score(ts_va, y[va_mask])
        if acc > best_acc:
            best_acc, best_c = acc, cmean

    if best_c is None:
        _log("CV não conseguiu selecionar C_mean — usando média no conjunto todo.")
        best_c = mean_covariance(Covariances("oas").transform(X))
    else:
        _log(f"melhor acurácia CV: {best_acc:.3f}")
    return best_c


# -----------------------------
# Plot PCA scatter (+ fronteira)
# -----------------------------
CLASS_COLOR = {0: "#4169e1", 1: "#dc143c"}  # LEFT, RIGHT

def plot_pca_space(X_pca: np.ndarray,
                   y: Optional[np.ndarray],
                   clf: Optional[SVC],
                   out_png: Optional[str],
                   show: bool,
                   title: str = "PCA space (tangent → PCA)") -> None:
    dim = X_pca.shape[1]
    if dim == 1:  # completar 2D para visualização
        X_pca = np.c_[X_pca[:, 0], np.zeros_like(X_pca[:, 0])]
        dim = 2

    fig, ax = plt.subplots(figsize=(7, 6))
    if y is None:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=12, alpha=0.7, edgecolor="none")
    else:
        for c in np.unique(y):
            m = (y == c)
            ax.scatter(X_pca[m, 0], X_pca[m, 1],
                       s=16, alpha=0.75, edgecolor="none",
                       label=f"class {c}", color=CLASS_COLOR.get(int(c), "k"))
            # centro por classe
            mu = X_pca[m].mean(axis=0)
            ax.plot(mu[0], mu[1], marker="x", ms=10, mew=2,
                    color=CLASS_COLOR.get(int(c), "k"))

    # Fronteira do SVM em 2D (se disponível)
    if clf is not None and hasattr(clf, "coef_") and X_pca.shape[1] >= 2:
        w = clf.coef_[0]
        b = clf.intercept_[0]
        if abs(w[1]) > 1e-9:
            x_min, x_max = X_pca[:, 0].min()-1, X_pca[:, 0].max()+1
            xs = np.linspace(x_min, x_max, 200)
            ys = -(w[0]*xs + b) / w[1]
            ax.plot(xs, ys, "k-", lw=1.2, alpha=0.8, label="SVM boundary")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    if y is not None:
        ax.legend(frameon=False, loc="best")
    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=150)
        _log(f"PCA scatter salvo em: {out_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Pipeline principal
# -----------------------------
def train_from_logs(markers_csv: str, signal_csv: str, out_prefix: Optional[str],
                    classes: str = "2classes", p: Params = Params(),
                    savefig: bool = False, noshow: bool = False) -> None:
    _log(f"lendo marcadores: {markers_csv}")
    t_mark, labels = read_markers_csv(markers_csv)

    _log(f"lendo sinal: {signal_csv}")
    t_sig, X_raw, ch_names = read_signal_csv(signal_csv)
    fs = estimate_fs(t_sig)
    if fs <= 0:
        raise RuntimeError("Não foi possível estimar Fs a partir do CSV do sinal.")
    _log(f"Fs estimada: {fs:.2f} Hz | shape(bruto)={X_raw.shape} | canais={len(ch_names)}")

    # Filtro bandpass
    X = bandpass(X_raw, fs, p.bp_order, p.bp_band)
    _log(f"EEG pós-filtro: {X.shape} (samples, channels)")

    # Alinhamento no ATTEMPT e rotulagem pela pista anterior
    attempts = attempts_by_class(t_mark, labels)
    _log(f"ATTEMPT: LEFT={len(attempts['LEFT_MI_STIM'])}, RIGHT={len(attempts['RIGHT_MI_STIM'])}")

    # Prefixo de saída
    if out_prefix is None:
        base = os.path.splitext(os.path.basename(signal_csv))[0]
        out_prefix = os.path.join(os.path.dirname(signal_csv), base)

    # ---------- FREE ----------
    if classes == "free":
        idxs = sliding_windows(len(t_sig), fs, p.epoch_s, p.step_s)
        if not idxs:
            raise ValueError("Nenhuma janela para 'free'. Ajuste epoch/step.")
        Xw = np.swapaxes(np.stack([X[i, :] for i in idxs], axis=0), 1, 2)  # (N, C, T)
        cov_all = Covariances("lwf").transform(Xw)
        cmean = mean_covariance(cov_all)
        ts_all = tangent_space(cov_all, cmean)
        dim = max(1, min(p.pca_dim, ts_all.shape[1]))
        pca = PCA(n_components=dim).fit(ts_all)
        X_pca = pca.transform(ts_all)
        clf = None
        kde_dict = {}
        # plot PCA (sem rótulos)
        out_png = out_prefix + "_pca_scatter.png" if savefig else None
        plot_pca_space(X_pca, y=None, clf=None, out_png=out_png,
                       show=not noshow, title="PCA (free)")

        # extents p/ UI
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        if dim >= 2:
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        else:
            y_min, y_max = 0.0, 1.0

    # ---------- 2 CLASSES ----------
    elif classes == "2classes":
        label_map = {"LEFT_MI_STIM": 0, "RIGHT_MI_STIM": 1}
        Xw, y, trial_id = epoch_trials_from_attempts(t_sig, X, attempts, label_map, p, fs)
        _log(f"#janelas: {Xw.shape[0]} | C={Xw.shape[1]} | T={Xw.shape[2]} | #trials={len(np.unique(trial_id))}")
        if Xw.shape[0] < 2:
            raise ValueError("Amostras insuficientes para treino.")

        cmean = pick_cmean_via_cv(Xw, y, trial_id, p)
        cov = Covariances("oas").transform(Xw)
        ts = tangent_space(cov, cmean)
        dim = max(1, min(p.pca_dim, ts.shape[1]))
        pca = PCA(n_components=dim).fit(ts)
        X_pca = pca.transform(ts)

        clf = SVC(kernel="linear", C=p.svc_c)
        clf.fit(X_pca, y)

        # plot PCA com classes e fronteira
        out_png = out_prefix + "_pca_scatter.png" if savefig else None
        plot_pca_space(X_pca, y=y, clf=clf, out_png=out_png,
                       show=not noshow, title="PCA (LEFT vs RIGHT)")

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        if dim >= 2:
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        else:
            y_min, y_max = 0.0, 1.0

        # KDE opcional (mantido do pipeline original; pode ser útil depois)
        kde_dict: Dict[str, gaussian_kde] = {}
        for c in np.unique(y):
            cls = X_pca[y == c]
            if cls.shape[0] >= 3 and dim >= 2:
                try:
                    kde_dict[f"kde{c}"] = gaussian_kde(cls[:, :2].T)
                except Exception as e:
                    _log(f"KDE falhou p/ classe {c}: {e}")
    else:
        raise ValueError("classes deve ser '2classes' ou 'free'.")

    # Persist artifacts
    with open(out_prefix + "_range_pca.pkl", "wb") as f:
        pickle.dump((x_min, x_max, y_min, y_max), f)
    with open(out_prefix + "_best_c_mean.pkl", "wb") as f:
        pickle.dump(cmean, f)
    with open(out_prefix + "_dim_red.pkl", "wb") as f:
        pickle.dump(pca, f)
    if classes == "2classes" and clf is not None:
        with open(out_prefix + "_classifier.pkl", "wb") as f:
            pickle.dump(clf, f)
    if classes == "2classes" and 'kde_dict' in locals() and kde_dict:
        with open(out_prefix + "_kde.pkl", "wb") as f:
            pickle.dump(kde_dict, f)

    _log("OK — artefatos salvos.")
    _log(f"prefixo: {out_prefix}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Treino Riemann/PCA/SVM a partir de logs (sinal+marcadores) do Graz, com plot do espaço PCA.")
    ap.add_argument("--dir", "-d", default=r"C:\Users\User\Desktop\Dados", help="Pasta com os CSVs.")
    ap.add_argument("--markers", "-m", default=None, help="CSV de marcadores (graz_*markers_*.csv). Se None, pega o mais recente.")
    ap.add_argument("--signal", "-s", default=None, help="CSV de sinal (graz_*signal_*.csv). Se None, pega o mais recente.")
    ap.add_argument("--classes", choices=["2classes", "free"], default="2classes", help="Treino supervisionado (2classes) ou só projeção (free).")
    ap.add_argument("--epoch", type=float, default=2.0, help="Tamanho da janela (s).")
    ap.add_argument("--step", type=float, default=0.1, help="Passo entre janelas (s).")
    ap.add_argument("--trialmax", type=float, default=3.75, help="Janela máxima pós-ATTEMPT (s).")
    ap.add_argument("--trialoffset", type=float, default=0.0, help="Offset dentro do trial (s) após ATTEMPT.")
    ap.add_argument("--band", type=float, nargs=2, default=(8.0, 30.0), help="Faixa do bandpass (low high).")
    ap.add_argument("--order", type=int, default=4, help="Ordem do Butter bandpass.")
    ap.add_argument("--pca", type=int, default=2, help="Dimensão da PCA no espaço tangente.")
    ap.add_argument("--C", type=float, default=1.0, help="Parâmetro C do SVM linear.")
    ap.add_argument("--cv", type=int, default=20, help="# repetições da CV para escolher C_mean.")
    ap.add_argument("--seed", type=int, default=42, help="Seed do RNG.")
    ap.add_argument("--out_prefix", default=None, help="Prefixo manual para os .pkl.")
    ap.add_argument("--savefig", action="store_true", help="Salvar PNG do PCA scatter.")
    ap.add_argument("--noshow", action="store_true", help="Não mostrar a figura (útil em servidores/headless).")
    args = ap.parse_args()

    markers_csv = args.markers or find_latest(args.dir, "graz_*markers_*.csv")
    signal_csv  = args.signal  or find_latest(args.dir, "graz_*signal_*.csv")

    p = Params(
        epoch_s=args.epoch,
        step_s=args.step,
        max_trial_s=args.trialmax,
        bp_order=args.order,
        bp_band=tuple(args.band),
        n_cv=args.cv,
        pca_dim=args.pca,
        svc_c=args.C,
        rng_seed=args.seed,
        trial_offset_s=args.trialoffset,
    )

    train_from_logs(markers_csv, signal_csv, args.out_prefix,
                    classes=args.classes, p=p,
                    savefig=args.savefig, noshow=args.noshow)


if __name__ == "__main__":
    main()
