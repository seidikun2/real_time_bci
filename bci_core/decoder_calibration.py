# -*- coding: utf-8 -*-
"""
decoder_calibration.py

Treino Riemann + PCA + SVM com a mesma lógica temporal do online:
- o período ATTEMPT do treino dura trial_duration_s;
- dentro de cada ATTEMPT são extraídas janelas deslizantes de model.epoch_s;
- o online usa a mesma janela model.epoch_s continuamente, sem conceito de trial;
- o filtro é aplicado como filtro causal contínuo no treino, para ficar condizente com tempo-real.
"""

import os, glob, re, sys, json, platform, pickle, shutil
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.tangentspace import tangent_space
import sklearn
import pyriemann

from .config_models import AppConfig
from .representation_feedback import prepare_training_representation
from .class_schema import active_motor_labels, label_map_for, display_name, class_details


def _raw(cfg: AppConfig) -> dict:
    return getattr(cfg, "_raw_config", {}) or {}


def _raw_get(cfg: AppConfig, path: list[str], default=None):
    obj = _raw(cfg)
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def get_trial_duration_s(cfg: AppConfig) -> float:
    val = _raw_get(cfg, ["protocol", "trial_duration_s"], None)
    if val is not None:
        return float(val)
    # compatibilidade: no seu check_data, tmax já representa o fim do ATTEMPT usado para QC
    return float(getattr(cfg.check_data, "tmax", 3.75))


# ---------- arquivos ----------
def _split_marker_name(fname: str) -> tuple[str, str] | None:
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

        # Compatibilidade com arquivos antigos sem run_id compartilhado.
        # Só é usada se não houver pares exatos no diretório.
        if "_markers_" not in base_m:
            continue
        prefix, _ = base_m.split("_markers_", 1)
        sig_files = glob.glob(os.path.join(os.path.dirname(m), prefix + "_signal_*.csv"))
        if sig_files:
            s = max(sig_files, key=os.path.getmtime)
            legacy_pairs.append((m, s))

    pairs = exact_pairs if exact_pairs else legacy_pairs
    return sorted(pairs, key=lambda p: os.path.getmtime(p[1]), reverse=True)

def choose_pair(folder: str):
    pairs = find_marker_signal_pairs(folder)
    if not pairs:
        raise FileNotFoundError(f"Nenhum par *_markers_*.csv / *_signal_*.csv em {folder}.")
    print(f"\nPares de arquivos encontrados em {folder}:")
    for i, (m, s) in enumerate(pairs, start=1):
        mt = pd.to_datetime(os.path.getmtime(m), unit="s").strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{i}] {os.path.basename(m)}  |  {os.path.basename(s)}   ({mt})")
    while True:
        ans = input("Selecione o número do par [1 = mais recente]: ").strip()
        idx = 1 if ans == "" else int(ans) if ans.isdigit() else -1
        if 1 <= idx <= len(pairs):
            return pairs[idx-1]
        print("Número inválido.")


def resolve_pair(folder: str, mark_explicit: Optional[str] = None, sig_explicit: Optional[str] = None):
    if mark_explicit and sig_explicit:
        m = mark_explicit if os.path.isabs(mark_explicit) else os.path.join(folder, mark_explicit)
        s = sig_explicit  if os.path.isabs(sig_explicit)  else os.path.join(folder, sig_explicit)
        if not os.path.exists(m):
            raise FileNotFoundError(f"Arquivo de marcadores não existe: {m}")
        if not os.path.exists(s):
            raise FileNotFoundError(f"Arquivo de sinal não existe: {s}")
        return m, s
    return choose_pair(folder)


# ---------- leitura ----------
def read_markers_csv(path: str, code_map: Optional[Dict[int, str]] = None) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if "lsl_time_s" in df.columns:
        t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)
    elif "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
    else:
        raise KeyError("Marcadores precisam de 'lsl_time_s' ou 'time_s'.")

    if "label" in df.columns:
        labels = df["label"].astype(str).str.strip().tolist()
    elif "event" in df.columns:
        labels = df["event"].astype(str).str.strip().tolist()
    elif "code" in df.columns:
        code = pd.to_numeric(df["code"], errors="coerce").fillna(-1).astype(int).to_numpy()
        cmap = dict(code_map or {
            1: "BASELINE", 2: "ATTENTION", 3: "LEFT_MI_STIM",
            4: "RIGHT_MI_STIM", 5: "ATTEMPT", 6: "REST",
            7: "BOTH_MI_STIM", 99: "BLOCK_END",
        })
        labels = [cmap.get(int(c), "UNKNOWN") for c in code]
    else:
        raise KeyError("Marcadores precisam de 'label', 'event' ou 'code'.")

    ok = np.isfinite(t)
    return t[ok], [str(labels[i]).strip().upper() for i in np.where(ok)[0]]


def read_signal_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path, low_memory=False)
    if "lsl_time_s" not in df.columns:
        raise KeyError("Sinal precisa ter a coluna 'lsl_time_s'.")
    t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)

    ch_cols = [c for c in df.columns if re.match(r"(?i)^ch\d+$", str(c))]
    if not ch_cols:
        cols  = df.columns.tolist()
        start = 0
        for meta in ("iso_time", "lsl_time_s", "local_recv_s", "time_s", "timestamp"):
            if meta in df.columns:
                start += 1
        ch_cols = cols[start:] if len(cols) > start else []
    if not ch_cols:
        raise KeyError("Não encontrei colunas de canais (ex.: ch1..chN).")

    X  = df[ch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    return t[ok], X[ok, :], [str(c) for c in ch_cols]


def check_time_overlap(t_mark: np.ndarray, t_sig: np.ndarray, trial_duration_s: float, window_s: float) -> None:
    if len(t_mark) == 0:
        raise ValueError("Arquivo de marcadores está vazio.")
    if len(t_sig) == 0:
        raise ValueError("Arquivo de sinal está vazio.")

    marker_lo, marker_hi = float(np.nanmin(t_mark)), float(np.nanmax(t_mark))
    signal_lo, signal_hi = float(np.nanmin(t_sig)), float(np.nanmax(t_sig))

    print(f"[calib] marker range: {marker_lo:.6f}–{marker_hi:.6f}")
    print(f"[calib] signal range : {signal_lo:.6f}–{signal_hi:.6f}")

    overlap_start = max(marker_lo, signal_lo)
    overlap_end   = min(marker_hi + float(trial_duration_s), signal_hi)
    overlap_s     = overlap_end - overlap_start

    if overlap_s < float(window_s):
        raise ValueError(
            "Marcadores e sinal não têm sobreposição temporal suficiente para janelar. "
            f"Sobreposição útil={overlap_s:.3f}s, janela={float(window_s):.3f}s. "
            "Provável pareamento errado de arquivos, stream de marcador antigo, ou PsychoPy iniciado antes do EEG/logger estar pronto."
        )

    print(f"[calib] sobreposição temporal útil: {overlap_s:.3f}s")


# ---------- processamento ----------
def bandpass_causal(data: np.ndarray, fs: float, order: int, band: Tuple[float, float]) -> np.ndarray:
    low, high = float(band[0]), float(band[1])
    if not (0 < low < high < fs / 2):
        raise ValueError(f"Banda inválida {band} para Fs={fs:.2f} Hz.")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, data, axis=0)


def nearest_index(t: np.ndarray, x: float) -> int:
    i = np.searchsorted(t, x)
    if i <= 0:
        return 0
    if i >= len(t):
        return len(t) - 1
    return i - 1 if abs(t[i-1] - x) <= abs(t[i] - x) else i


def attempts_by_class(t_mark: np.ndarray, labels: List[str], motor_labels: List[str]) -> Dict[str, List[float]]:
    out = {label: [] for label in motor_labels}
    last = None
    for ts, lab in zip(t_mark, labels):
        lab = str(lab).strip().upper()
        if lab in out:
            last = lab
        elif lab == "ATTEMPT" and last in out:
            out[last].append(float(ts))
    return out


def sliding_window_indices(start: int, stop: int, win_n: int, hop_n: int) -> List[np.ndarray]:
    out = []
    i0  = int(start)
    while i0 + win_n <= stop:
        out.append(np.arange(i0, i0 + win_n, dtype=int))
        i0 += hop_n
    return out


def epoch_trials_sliding(t_sig: np.ndarray, X: np.ndarray, attempts: Dict[str, List[float]], label_map: Dict[str, int], fs: float, window_s: float, step_s: float, trial_duration_s: float, trial_offset_s: float, seed: int = 42):
    """
    Treino: várias janelas de 1 s dentro de cada tentativa de 3.75 s.
    O trial_id é mantido igual para todas as janelas da mesma tentativa, evitando vazamento na CV.
    """
    win_n = int(round(window_s * fs))
    hop_n = max(1, int(round(step_s * fs)))
    off_n = int(round(trial_offset_s * fs))
    dur_n = int(round(trial_duration_s * fs))

    epochs, y, trial_id, t_center = [], [], [], []
    cur_trial = 0
    empty_trials = 0

    for cls, times in attempts.items():
        for t0 in times:
            i0 = nearest_index(t_sig, t0) + off_n
            i1 = nearest_index(t_sig, t0) + off_n + dur_n
            i0 = max(0, i0)
            i1 = min(len(t_sig), i1)
            idxs = sliding_window_indices(i0, i1, win_n, hop_n)
            if not idxs:
                empty_trials += 1
                continue
            for idx in idxs:
                epochs.append(X[idx, :].T)  # (C,T)
                y.append(label_map[cls])
                trial_id.append(cur_trial)
                t_center.append(float(t_sig[idx[len(idx)//2]] - t0))
            cur_trial += 1

    if empty_trials:
        print(f"[calib] {empty_trials} tentativa(s) sem janelas válidas.")
    if not epochs:
        raise ValueError("Nenhuma janela válida — confira ATTEMPT, trial_duration_s, epoch_s e step_s.")

    Xw       = np.stack(epochs, axis=0)
    y        = np.asarray(y, int)
    trial_id = np.asarray(trial_id, int)
    t_center = np.asarray(t_center, float)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return Xw[perm], y[perm], trial_id[perm], t_center[perm]


def select_channel_indices(select_by: str, selection: List, ch_names: List[str], index_base: int = 1) -> List[int]:
    if selection is None or len(selection) == 0:
        return list(range(len(ch_names)))
    if str(select_by).lower() == "name":
        name_to_idx = {str(nm): i for i, nm in enumerate(ch_names)}
        missing = [nm for nm in selection if str(nm) not in name_to_idx]
        if missing:
            raise ValueError(f"Canais não encontrados: {missing}")
        return [name_to_idx[str(nm)] for nm in selection]
    idx = [int(v) - (1 if index_base == 1 else 0) for v in selection]
    if any(i < 0 or i >= len(ch_names) for i in idx):
        raise ValueError(f"Índice de canal fora do range após base={index_base}: {selection}")
    return idx


def safe_cv_splits(y: np.ndarray, trial_id: np.ndarray, requested: int) -> int:
    # Conta trials únicos por classe; StratifiedGroupKFold exige ao menos n_splits grupos por classe.
    counts = []
    for cls in np.unique(y):
        counts.append(len(np.unique(trial_id[y == cls])))
    return max(2, min(int(requested), min(counts))) if counts and min(counts) >= 2 else 2


def cv_trialwise_accuracies(Xw: np.ndarray, y: np.ndarray, trial_id: np.ndarray, pca_dim: int, svc_c: float, n_splits: int, seed: int = 42):
    n_splits = safe_cv_splits(y, trial_id, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr_idx, va_idx in splitter.split(Xw, y, groups=trial_id):
        cov_tr = Covariances("oas").transform(Xw[tr_idx])
        cmean  = mean_covariance(cov_tr)
        ts_tr  = tangent_space(cov_tr, cmean)
        ts_va  = tangent_space(Covariances("oas").transform(Xw[va_idx]), cmean)
        dim    = max(1, min(int(pca_dim), ts_tr.shape[1]))
        pca    = PCA(n_components=dim).fit(ts_tr)
        clf    = SVC(kernel="linear", C=svc_c)
        clf.fit(pca.transform(ts_tr), y[tr_idx])
        accs.append(float(clf.score(pca.transform(ts_va), y[va_idx])))
    return accs


def fit_final_model(Xw: np.ndarray, y: np.ndarray, pca_dim: int, svc_c: float, seed: int = 42):
    cov   = Covariances("oas").transform(Xw)
    cmean = mean_covariance(cov)
    ts    = tangent_space(cov, cmean)
    dim   = max(1, min(int(pca_dim), ts.shape[1]))
    pca   = PCA(n_components=dim, random_state=seed).fit(ts)
    Xp    = pca.transform(ts)
    clf   = SVC(kernel="linear", C=svc_c, probability=True, random_state=seed)
    clf.fit(Xp, y)
    return cmean, pca, clf, Xp


def padded_limits(x: np.ndarray, pad_frac: float = 0.10):
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = pad_frac * max(hi - lo, 1e-6)
    return [lo - pad, hi + pad]


def plot_pca_scatter(X_pca: np.ndarray, y: np.ndarray, clf: SVC, out_png: str, class_labels: Dict[int, str]):
    Xp = X_pca if X_pca.shape[1] > 1 else np.c_[X_pca[:, 0], np.zeros_like(X_pca[:, 0])]
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(np.unique(y)):
        m = (y == c)
        color = cmap(i % 10)
        ax.scatter(Xp[m, 0], Xp[m, 1], s=18, alpha=0.75, color=color, label=class_labels.get(int(c), f"class {int(c)}"))
        mu = Xp[m].mean(axis=0)
        ax.plot(mu[0], mu[1], marker="x", ms=10, mew=2, color=color)

    if X_pca.shape[1] >= 2:
        x0, x1 = Xp[:, 0], Xp[:, 1]
        dx = max(float(np.ptp(x0)), 1.0) * 0.10
        dy = max(float(np.ptp(x1)), 1.0) * 0.10
        gx = np.linspace(float(x0.min() - dx), float(x0.max() + dx), 180)
        gy = np.linspace(float(x1.min() - dy), float(x1.max() + dy), 180)
        xx, yy = np.meshgrid(gx, gy)
        grid2 = np.c_[xx.ravel(), yy.ravel()]
        if X_pca.shape[1] > 2:
            grid = np.zeros((grid2.shape[0], X_pca.shape[1]), dtype=float)
            grid[:, :2] = grid2
        else:
            grid = grid2
        zz = clf.predict(grid).reshape(xx.shape)
        levels = np.arange(len(np.unique(y)) + 1) - 0.5
        ax.contour(xx, yy, zz, levels=levels, linewidths=0.8, alpha=0.35)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA treino — janelas dentro do ATTEMPT")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _run_id_from_signal(signal_csv: str) -> str:
    m = re.search(r"_(\d{8}_\d{6})\.csv$", os.path.basename(signal_csv))
    if m:
        return m.group(1)
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def model_output_dir(cfg: AppConfig, signal_csv: str) -> str:
    """Artefatos técnicos ficam junto dos dados, mas recolhidos em train/_model/<run_id>."""
    train_dir = os.path.dirname(os.path.abspath(signal_csv))
    out = os.path.join(train_dir, "_model", _run_id_from_signal(signal_csv))
    os.makedirs(out, exist_ok=True)
    return out


def qc_output_dir(signal_csv: str) -> str:
    train_dir = os.path.dirname(os.path.abspath(signal_csv))
    out = os.path.join(train_dir, "_qc", _run_id_from_signal(signal_csv))
    os.makedirs(out, exist_ok=True)
    return out


def run_calibration(cfg: AppConfig, markers_file: Optional[str] = None, signal_file: Optional[str] = None):
    data_dir = os.path.join(cfg.experiment.log_root, cfg.experiment.subject_id, f"S{cfg.experiment.session_id}", cfg.experiment.session_type, "train")
    print(f"[calib] Procurando dados de treino em: {data_dir}")

    mcfg = cfg.model
    decfg = getattr(cfg, "decoder", None)

    FS_HZ            = float(mcfg.fs_hz)
    BP_ORDER         = int(getattr(mcfg, "bp_order", getattr(decfg, "filter_order", 4)))
    BP_BAND          = tuple(getattr(mcfg, "bp_band", getattr(decfg, "band_hz", [5.0, 40.0])))
    WINDOW_S         = float(getattr(mcfg, "epoch_s", getattr(decfg, "epoch_s", 1.0)))
    STEP_S           = float(getattr(decfg, "step_s", _raw_get(cfg, ["model", "step_s"], 0.05)))
    TRIAL_DURATION_S = get_trial_duration_s(cfg)
    TRIAL_OFFSET_S   = float(getattr(mcfg, "trial_offset_s", 0.0))
    PCA_DIM          = int(mcfg.pca_dim)
    SVC_C            = float(mcfg.svc_c)
    CV_SPLITS        = int(mcfg.cv_splits)
    RNG_SEED         = int(mcfg.rng_seed)
    SELECT_BY        = getattr(mcfg, "select_by", "index")
    INDEX_BASE       = int(getattr(mcfg, "index_base", 1))
    SELECT_CHANNELS  = mcfg.select_channels or []

    markers_csv, signal_csv = resolve_pair(data_dir, markers_file, signal_file)
    print(f"[calib] Marcadores: {markers_csv}")
    print(f"[calib] Sinal     : {signal_csv}")
    print(f"[calib] Fs={FS_HZ:.1f} | banda={BP_BAND} | janela={WINDOW_S:.2f}s | step={STEP_S:.3f}s | trial={TRIAL_DURATION_S:.2f}s")

    t_mark, labels       = read_markers_csv(markers_csv, cfg.codes.code_map)
    t_sig, X_all, ch_all = read_signal_csv(signal_csv)

    check_time_overlap(t_mark, t_sig, TRIAL_DURATION_S, WINDOW_S)

    sel_idx = select_channel_indices(SELECT_BY, SELECT_CHANNELS, ch_all, index_base=INDEX_BASE)
    ch_sel  = [ch_all[i] for i in sel_idx]
    X_sel   = X_all[:, sel_idx]
    print(f"[calib] Canais: {SELECT_CHANNELS if SELECT_CHANNELS else 'todos'} -> {ch_sel}")

    Xf = bandpass_causal(X_sel, FS_HZ, BP_ORDER, BP_BAND)
    print(f"[calib] Sinal filtrado causal: {Xf.shape}")

    motor_labels = active_motor_labels(labels, require_at_least=2)
    label_map = label_map_for(motor_labels)
    attempts = attempts_by_class(t_mark, labels, motor_labels)
    counts_txt = ", ".join(f"{display_name(k)}={len(attempts[k])}" for k in motor_labels)
    print(f"[calib] Classes ativas (derivadas dos marcadores): {motor_labels}")
    print(f"[calib] ATTEMPTs: {counts_txt}")

    Xw, y, trial_id, t_center = epoch_trials_sliding(t_sig, Xf, attempts, label_map, FS_HZ, WINDOW_S, STEP_S, TRIAL_DURATION_S, TRIAL_OFFSET_S, RNG_SEED)
    print(f"[calib] Janelas: N={Xw.shape[0]} | C={Xw.shape[1]} | T={Xw.shape[2]} | trials={len(np.unique(trial_id))}")

    accs = cv_trialwise_accuracies(Xw, y, trial_id, PCA_DIM, SVC_C, CV_SPLITS, RNG_SEED)
    print(f"[calib] Acurácias CV: {[f'{a:.3f}' for a in accs]}")
    print(f"[calib] Média={np.mean(accs):.3f} | DP={np.std(accs, ddof=1) if len(accs)>1 else 0.0:.3f}")

    cmean, pca, clf, Xp = fit_final_model(Xw, y, PCA_DIM, SVC_C, RNG_SEED)

    model_dir = model_output_dir(cfg, signal_csv)
    qc_dir = qc_output_dir(signal_csv)
    run_id = _run_id_from_signal(signal_csv)
    details = class_details(label_map, cfg.codes.code_map)
    class_names = {int(row["model_class"]): str(row["display_label"]) for row in details}

    # Espaço FINAL usado no feedback visual.
    # Hoje é identity sobre o PCA; futuramente o alinhamento entra na camada
    # representation_feedback.py, sem exigir mudanças no Unity.
    rep_result = prepare_training_representation(
        Xp,
        y,
        cfg,
        qc_dir,
        class_labels=class_names,
        class_details=details,
    )
    rep_meta = rep_result["meta"]

    # JSON + PNG são artefatos de uso direto no Unity e ficam juntos na raiz
    # da pasta train/. O _qc/ guarda apenas figuras de diagnóstico.
    visible_pca_map_json = os.path.join(data_dir, f"pca_map_{run_id}.json")
    visible_pca_map_png  = os.path.join(data_dir, f"pca_map_{run_id}.png")
    shutil.copy2(rep_result["map_png"], visible_pca_map_png)

    # O JSON deve referenciar exatamente o PNG que está ao lado dele.
    with open(rep_result["manifest_json"], "r", encoding="utf-8") as f:
        visible_manifest = json.load(f)
    visible_manifest["map_image"] = os.path.basename(visible_pca_map_png)
    with open(visible_pca_map_json, "w", encoding="utf-8") as f:
        json.dump(visible_manifest, f, ensure_ascii=False, indent=2)

    rep_meta["map_image"] = os.path.basename(visible_pca_map_png)
    rep_meta["manifest_file"] = os.path.basename(visible_pca_map_json)

    # Remove as cópias transitórias do mapa de _qc para que essa pasta contenha
    # somente figuras diagnósticas.
    for tmp_path in (rep_result["manifest_json"], rep_result["map_png"]):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    print(f"[calib] PCA map PNG (Unity) : {visible_pca_map_png}")
    print(f"[calib] PCA map JSON (Unity): {visible_pca_map_json}")
    print(
        "[calib] Espaço feedback      : "
        f"id={rep_meta['representation_id']} | "
        f"nome={rep_meta['representation_name']} | "
        f"dim={rep_meta['dimensions']} | "
        f"transform={rep_meta['transform_type']}"
    )

    # Mantém também a figura diagnóstica PCA original.
    pca_png = os.path.join(qc_dir, "pca_diagnostic.png")
    plot_pca_scatter(Xp, y, clf, pca_png, class_names)
    print(f"[calib] Figura PCA salva: {pca_png}")

    paths = {
        "cmean": os.path.join(model_dir, "riemann_mean.pkl"),
        "pca": os.path.join(model_dir, "pca.pkl"),
        "clf": os.path.join(model_dir, "classifier.pkl"),
        "meta": os.path.join(model_dir, "model_meta.json"),
        "ch": os.path.join(model_dir, "channels.txt"),
        "pca_map_png": visible_pca_map_png,
        "pca_map_json": visible_pca_map_json,
        "pca_diagnostic": pca_png,
    }
    with open(paths["cmean"], "wb") as f: pickle.dump(cmean, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["pca"], "wb") as f: pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["clf"], "wb") as f: pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["ch"], "w", encoding="utf-8") as f: f.write("\n".join(ch_sel))

    Xp2 = Xp if Xp.shape[1] > 1 else np.c_[Xp[:, 0], np.zeros_like(Xp[:, 0])]
    meta = {
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "data_dir": data_dir,
        "markers_file": os.path.basename(markers_csv),
        "signal_file": os.path.basename(signal_csv),
        "fs_hz": FS_HZ,
        "filter": {"type": "butter_bandpass_causal_sosfilt", "order": BP_ORDER, "band_hz": list(BP_BAND)},
        "window_s": WINDOW_S,
        "epoch_s": WINDOW_S,
        "step_s": STEP_S,
        "trial_duration_s": TRIAL_DURATION_S,
        "trial_offset_s": TRIAL_OFFSET_S,
        "pca_dim": int(getattr(pca, "n_components_", PCA_DIM) or PCA_DIM),

        # Nomes mantidos por compatibilidade com plot_decoder_realtime.py.
        # Estes limites agora correspondem à representação FINAL publicada
        # nos canais 0/1 do stream Signal/BCI.
        "pca_train_xlim": [rep_meta["x_min"], rep_meta["x_max"]],
        "pca_train_ylim": [rep_meta["y_min"], rep_meta["y_max"]],

        # Contrato genérico usado pelo Unity e pelo online.
        "representation": rep_meta,
        "svc": {"C": SVC_C, "kernel": "linear", "probability": True, "random_state": RNG_SEED},
        "classes_map": {str(k): int(v) for k, v in label_map.items()},
        "classes": details,
        "model_dir": model_dir,
        "run_id": run_id,
        "pca_map_json": visible_pca_map_json,
        "pca_map_png": visible_pca_map_png,
        "cv": {"splits": len(accs), "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1) if len(accs)>1 else 0.0), "accs": [float(a) for a in accs]},
        "channels_selected": ch_sel,
        "select_by": SELECT_BY,
        "index_base": INDEX_BASE,
        "rng_seed": RNG_SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "pyriemann": pyriemann.__version__},
    }
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[calib] Artefatos salvos:")
    for p in paths.values():
        print(" ", p)

    return {
        "Xw": Xw,
        "y": y,
        "trial_id": trial_id,
        "t_rel": t_center,
        "accs_cv": accs,
        "Xp": Xp,
        "X_representation": rep_result["X_rep"],
        "representation_meta": rep_meta,
        "representation_map_path": visible_pca_map_png,
        "representation_manifest_path": visible_pca_map_json,
        "meta": meta,
        "markers_csv": markers_csv,
        "signal_csv": signal_csv,
        "cmean_path": paths["cmean"],
        "pca_path": paths["pca"],
        "clf_path": paths["clf"],
        "model_dir": model_dir,
    }
