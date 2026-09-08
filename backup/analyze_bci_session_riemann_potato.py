#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_bci_session_riemann_potato.py

Análise offline de uma pasta de sessão BCI Graz MI.

Objetivo
--------
1) Encontrar pares *_markers_<run_id>.csv + *_signal_<run_id>.csv em:
   <session_folder>/EM_treino/train
   <session_folder>/IM_treino/train
   <session_folder>/IM_online/realtime

2) Para execução motora e imagética:
   - extrair janelas dentro de ATTEMPT;
   - calcular covariâncias OAS;
   - detectar outliers por Potato Riemanniana/critério robusto em distância Riemanniana;
   - treinar/classificar com Tangent Space + PCA + SVM RBF;
   - fazer validação cruzada por bloco e por sessão;
   - gerar curvas de probabilidade por tentativa e médias por classe;
   - gerar PCA limpo com superfície do classificador.

3) Para online:
   - usar um modelo específico de treino MI, passado por --mi-model-run-id ou --mi-model-signal;
   - se não for passado, selecionar automaticamente o bloco MI com maior CV média;
   - aplicar o mesmo template Riemann/PCA/SVM RBF;
   - usar Potato global do treino para remover janelas online outliers;
   - gerar timeline de decodificação, médias por trial e PCA online alinhado ao PCA template.

Uso típico
----------
python analyze_bci_session_riemann_potato.py \
  --session-folder "C:/Users/Unifesp/Desktop/Dados Seidi/SY001/S3" \
  --config "config.yaml" \
  --mi-model-run-id 20260622_153353

ou

python analyze_bci_session_riemann_potato.py \
  --session-folder "C:/Users/Unifesp/Desktop/Dados Seidi/SY001/S3" \
  --config "config.yaml" \
  --mi-model-signal "C:/.../IM_treino/train/SY001_TEST_S3_IM_treino_train_signal_20260622_153353.csv"

Dependências
------------
numpy, pandas, scipy, scikit-learn, matplotlib.
pyriemann é opcional. Quando disponível, o script usa Covariances/mean/tangent_space do pyriemann.
Quando não disponível, usa OAS + mapeamento tangente log-Euclidiano compatível para análise offline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal
from scipy.linalg import eigh
from sklearn.covariance import OAS
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Instale pyyaml: pip install pyyaml") from exc

try:
    from pyriemann.estimation import Covariances as PRCovariances
    from pyriemann.utils.mean import mean_covariance as pr_mean_covariance
    from pyriemann.tangentspace import tangent_space as pr_tangent_space
    PYRIEMANN_OK = True
except Exception:
    PRCovariances = None
    pr_mean_covariance = None
    pr_tangent_space = None
    PYRIEMANN_OK = False


# =============================================================================
# Configuração e estrutura
# =============================================================================

LABEL_TO_INT = {"LEFT_MI_STIM": 0, "RIGHT_MI_STIM": 1}
INT_TO_LABEL = {0: "LEFT_MI_STIM", 1: "RIGHT_MI_STIM"}
PROB_COLS = {0: "prob_left", 1: "prob_right"}
PAIR_RE = re.compile(r"^(?P<prefix>.+)_(?P<kind>markers|signal)_(?P<run_id>\d{8}_\d{6})\.csv$")


@dataclass
class AnalysisConfig:
    fs_hz: float = 256.0
    trial_duration_s: float = 3.75
    trial_offset_s: float = 0.0
    window_s: float = 1.0
    step_s: float = 0.05
    bp_order: int = 3
    bp_band_hz: tuple[float, float] = (5.0, 40.0)
    filter_mode: str = "causal"       # causal | zero_phase_offline
    select_by: str = "index"          # index | name
    index_base: int = 1
    select_channels: tuple[Any, ...] = ()
    pca_dim: int = 2
    svc_c: float = 1.0
    svm_gamma: str | float = "scale"
    cv_splits: int = 5
    rng_seed: int = 42
    potato_z: float = 3.0
    potato_min_keep_frac: float = 0.50
    output_subdir: str = "analysis_riemann_potato"
    motor_session_type: str = "EM_treino"
    imagery_session_type: str = "IM_treino"
    online_session_type: str = "IM_online"


def first(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def nested(raw: dict, *keys: str, default=None):
    obj = raw
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def as_tuple(value) -> tuple:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def load_analysis_config(config_path: Optional[Path]) -> tuple[AnalysisConfig, dict]:
    raw = {}
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    bp_band = first(raw.get("bp_band_hz"), nested(raw, "model", "bp_band"), nested(raw, "decoder", "band_hz"), default=[5.0, 40.0])
    protocol = raw.get("protocol", {}) or {}

    cfg = AnalysisConfig(
        fs_hz              = float(first(raw.get("fs_hz"), nested(raw, "model", "fs_hz"), default=256.0)),
        trial_duration_s   = float(first(raw.get("trial_duration_s"), protocol.get("trial_duration_s"), nested(raw, "check_data", "tmax"), default=3.75)),
        trial_offset_s     = float(first(raw.get("trial_offset_s"), nested(raw, "model", "trial_offset_s"), default=0.0)),
        window_s           = float(first(raw.get("window_s"), nested(raw, "model", "epoch_s"), nested(raw, "decoder", "epoch_s"), default=1.0)),
        step_s             = float(first(raw.get("step_s"), nested(raw, "decoder", "step_s"), nested(raw, "model", "step_s"), default=0.05)),
        bp_order           = int(first(raw.get("bp_order"), nested(raw, "model", "bp_order"), nested(raw, "decoder", "filter_order"), default=3)),
        bp_band_hz         = (float(bp_band[0]), float(bp_band[1])),
        filter_mode        = str(first(raw.get("filter_mode"), default="causal")),
        select_by          = str(first(raw.get("select_by"), nested(raw, "model", "select_by"), default="index")),
        index_base         = int(first(raw.get("index_base"), nested(raw, "model", "index_base"), default=1)),
        select_channels    = as_tuple(first(raw.get("select_channels"), nested(raw, "model", "select_channels"), default=[])),
        pca_dim            = int(first(raw.get("pca_dim"), nested(raw, "model", "pca_dim"), default=2)),
        svc_c              = float(first(raw.get("svc_c"), nested(raw, "model", "svc_c"), default=1.0)),
        cv_splits          = int(first(raw.get("cv_splits"), nested(raw, "model", "cv_splits"), default=5)),
        rng_seed           = int(first(raw.get("rng_seed"), nested(raw, "model", "rng_seed"), default=42)),
        potato_z           = float(first(nested(raw, "analysis", "potato_z"), default=3.0)),
        potato_min_keep_frac = float(first(nested(raw, "analysis", "potato_min_keep_frac"), default=0.50)),
        motor_session_type = str(protocol.get("motor_session_type", "EM_treino")),
        imagery_session_type = str(protocol.get("imagery_session_type", "IM_treino")),
        online_session_type = str(protocol.get("online_session_type", "IM_online")),
    )
    return cfg, raw


@dataclass
class PairInfo:
    marker: Path
    signal: Path
    run_id: str
    prefix: str
    session_type: str
    mode: str


@dataclass
class FittedModel:
    session_type: str
    run_id: str
    signal_file: str
    marker_file: str
    fs: float
    ch_names: list[str]
    cmean: np.ndarray
    pca: PCA
    clf: SVC
    potato_global: Any
    train_pca: np.ndarray
    train_y: np.ndarray
    train_trial_id: np.ndarray
    train_t_rel: np.ndarray
    train_keep: np.ndarray
    train_df: pd.DataFrame
    acc_mean: float
    acc_balanced_mean: float


# =============================================================================
# Leitura e pareamento
# =============================================================================

def parse_pair_name(path: Path) -> Optional[tuple[str, str, str]]:
    m = PAIR_RE.match(path.name)
    if not m:
        return None
    return m.group("prefix"), m.group("kind"), m.group("run_id")


def find_pairs(folder: Path, session_type: str, mode: str) -> list[PairInfo]:
    folder = Path(folder)
    if not folder.exists():
        return []

    pairs: list[PairInfo] = []
    for marker in folder.glob("*markers_*.csv"):
        parsed = parse_pair_name(marker)
        if parsed is None:
            continue
        prefix, _, run_id = parsed
        signal_file = folder / f"{prefix}_signal_{run_id}.csv"
        if signal_file.exists():
            pairs.append(PairInfo(marker=marker.resolve(), signal=signal_file.resolve(), run_id=run_id, prefix=prefix, session_type=session_type, mode=mode))

    return sorted(pairs, key=lambda p: p.signal.stat().st_mtime)


def session_subfolder(session_folder: Path, session_type: str, mode: str) -> Path:
    return Path(session_folder) / session_type / mode


def read_signal_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    if "lsl_time_s" not in df.columns:
        raise KeyError(f"{path.name}: coluna 'lsl_time_s' ausente.")

    t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)
    ch_cols = [c for c in df.columns if re.match(r"(?i)^ch\d+$", str(c))]
    if not ch_cols:
        meta = {"iso_time", "lsl_time_s", "local_recv_s", "time_s", "timestamp"}
        ch_cols = [c for c in df.columns if str(c) not in meta]
    if not ch_cols:
        raise KeyError(f"{path.name}: não encontrei colunas de canais.")

    X = df[ch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    return t[ok], X[ok], [str(c) for c in ch_cols]


def read_markers_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    if "lsl_time_s" in df.columns:
        t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)
    elif "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
    else:
        raise KeyError(f"{path.name}: coluna 'lsl_time_s' ou 'time_s' ausente.")

    if "label" in df.columns:
        labels = df["label"].astype(str).str.strip().tolist()
    elif "event" in df.columns:
        labels = df["event"].astype(str).str.strip().tolist()
    elif "code" in df.columns:
        code_map = {1: "BASELINE", 2: "ATTENTION", 3: "LEFT_MI_STIM", 4: "RIGHT_MI_STIM", 5: "ATTEMPT", 6: "REST", 99: "BLOCK_END"}
        codes = pd.to_numeric(df["code"], errors="coerce").fillna(-1).astype(int).to_numpy()
        labels = [code_map.get(int(c), f"UNKNOWN_{int(c)}") for c in codes]
    else:
        labels = [""] * len(df)

    if "code" in df.columns:
        codes = pd.to_numeric(df["code"], errors="coerce").fillna(-1).astype(int).to_numpy()
    else:
        inv = {"BASELINE": 1, "ATTENTION": 2, "LEFT_MI_STIM": 3, "RIGHT_MI_STIM": 4, "ATTEMPT": 5, "REST": 6, "BLOCK_END": 99}
        codes = np.asarray([inv.get(str(l).strip().upper(), -1) for l in labels], int)

    ok = np.isfinite(t)
    return t[ok], [labels[i] for i in np.where(ok)[0]], codes[ok]


def estimate_fs(t: np.ndarray) -> float:
    d = np.diff(t)
    d = d[(d > 0) & np.isfinite(d)]
    return float(1.0 / np.median(d)) if d.size else 0.0


def check_time_overlap(t_mark: np.ndarray, t_sig: np.ndarray, cfg: AnalysisConfig) -> None:
    if len(t_mark) == 0 or len(t_sig) == 0:
        raise ValueError("Marcadores ou sinal vazios.")
    marker_lo, marker_hi = float(np.nanmin(t_mark)), float(np.nanmax(t_mark))
    signal_lo, signal_hi = float(np.nanmin(t_sig)), float(np.nanmax(t_sig))
    overlap_start = max(marker_lo, signal_lo)
    overlap_end = min(marker_hi + cfg.trial_duration_s, signal_hi)
    overlap_s = overlap_end - overlap_start
    if overlap_s < cfg.window_s:
        raise ValueError(
            f"Sem sobreposição temporal suficiente: markers={marker_lo:.3f}-{marker_hi:.3f}, "
            f"signal={signal_lo:.3f}-{signal_hi:.3f}, overlap útil={overlap_s:.3f}s."
        )


# =============================================================================
# Eventos, filtros e janelamento
# =============================================================================

def select_channel_indices(cfg: AnalysisConfig, ch_names: list[str]) -> list[int]:
    selection = list(cfg.select_channels)
    if not selection:
        return list(range(len(ch_names)))

    if cfg.select_by.lower() == "name":
        name_to_idx = {str(nm): i for i, nm in enumerate(ch_names)}
        missing = [nm for nm in selection if str(nm) not in name_to_idx]
        if missing:
            raise ValueError(f"Canais não encontrados: {missing}")
        return [name_to_idx[str(nm)] for nm in selection]

    idx = [int(v) - (1 if cfg.index_base == 1 else 0) for v in selection]
    bad = [i for i in idx if i < 0 or i >= len(ch_names)]
    if bad:
        raise ValueError(f"Índice de canal fora do range: selection={selection}, base={cfg.index_base}, canais={len(ch_names)}")
    return idx


def bandpass(data: np.ndarray, fs: float, cfg: AnalysisConfig) -> np.ndarray:
    low, high = cfg.bp_band_hz
    if not (0 < low < high < fs / 2):
        raise ValueError(f"Banda inválida {cfg.bp_band_hz} para fs={fs:.3f}")
    sos = signal.butter(cfg.bp_order, [low, high], btype="bandpass", fs=fs, output="sos")
    if cfg.filter_mode == "zero_phase_offline":
        return signal.sosfiltfilt(sos, data, axis=0)
    return signal.sosfilt(sos, data, axis=0)


def nearest_index(t: np.ndarray, x: float) -> int:
    i = int(np.searchsorted(t, x))
    if i <= 0:
        return 0
    if i >= len(t):
        return len(t) - 1
    return i - 1 if abs(t[i - 1] - x) <= abs(t[i] - x) else i


def trials_from_markers(t_mark: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    trials = []
    last_class = None
    for ts, lab in zip(t_mark, labels):
        lab = str(lab).strip().upper()
        if lab in LABEL_TO_INT:
            last_class = lab
        elif lab == "ATTEMPT" and last_class in LABEL_TO_INT:
            trials.append({"t0": float(ts), "label": last_class, "y": LABEL_TO_INT[last_class], "trial_local": len(trials)})
    return trials


def sliding_indices(start: int, stop: int, win_n: int, hop_n: int) -> list[np.ndarray]:
    out = []
    i0 = int(start)
    while i0 + win_n <= stop:
        out.append(np.arange(i0, i0 + win_n, dtype=int))
        i0 += hop_n
    return out


def epoch_trials(t_sig: np.ndarray, Xf: np.ndarray, trials: list[dict[str, Any]], fs: float, cfg: AnalysisConfig, run_id: str) -> tuple[np.ndarray, pd.DataFrame]:
    win_n = int(round(cfg.window_s * fs))
    hop_n = max(1, int(round(cfg.step_s * fs)))
    off_n = int(round(cfg.trial_offset_s * fs))
    dur_n = int(round(cfg.trial_duration_s * fs))

    epochs = []
    rows = []
    for tr in trials:
        base = nearest_index(t_sig, tr["t0"])
        i0 = max(0, base + off_n)
        i1 = min(len(t_sig), base + off_n + dur_n)
        for idx in sliding_indices(i0, i1, win_n, hop_n):
            epochs.append(Xf[idx, :].T)
            t_center_abs = float(t_sig[idx[len(idx) // 2]])
            rows.append({
                "run_id": run_id,
                "trial_local": int(tr["trial_local"]),
                "trial_uid": f"{run_id}_trial{int(tr['trial_local']):03d}",
                "true_label": tr["label"],
                "y": int(tr["y"]),
                "t_abs": t_center_abs,
                "t_rel": t_center_abs - float(tr["t0"]),
                "win_start_abs": float(t_sig[idx[0]]),
                "win_end_abs": float(t_sig[idx[-1]]),
            })

    if not epochs:
        raise ValueError("Nenhuma janela válida dentro de ATTEMPT.")

    return np.stack(epochs, axis=0), pd.DataFrame(rows)


def continuous_windows(t_sig: np.ndarray, Xf: np.ndarray, fs: float, cfg: AnalysisConfig, run_id: str) -> tuple[np.ndarray, pd.DataFrame]:
    win_n = int(round(cfg.window_s * fs))
    hop_n = max(1, int(round(cfg.step_s * fs)))
    idxs = sliding_indices(0, len(t_sig), win_n, hop_n)
    if not idxs:
        raise ValueError("Sinal curto demais para janelas contínuas.")
    epochs, rows = [], []
    for k, idx in enumerate(idxs):
        epochs.append(Xf[idx, :].T)
        rows.append({
            "run_id": run_id,
            "win_idx": k,
            "t_abs": float(t_sig[idx[len(idx) // 2]]),
            "t_rel_recording": float(t_sig[idx[len(idx) // 2]] - t_sig[0]),
            "win_start_abs": float(t_sig[idx[0]]),
            "win_end_abs": float(t_sig[idx[-1]]),
        })
    return np.stack(epochs, axis=0), pd.DataFrame(rows)


def attach_trial_context(df: pd.DataFrame, trials: list[dict[str, Any]], cfg: AnalysisConfig) -> pd.DataFrame:
    df = df.copy()
    df["trial_uid"] = None
    df["trial_local"] = np.nan
    df["true_label"] = None
    df["y"] = np.nan
    df["t_rel"] = np.nan
    for tr in trials:
        t0 = float(tr["t0"])
        mask = (df["t_abs"] >= t0 + cfg.trial_offset_s) & (df["t_abs"] <= t0 + cfg.trial_duration_s)
        df.loc[mask, "trial_uid"] = f"{df.loc[mask, 'run_id'].iloc[0]}_trial{int(tr['trial_local']):03d}" if mask.any() else None
        df.loc[mask, "trial_local"] = int(tr["trial_local"])
        df.loc[mask, "true_label"] = tr["label"]
        df.loc[mask, "y"] = int(tr["y"])
        df.loc[mask, "t_rel"] = df.loc[mask, "t_abs"] - t0
    return df


# =============================================================================
# Riemannian features e Potato
# =============================================================================

def ensure_spd(C: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    C = 0.5 * (C + C.T)
    vals = np.linalg.eigvalsh(C)
    minv = float(np.min(vals))
    if minv < eps:
        C = C + np.eye(C.shape[0]) * (eps - minv)
    return C


def compute_covariances(Xw: np.ndarray) -> np.ndarray:
    if PYRIEMANN_OK:
        return PRCovariances("oas").transform(Xw)

    covs = []
    for ep in Xw:
        # ep = C x T; OAS espera T x C.
        cov = OAS(store_precision=False, assume_centered=False).fit(ep.T).covariance_
        covs.append(ensure_spd(cov))
    return np.stack(covs, axis=0)


def spd_power(C: np.ndarray, power: float) -> np.ndarray:
    w, v = eigh(ensure_spd(C))
    w = np.maximum(w, 1e-12)
    return (v * (w ** power)) @ v.T


def spd_log(C: np.ndarray) -> np.ndarray:
    w, v = eigh(ensure_spd(C))
    w = np.maximum(w, 1e-12)
    return (v * np.log(w)) @ v.T


def spd_exp(S: np.ndarray) -> np.ndarray:
    w, v = eigh(0.5 * (S + S.T))
    return (v * np.exp(w)) @ v.T


def mean_cov(covs: np.ndarray) -> np.ndarray:
    if PYRIEMANN_OK:
        return pr_mean_covariance(covs, metric="riemann")
    # fallback log-Euclidiano
    return ensure_spd(spd_exp(np.mean([spd_log(C) for C in covs], axis=0)))


def riemann_distances(covs: np.ndarray, center: np.ndarray) -> np.ndarray:
    inv_sqrt = spd_power(center, -0.5)
    out = []
    for C in covs:
        S = ensure_spd(inv_sqrt @ C @ inv_sqrt)
        vals = np.linalg.eigvalsh(S)
        vals = np.maximum(vals, 1e-12)
        out.append(float(np.sqrt(np.sum(np.log(vals) ** 2))))
    return np.asarray(out, float)


def tangent(covs: np.ndarray, center: np.ndarray) -> np.ndarray:
    if PYRIEMANN_OK:
        return pr_tangent_space(covs, center)

    inv_sqrt = spd_power(center, -0.5)
    C = covs.shape[1]
    iu = np.triu_indices(C)
    vecs = []
    for cov in covs:
        S = spd_log(ensure_spd(inv_sqrt @ cov @ inv_sqrt))
        v = S[iu].copy()
        offdiag = iu[0] != iu[1]
        v[offdiag] *= np.sqrt(2.0)
        vecs.append(v)
    return np.asarray(vecs, float)


class RiemannianPotatoLite:
    """Potato robusta por distância Riemanniana ao centro da distribuição."""

    def __init__(self, z_threshold: float = 3.0, min_keep_frac: float = 0.50):
        self.z_threshold = float(z_threshold)
        self.min_keep_frac = float(min_keep_frac)
        self.center_: Optional[np.ndarray] = None
        self.median_: float = 0.0
        self.scale_: float = 1.0
        self.cutoff_: float = np.inf

    def fit(self, covs: np.ndarray):
        self.center_ = mean_cov(covs)
        d = riemann_distances(covs, self.center_)
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        scale = 1.4826 * mad if mad > 1e-12 else float(np.std(d) if np.std(d) > 1e-12 else 1.0)
        cutoff = med + self.z_threshold * scale
        # Evita remover quase tudo em blocos pequenos/ruidosos.
        q = float(np.quantile(d, max(self.min_keep_frac, 0.01)))
        self.median_ = med
        self.scale_ = scale
        self.cutoff_ = max(cutoff, q)
        return self

    def distances(self, covs: np.ndarray) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("Potato ainda não foi ajustada.")
        return riemann_distances(covs, self.center_)

    def predict(self, covs: np.ndarray) -> np.ndarray:
        d = self.distances(covs)
        return np.where(d <= self.cutoff_, 1, -1)

    def decision_function(self, covs: np.ndarray) -> np.ndarray:
        d = self.distances(covs)
        return (d - self.median_) / max(self.scale_, 1e-12)


def classwise_potato_mask(covs: np.ndarray, y: np.ndarray, cfg: AnalysisConfig) -> tuple[np.ndarray, dict[int, RiemannianPotatoLite], np.ndarray]:
    keep = np.zeros(len(covs), dtype=bool)
    z = np.full(len(covs), np.nan, dtype=float)
    potatoes: dict[int, RiemannianPotatoLite] = {}
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        pot = RiemannianPotatoLite(cfg.potato_z, cfg.potato_min_keep_frac).fit(covs[idx])
        pred = pot.predict(covs[idx])
        keep[idx] = pred == 1
        z[idx] = pot.decision_function(covs[idx])
        potatoes[int(cls)] = pot
    return keep, potatoes, z


def fit_feature_model(covs: np.ndarray, y: np.ndarray, cfg: AnalysisConfig) -> tuple[np.ndarray, PCA, SVC, np.ndarray]:
    cmean = mean_cov(covs)
    ts = tangent(covs, cmean)
    dim = max(1, min(int(cfg.pca_dim), ts.shape[1], ts.shape[0]))
    pca = PCA(n_components=dim, random_state=cfg.rng_seed).fit(ts)
    Xp = pca.transform(ts)
    clf = SVC(kernel="rbf", C=cfg.svc_c, gamma=cfg.svm_gamma, probability=True, random_state=cfg.rng_seed)
    clf.fit(Xp, y)
    return cmean, pca, clf, Xp


def predict_feature_model(covs: np.ndarray, cmean: np.ndarray, pca: PCA, clf: SVC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = tangent(covs, cmean)
    Xp = pca.transform(ts)
    proba_raw = clf.predict_proba(Xp)
    # Garantir colunas [LEFT, RIGHT]
    proba = np.zeros((len(Xp), 2), float)
    for j, cls in enumerate(clf.classes_):
        if int(cls) in (0, 1):
            proba[:, int(cls)] = proba_raw[:, j]
    pred = np.argmax(proba, axis=1)
    return Xp, proba, pred


# =============================================================================
# Plots
# =============================================================================

def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pca_2d(Xp: np.ndarray) -> np.ndarray:
    if Xp.shape[1] >= 2:
        return Xp[:, :2]
    return np.c_[Xp[:, 0], np.zeros(len(Xp))]


def plot_pca_classifier(Xp: np.ndarray, y: np.ndarray, clf: SVC, keep: Optional[np.ndarray], out_png: Path, title: str) -> None:
    X2 = pca_2d(Xp)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Superfície do RBF no plano do PCA.
    xpad = 0.10 * max(np.ptp(X2[:, 0]), 1e-6)
    ypad = 0.10 * max(np.ptp(X2[:, 1]), 1e-6)
    xx, yy = np.meshgrid(
        np.linspace(X2[:, 0].min() - xpad, X2[:, 0].max() + xpad, 180),
        np.linspace(X2[:, 1].min() - ypad, X2[:, 1].max() + ypad, 180),
    )
    if Xp.shape[1] == 1:
        grid = xx.reshape(-1, 1)
    else:
        grid = np.c_[xx.ravel(), yy.ravel()]
        if Xp.shape[1] > 2:
            grid = np.c_[grid, np.zeros((len(grid), Xp.shape[1] - 2))]
    zz = clf.predict_proba(grid)
    p_right = np.zeros(len(grid), float)
    for j, cls in enumerate(clf.classes_):
        if int(cls) == 1:
            p_right = zz[:, j]
    p_right = p_right.reshape(xx.shape)
    ax.contourf(xx, yy, p_right, levels=np.linspace(0, 1, 21), alpha=0.25)
    ax.contour(xx, yy, p_right, levels=[0.5], linewidths=1.5)

    if keep is not None and (~keep).any():
        ax.scatter(X2[~keep, 0], X2[~keep, 1], marker="x", s=24, alpha=0.55, label="outlier Potato")

    for cls, lab in [(0, "LEFT"), (1, "RIGHT")]:
        m = (y == cls) & (keep if keep is not None else np.ones(len(y), dtype=bool))
        if m.any():
            ax.scatter(X2[m, 0], X2[m, 1], s=22, alpha=0.8, label=lab)
            mu = X2[m].mean(axis=0)
            ax.scatter([mu[0]], [mu[1]], marker="P", s=120, edgecolor="black", label=f"centro {lab}")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    savefig(fig, out_png)


def plot_cv_summary(summary: pd.DataFrame, out_png: Path, title: str) -> None:
    if summary.empty:
        return
    df = summary.copy().sort_values("acc_mean")
    labels = df["stage"].astype(str) + " | " + df["run_id"].astype(str)
    fig_h = max(4, 0.45 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(labels, df["acc_mean"])
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Acurácia média CV")
    ax.set_title(title)
    for i, v in enumerate(df["acc_mean"]):
        ax.text(float(v) + 0.01, i, f"{float(v):.2f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    savefig(fig, out_png)


def aggregate_mean_sem(df: pd.DataFrame, value_col: str, t_col: str = "t_rel", round_decimals: int = 2) -> pd.DataFrame:
    tmp = df.dropna(subset=[t_col, value_col]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=[t_col, "mean", "sem", "n"])
    tmp["t_bin"] = tmp[t_col].round(round_decimals)
    g = tmp.groupby("t_bin")[value_col]
    out = g.agg(["mean", "count", "std"]).reset_index().rename(columns={"t_bin": t_col, "count": "n"})
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out[[t_col, "mean", "sem", "n"]]


def plot_trial_curves(pred_df: pd.DataFrame, out_png: Path, title: str, inlier_only: bool = True) -> None:
    df = pred_df.copy()
    if inlier_only and "inlier" in df.columns:
        df = df[df["inlier"]]
    df = df.dropna(subset=["trial_uid", "t_rel", "true_label"])
    if df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
    panels = [("LEFT_MI_STIM", "prob_left", "LEFT: P(LEFT)"), ("RIGHT_MI_STIM", "prob_right", "RIGHT: P(RIGHT)")]
    for ax, (lab, col, ylabel) in zip(axes, panels):
        sub = df[df["true_label"] == lab]
        for _, tr in sub.groupby("trial_uid"):
            tr = tr.sort_values("t_rel")
            ax.plot(tr["t_rel"], tr[col], linewidth=0.7, alpha=0.25)
        mean = aggregate_mean_sem(sub, col)
        if not mean.empty:
            ax.plot(mean["t_rel"], mean["mean"], linewidth=2.2, label="média")
            ax.fill_between(mean["t_rel"].to_numpy(), (mean["mean"] - mean["sem"]).to_numpy(), (mean["mean"] + mean["sem"]).to_numpy(), alpha=0.18, label="SEM")
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="lower right")
    axes[-1].set_xlabel("Tempo relativo ao ATTEMPT (s)")
    fig.suptitle(title)
    savefig(fig, out_png)


def plot_online_timeline(pred_df: pd.DataFrame, trials: list[dict[str, Any]], out_png: Path, title: str, trial_duration_s: float = 3.75) -> None:
    df = pred_df.copy().sort_values("t_rel_recording")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    x = df["t_rel_recording"]
    ax.plot(x, df["prob_left"], linewidth=1.2, label="P(LEFT)")
    ax.plot(x, df["prob_right"], linewidth=1.2, label="P(RIGHT)")
    if "inlier" in df.columns and (~df["inlier"]).any():
        bad = df[~df["inlier"]]
        ax.scatter(bad["t_rel_recording"], np.full(len(bad), 1.03), marker="x", s=15, label="outlier Potato")

    t0_recording = float(df["t_abs"].min() - df["t_rel_recording"].min())
    for tr in trials:
        start = float(tr["t0"] - t0_recording)
        end = start + float(np.nanmax([0.0, df["t_rel"].dropna().max() if "t_rel" in df else 0.0]))
        end = start + float(trial_duration_s)
        ax.axvspan(start, end, alpha=0.07)
        ax.text(start, 1.08, "L" if tr["y"] == 0 else "R", fontsize=8, ha="center")

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(-0.02, 1.14)
    ax.set_xlabel("Tempo desde o início da gravação (s)")
    ax.set_ylabel("Probabilidade")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    savefig(fig, out_png)


def plot_online_pca_template(model: FittedModel, Xp_online: np.ndarray, pred_df: pd.DataFrame, out_png: Path, title: str) -> None:
    Xt = pca_2d(model.train_pca)
    Xo = pca_2d(Xp_online)
    df = pred_df.copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls, lab in [(0, "template LEFT"), (1, "template RIGHT")]:
        m = model.train_y == cls
        ax.scatter(Xt[m, 0], Xt[m, 1], s=16, alpha=0.20, label=lab)

    inlier = df["inlier"].to_numpy(bool) if "inlier" in df.columns else np.ones(len(df), dtype=bool)
    if (~inlier).any():
        ax.scatter(Xo[~inlier, 0], Xo[~inlier, 1], marker="x", s=22, alpha=0.45, label="online outlier")

    pred = df["pred"].to_numpy(int)
    for cls, lab in [(0, "online pred LEFT"), (1, "online pred RIGHT")]:
        m = inlier & (pred == cls)
        if m.any():
            ax.scatter(Xo[m, 0], Xo[m, 1], s=20, alpha=0.65, label=lab)

    ax.set_xlabel("PC1 do template MI")
    ax.set_ylabel("PC2 do template MI")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    savefig(fig, out_png)


# =============================================================================
# Análise de blocos de treino
# =============================================================================

def load_pair_epochs(pair: PairInfo, cfg: AnalysisConfig) -> tuple[np.ndarray, pd.DataFrame, list[str], float, list[dict[str, Any]]]:
    t_mark, labels, _ = read_markers_csv(pair.marker)
    t_sig, X, ch_names = read_signal_csv(pair.signal)
    check_time_overlap(t_mark, t_sig, cfg)

    fs_est = estimate_fs(t_sig)
    fs = cfg.fs_hz if cfg.fs_hz else fs_est
    if fs_est and abs(fs_est - cfg.fs_hz) / max(cfg.fs_hz, 1e-9) > 0.02:
        print(f"[warn] {pair.run_id}: fs estimada={fs_est:.3f} Hz difere da config={cfg.fs_hz:.3f} Hz. Usando fs estimada.")
        fs = fs_est

    sel = select_channel_indices(cfg, ch_names)
    ch_sel = [ch_names[i] for i in sel]
    X_sel = X[:, sel]
    Xf = bandpass(X_sel, fs, cfg)
    trials = trials_from_markers(t_mark, labels)
    Xw, meta = epoch_trials(t_sig, Xf, trials, fs, cfg, pair.run_id)
    return Xw, meta, ch_sel, fs, trials


def safe_splits(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    counts = []
    for cls in np.unique(y):
        counts.append(len(np.unique(groups[y == cls])))
    if counts and min(counts) >= 2:
        return max(2, min(int(requested), min(counts)))
    return 0


def cross_validate_epochs(Xw: np.ndarray, meta: pd.DataFrame, cfg: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = meta["y"].to_numpy(int)
    groups = pd.factorize(meta["trial_uid"])[0]
    covs = compute_covariances(Xw)

    n_splits = safe_splits(y, groups, cfg.cv_splits)
    pred_df = meta.copy()
    pred_df["prob_left"] = np.nan
    pred_df["prob_right"] = np.nan
    pred_df["pred"] = np.nan
    pred_df["fold"] = np.nan
    pred_df["inlier_train_context"] = True

    if n_splits >= 2:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.rng_seed)
        split_iter = splitter.split(covs, y, groups=groups)
    else:
        # Fallback se houver poucos trials: não garante independência por trial.
        n_splits = min(3, max(2, np.min(np.bincount(y))))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.rng_seed)
        split_iter = splitter.split(covs, y)

    fold_rows = []
    for fold, (tr, va) in enumerate(split_iter, start=1):
        keep_tr, _, _ = classwise_potato_mask(covs[tr], y[tr], cfg)
        tr_clean = tr[keep_tr]
        if len(np.unique(y[tr_clean])) < 2:
            tr_clean = tr
        cmean, pca, clf, _ = fit_feature_model(covs[tr_clean], y[tr_clean], cfg)
        _, proba, pred = predict_feature_model(covs[va], cmean, pca, clf)
        pred_df.loc[va, "prob_left"] = proba[:, 0]
        pred_df.loc[va, "prob_right"] = proba[:, 1]
        pred_df.loc[va, "pred"] = pred
        pred_df.loc[va, "fold"] = fold
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_train_clean": int(len(tr_clean)),
            "n_valid": int(len(va)),
            "acc": float(accuracy_score(y[va], pred)),
            "balanced_acc": float(balanced_accuracy_score(y[va], pred)),
        })

    fold_df = pd.DataFrame(fold_rows)
    stats = {
        "n_windows": int(len(Xw)),
        "n_trials": int(len(np.unique(groups))),
        "n_splits": int(n_splits),
        "acc_mean": float(fold_df["acc"].mean()) if not fold_df.empty else float("nan"),
        "acc_std": float(fold_df["acc"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "balanced_acc_mean": float(fold_df["balanced_acc"].mean()) if not fold_df.empty else float("nan"),
        "balanced_acc_std": float(fold_df["balanced_acc"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "folds": fold_rows,
    }
    return pred_df, stats


def fit_final_from_epochs(pair: PairInfo, Xw: np.ndarray, meta: pd.DataFrame, ch_names: list[str], fs: float, cfg: AnalysisConfig, acc_mean: float, balanced_acc_mean: float) -> FittedModel:
    y = meta["y"].to_numpy(int)
    trial_id = pd.factorize(meta["trial_uid"])[0]
    covs = compute_covariances(Xw)
    keep, _, z = classwise_potato_mask(covs, y, cfg)
    if len(np.unique(y[keep])) < 2:
        keep[:] = True
    cmean, pca, clf, Xp_clean = fit_feature_model(covs[keep], y[keep], cfg)
    potato_global = RiemannianPotatoLite(cfg.potato_z, cfg.potato_min_keep_frac).fit(covs[keep])

    # PCA de todos os pontos no template final para permitir plotar outliers junto.
    Xp_all, proba_all, pred_all = predict_feature_model(covs, cmean, pca, clf)
    train_df = meta.copy()
    train_df["inlier"] = keep
    train_df["potato_z"] = z
    train_df["prob_left_final"] = proba_all[:, 0]
    train_df["prob_right_final"] = proba_all[:, 1]
    train_df["pred_final"] = pred_all
    train_df["pc1"] = pca_2d(Xp_all)[:, 0]
    train_df["pc2"] = pca_2d(Xp_all)[:, 1]

    return FittedModel(
        session_type=pair.session_type,
        run_id=pair.run_id,
        signal_file=str(pair.signal),
        marker_file=str(pair.marker),
        fs=float(fs),
        ch_names=list(ch_names),
        cmean=cmean,
        pca=pca,
        clf=clf,
        potato_global=potato_global,
        train_pca=Xp_clean,
        train_y=y[keep],
        train_trial_id=trial_id[keep],
        train_t_rel=meta["t_rel"].to_numpy(float)[keep],
        train_keep=keep,
        train_df=train_df,
        acc_mean=float(acc_mean),
        acc_balanced_mean=float(balanced_acc_mean),
    )


def analyze_training_pair(pair: PairInfo, cfg: AnalysisConfig, out_root: Path, stage_label: str) -> tuple[dict[str, Any], FittedModel]:
    print(f"\n[{stage_label}] {pair.run_id}")
    Xw, meta, ch_names, fs, _ = load_pair_epochs(pair, cfg)
    pred_oof, cv_stats = cross_validate_epochs(Xw, meta, cfg)
    model = fit_final_from_epochs(pair, Xw, meta, ch_names, fs, cfg, cv_stats["acc_mean"], cv_stats["balanced_acc_mean"])

    out_dir = out_root / pair.session_type / pair.mode / cfg.output_subdir / pair.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_oof.to_csv(out_dir / f"{pair.run_id}_cv_oof_predictions.csv", index=False)
    pd.DataFrame(cv_stats["folds"]).to_csv(out_dir / f"{pair.run_id}_cv_folds.csv", index=False)
    model.train_df.to_csv(out_dir / f"{pair.run_id}_final_train_windows.csv", index=False)

    with open(out_dir / f"{pair.run_id}_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    plot_trial_curves(pred_oof, out_dir / f"{pair.run_id}_cv_trial_curves.png", f"{stage_label} {pair.run_id} — curvas CV por tentativa")

    # PCA com superfície do classificador final. Precisa projetar todos os pontos no template final.
    Xp_all = model.train_df[["pc1", "pc2"]].to_numpy(float)
    if model.pca.n_components_ == 1:
        Xp_plot = Xp_all[:, :1]
    else:
        # Para decisão, usa espaço de PCA real. Aqui recalculamos via predict_feature_model nos covs para manter dimensões completas.
        covs = compute_covariances(Xw)
        Xp_full, _, _ = predict_feature_model(covs, model.cmean, model.pca, model.clf)
        Xp_plot = Xp_full
    plot_pca_classifier(Xp_plot, meta["y"].to_numpy(int), model.clf, model.train_df["inlier"].to_numpy(bool), out_dir / f"{pair.run_id}_pca_potato_svm_rbf.png", f"{stage_label} {pair.run_id} — PCA após Potato + SVM RBF")

    info = {
        "stage": stage_label,
        "session_type": pair.session_type,
        "mode": pair.mode,
        "run_id": pair.run_id,
        "marker_file": str(pair.marker),
        "signal_file": str(pair.signal),
        "n_windows": int(cv_stats["n_windows"]),
        "n_trials": int(cv_stats["n_trials"]),
        "n_channels": int(len(ch_names)),
        "fs": float(fs),
        "acc_mean": float(cv_stats["acc_mean"]),
        "acc_std": float(cv_stats["acc_std"]),
        "balanced_acc_mean": float(cv_stats["balanced_acc_mean"]),
        "balanced_acc_std": float(cv_stats["balanced_acc_std"]),
        "n_inliers_final": int(model.train_df["inlier"].sum()),
        "n_outliers_final": int((~model.train_df["inlier"].to_numpy(bool)).sum()),
        "out_dir": str(out_dir),
    }
    with open(out_dir / f"{pair.run_id}_summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info, model



def analyze_training_session_pooled(pairs: list[PairInfo], cfg: AnalysisConfig, out_root: Path, stage_label: str) -> Optional[dict[str, Any]]:
    """Validação cruzada agregada em todos os blocos de uma fase da sessão."""
    if not pairs:
        return None

    print(f"\n[{stage_label}] validação cruzada agregada da sessão | blocos={len(pairs)}")
    Xs, metas = [], []
    ch_ref: Optional[list[str]] = None
    fs_ref: Optional[float] = None

    for pair in pairs:
        try:
            Xw, meta, ch_names, fs, _ = load_pair_epochs(pair, cfg)
        except Exception as exc:
            print(f"[pooled {stage_label}] pulando {pair.run_id}: {type(exc).__name__}: {exc}")
            continue
        if ch_ref is None:
            ch_ref = ch_names
            fs_ref = fs
        elif len(ch_names) != len(ch_ref):
            print(f"[pooled {stage_label}] pulando {pair.run_id}: número de canais diferente ({len(ch_names)} != {len(ch_ref)})")
            continue
        Xs.append(Xw)
        metas.append(meta)

    if not Xs:
        return None

    X_all = np.concatenate(Xs, axis=0)
    meta_all = pd.concat(metas, ignore_index=True)
    pseudo_pair = PairInfo(marker=pairs[0].marker, signal=pairs[0].signal, run_id=f"{stage_label}_session", prefix=stage_label, session_type=pairs[0].session_type, mode=pairs[0].mode)

    pred_oof, cv_stats = cross_validate_epochs(X_all, meta_all, cfg)
    model = fit_final_from_epochs(pseudo_pair, X_all, meta_all, ch_ref or [], fs_ref or cfg.fs_hz, cfg, cv_stats["acc_mean"], cv_stats["balanced_acc_mean"])

    out_dir = out_root / cfg.output_subdir / f"{stage_label}_session_cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_oof.to_csv(out_dir / f"{stage_label}_session_cv_oof_predictions.csv", index=False)
    pd.DataFrame(cv_stats["folds"]).to_csv(out_dir / f"{stage_label}_session_cv_folds.csv", index=False)
    model.train_df.to_csv(out_dir / f"{stage_label}_session_final_train_windows.csv", index=False)

    plot_trial_curves(pred_oof, out_dir / f"{stage_label}_session_cv_trial_curves.png", f"{stage_label} — validação cruzada agregada da sessão")
    covs = compute_covariances(X_all)
    Xp_full, _, _ = predict_feature_model(covs, model.cmean, model.pca, model.clf)
    plot_pca_classifier(Xp_full, meta_all["y"].to_numpy(int), model.clf, model.train_df["inlier"].to_numpy(bool), out_dir / f"{stage_label}_session_pca_potato_svm_rbf.png", f"{stage_label} — PCA sessão após Potato + SVM RBF")

    info = {
        "stage": f"{stage_label}_session",
        "session_type": pairs[0].session_type,
        "mode": pairs[0].mode,
        "run_id": f"{stage_label}_session",
        "n_blocks": int(len(pairs)),
        "n_windows": int(cv_stats["n_windows"]),
        "n_trials": int(cv_stats["n_trials"]),
        "n_channels": int(len(ch_ref or [])),
        "fs": float(fs_ref or cfg.fs_hz),
        "acc_mean": float(cv_stats["acc_mean"]),
        "acc_std": float(cv_stats["acc_std"]),
        "balanced_acc_mean": float(cv_stats["balanced_acc_mean"]),
        "balanced_acc_std": float(cv_stats["balanced_acc_std"]),
        "n_inliers_final": int(model.train_df["inlier"].sum()),
        "n_outliers_final": int((~model.train_df["inlier"].to_numpy(bool)).sum()),
        "out_dir": str(out_dir),
    }
    with open(out_dir / f"{stage_label}_session_summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


# =============================================================================
# Online
# =============================================================================

def analyze_online_pair(pair: PairInfo, cfg: AnalysisConfig, model: FittedModel, out_root: Path) -> dict[str, Any]:
    print(f"\n[ONLINE] {pair.run_id} usando modelo MI {model.run_id}")
    t_mark, labels, _ = read_markers_csv(pair.marker)
    t_sig, X, ch_names = read_signal_csv(pair.signal)
    check_time_overlap(t_mark, t_sig, cfg)
    fs_est = estimate_fs(t_sig)
    fs = cfg.fs_hz if cfg.fs_hz else fs_est
    if fs_est and abs(fs_est - cfg.fs_hz) / max(cfg.fs_hz, 1e-9) > 0.02:
        print(f"[warn] online {pair.run_id}: fs estimada={fs_est:.3f} Hz difere da config={cfg.fs_hz:.3f} Hz. Usando fs estimada.")
        fs = fs_est

    # Seleciona por nomes do modelo quando possível. Fallback: mesmos índices.
    if set(model.ch_names).issubset(set(ch_names)):
        sel = [ch_names.index(ch) for ch in model.ch_names]
        ch_sel = [ch_names[i] for i in sel]
    else:
        sel = select_channel_indices(cfg, ch_names)
        ch_sel = [ch_names[i] for i in sel]
        if len(ch_sel) != len(model.ch_names):
            raise ValueError(f"Online tem {len(ch_sel)} canais selecionados, modelo espera {len(model.ch_names)}.")

    Xf = bandpass(X[:, sel], fs, cfg)
    trials = trials_from_markers(t_mark, labels)
    Xw, pred_df = continuous_windows(t_sig, Xf, fs, cfg, pair.run_id)
    pred_df = attach_trial_context(pred_df, trials, cfg)
    covs = compute_covariances(Xw)
    potato_pred = model.potato_global.predict(covs)
    pred_df["inlier"] = potato_pred == 1
    pred_df["potato_z"] = model.potato_global.decision_function(covs)

    Xp, proba, pred = predict_feature_model(covs, model.cmean, model.pca, model.clf)
    pred_df["prob_left"] = proba[:, 0]
    pred_df["prob_right"] = proba[:, 1]
    pred_df["pred"] = pred
    pred_df["pred_label"] = [INT_TO_LABEL[int(p)] for p in pred]
    pred_df["pc1"] = pca_2d(Xp)[:, 0]
    pred_df["pc2"] = pca_2d(Xp)[:, 1]

    # Métricas só em janelas que caem dentro de trials rotulados.
    valid = pred_df["inlier"] & pred_df["y"].notna()
    if valid.any():
        y_true = pred_df.loc[valid, "y"].astype(int).to_numpy()
        y_pred = pred_df.loc[valid, "pred"].astype(int).to_numpy()
        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    else:
        acc = float("nan")
        bacc = float("nan")
        cm = np.zeros((2, 2), int)

    out_dir = out_root / pair.session_type / pair.mode / cfg.output_subdir / pair.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / f"{pair.run_id}_online_predictions.csv", index=False)
    pd.DataFrame(cm, index=["true_LEFT", "true_RIGHT"], columns=["pred_LEFT", "pred_RIGHT"]).to_csv(out_dir / f"{pair.run_id}_online_confusion.csv")

    plot_online_timeline(pred_df, trials, out_dir / f"{pair.run_id}_online_decode_timeline.png", f"Online {pair.run_id} — timeline de decodificação", trial_duration_s=cfg.trial_duration_s)
    plot_trial_curves(pred_df, out_dir / f"{pair.run_id}_online_trial_curves.png", f"Online {pair.run_id} — curvas por tentativa")
    plot_online_pca_template(model, Xp, pred_df, out_dir / f"{pair.run_id}_online_pca_template.png", f"Online {pair.run_id} — PCA alinhado ao template MI {model.run_id}")

    info = {
        "stage": "online",
        "session_type": pair.session_type,
        "mode": pair.mode,
        "run_id": pair.run_id,
        "model_run_id": model.run_id,
        "marker_file": str(pair.marker),
        "signal_file": str(pair.signal),
        "n_windows": int(len(pred_df)),
        "n_inlier_windows": int(pred_df["inlier"].sum()),
        "n_outlier_windows": int((~pred_df["inlier"].to_numpy(bool)).sum()),
        "n_trial_windows_inlier": int(valid.sum()),
        "acc_trial_windows": acc,
        "balanced_acc_trial_windows": bacc,
        "out_dir": str(out_dir),
    }
    with open(out_dir / f"{pair.run_id}_online_summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


# =============================================================================
# Sessão completa
# =============================================================================

def find_specific_mi_pair(mi_pairs: list[PairInfo], run_id: Optional[str], signal_path: Optional[Path]) -> Optional[PairInfo]:
    if signal_path is not None:
        signal_path = signal_path.resolve()
        for pair in mi_pairs:
            if pair.signal.resolve() == signal_path:
                return pair
        raise FileNotFoundError(f"--mi-model-signal não corresponde a nenhum par MI encontrado: {signal_path}")
    if run_id is not None:
        for pair in mi_pairs:
            if pair.run_id == run_id:
                return pair
        raise FileNotFoundError(f"--mi-model-run-id não encontrado entre os pares MI: {run_id}")
    return None


def write_readme(out_root: Path, cfg: AnalysisConfig, selected_model: Optional[FittedModel]) -> None:
    lines = [
        "# Análise BCI — Riemannian Potato + Tangent/PCA + SVM RBF",
        "",
        "Arquivos gerados automaticamente por analyze_bci_session_riemann_potato.py.",
        "",
        "## Configuração principal",
        f"- fs_hz: {cfg.fs_hz}",
        f"- banda: {cfg.bp_band_hz}",
        f"- janela: {cfg.window_s}s",
        f"- step: {cfg.step_s}s",
        f"- trial_duration_s: {cfg.trial_duration_s}s",
        f"- filtro: {cfg.filter_mode}",
        f"- Potato z-threshold: {cfg.potato_z}",
        f"- classificador: SVM RBF, C={cfg.svc_c}, gamma={cfg.svm_gamma}",
        "",
    ]
    if selected_model is not None:
        lines += [
            "## Modelo MI usado no online",
            f"- run_id: {selected_model.run_id}",
            f"- signal: {selected_model.signal_file}",
            f"- CV acc média: {selected_model.acc_mean:.4f}",
            "",
        ]
    lines += [
        "## Figuras principais por bloco",
        "- *_cv_trial_curves.png: curvas de probabilidade por tentativa na validação cruzada.",
        "- *_pca_potato_svm_rbf.png: PCA após Potato com superfície do SVM RBF.",
        "- *_online_decode_timeline.png: timeline online de P(LEFT)/P(RIGHT).",
        "- *_online_trial_curves.png: médias online por classe de tentativa.",
        "- *_online_pca_template.png: online projetado no PCA/template MI selecionado.",
    ]
    (out_root / "README_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Análise offline de sessão BCI com Riemannian Potato + PCA + SVM RBF.")
    ap.add_argument("--session-folder", required=True, type=Path, help="Pasta da sessão. Ex.: .../SY001/S3")
    ap.add_argument("--config", type=Path, default=None, help="config.yaml do protocolo")
    ap.add_argument("--mi-model-run-id", default=None, help="run_id específico do treino MI usado para calibrar o online. Ex.: 20260622_153353")
    ap.add_argument("--mi-model-signal", type=Path, default=None, help="Arquivo *_IM_treino_train_signal_<run_id>.csv específico usado para calibrar o online")
    ap.add_argument("--output-subdir", default=None, help="Nome da subpasta de saída dentro das pastas train/realtime")
    ap.add_argument("--potato-z", type=float, default=None, help="Threshold robusto da Potato. Default: 3.0")
    ap.add_argument("--svm-c", type=float, default=None, help="C do SVM RBF")
    ap.add_argument("--svm-gamma", default=None, help="gamma do SVM RBF: scale, auto ou número")
    ap.add_argument("--skip-online", action="store_true", help="Analisa apenas treinos")
    args = ap.parse_args()

    cfg, raw = load_analysis_config(args.config)
    if args.output_subdir:
        cfg.output_subdir = args.output_subdir
    if args.potato_z is not None:
        cfg.potato_z = float(args.potato_z)
    if args.svm_c is not None:
        cfg.svc_c = float(args.svm_c)
    if args.svm_gamma is not None:
        try:
            cfg.svm_gamma = float(args.svm_gamma)
        except ValueError:
            cfg.svm_gamma = args.svm_gamma

    session_folder = args.session_folder.resolve()
    out_root = session_folder / cfg.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    print("===== ANÁLISE BCI OFFLINE =====")
    print(f"session_folder: {session_folder}")
    print(f"pyriemann disponível: {PYRIEMANN_OK}")
    print(f"saída resumo: {out_root}")

    em_folder = session_subfolder(session_folder, cfg.motor_session_type, "train")
    mi_folder = session_subfolder(session_folder, cfg.imagery_session_type, "train")
    on_folder = session_subfolder(session_folder, cfg.online_session_type, "realtime")

    em_pairs = find_pairs(em_folder, cfg.motor_session_type, "train")
    mi_pairs = find_pairs(mi_folder, cfg.imagery_session_type, "train")
    online_pairs = find_pairs(on_folder, cfg.online_session_type, "realtime")

    print(f"Pares EM treino: {len(em_pairs)} | pasta={em_folder}")
    print(f"Pares MI treino: {len(mi_pairs)} | pasta={mi_folder}")
    print(f"Pares online  : {len(online_pairs)} | pasta={on_folder}")

    summaries = []
    models: dict[str, FittedModel] = {}

    for pair in em_pairs:
        try:
            info, model = analyze_training_pair(pair, cfg, session_folder, "EM_treino")
            summaries.append(info)
            models[f"EM:{pair.run_id}"] = model
        except Exception as exc:
            print(f"[erro EM {pair.run_id}] {type(exc).__name__}: {exc}")
            summaries.append({"stage": "EM_treino", "run_id": pair.run_id, "error": f"{type(exc).__name__}: {exc}"})

    for pair in mi_pairs:
        try:
            info, model = analyze_training_pair(pair, cfg, session_folder, "IM_treino")
            summaries.append(info)
            models[f"MI:{pair.run_id}"] = model
        except Exception as exc:
            print(f"[erro MI {pair.run_id}] {type(exc).__name__}: {exc}")
            summaries.append({"stage": "IM_treino", "run_id": pair.run_id, "error": f"{type(exc).__name__}: {exc}"})

    # Validação cruzada agregada por fase da sessão.
    for pooled_info in [
        analyze_training_session_pooled(em_pairs, cfg, session_folder, "EM_treino"),
        analyze_training_session_pooled(mi_pairs, cfg, session_folder, "IM_treino"),
    ]:
        if pooled_info is not None:
            summaries.append(pooled_info)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_root / "training_cv_summary.csv", index=False)
    if "acc_mean" in summary_df.columns:
        plot_cv_summary(summary_df.dropna(subset=["acc_mean"]), out_root / "training_cv_summary.png", "Validação cruzada por bloco/sessão")

    selected_pair = find_specific_mi_pair(mi_pairs, args.mi_model_run_id, args.mi_model_signal)
    selected_model: Optional[FittedModel] = None
    if selected_pair is not None:
        selected_model = models.get(f"MI:{selected_pair.run_id}")
        if selected_model is None:
            raise RuntimeError(f"O par MI selecionado {selected_pair.run_id} não gerou modelo válido.")
    else:
        mi_models = [m for k, m in models.items() if k.startswith("MI:")]
        if mi_models:
            selected_model = sorted(mi_models, key=lambda m: (np.nan_to_num(m.acc_balanced_mean, nan=-1), np.nan_to_num(m.acc_mean, nan=-1)), reverse=True)[0]
            print(f"[modelo MI automático] {selected_model.run_id} | bacc={selected_model.acc_balanced_mean:.3f} | acc={selected_model.acc_mean:.3f}")

    if selected_model is not None:
        with open(out_root / "selected_mi_model.pkl", "wb") as f:
            pickle.dump(selected_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_root / "selected_mi_model.json", "w", encoding="utf-8") as f:
            json.dump({
                "run_id": selected_model.run_id,
                "signal_file": selected_model.signal_file,
                "marker_file": selected_model.marker_file,
                "acc_mean": selected_model.acc_mean,
                "balanced_acc_mean": selected_model.acc_balanced_mean,
                "ch_names": selected_model.ch_names,
            }, f, ensure_ascii=False, indent=2)

    online_summaries = []
    if not args.skip_online and selected_model is not None:
        for pair in online_pairs:
            try:
                online_summaries.append(analyze_online_pair(pair, cfg, selected_model, session_folder))
            except Exception as exc:
                print(f"[erro online {pair.run_id}] {type(exc).__name__}: {exc}")
                online_summaries.append({"stage": "online", "run_id": pair.run_id, "error": f"{type(exc).__name__}: {exc}"})
    elif not args.skip_online:
        print("[online] nenhum modelo MI válido disponível; blocos online não analisados.")

    online_df = pd.DataFrame(online_summaries)
    online_df.to_csv(out_root / "online_summary.csv", index=False)

    write_readme(out_root, cfg, selected_model)

    print("\n===== FINALIZADO =====")
    print(f"Resumo de treino: {out_root / 'training_cv_summary.csv'}")
    print(f"Resumo online   : {out_root / 'online_summary.csv'}")
    print(f"README          : {out_root / 'README_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
