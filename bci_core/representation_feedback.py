# -*- coding: utf-8 -*-
"""
representation_feedback.py

Camada genérica entre o espaço usado pelo modelo e o feedback visual.

Princípio:
- o Unity nunca precisa saber se o espaço é PCA, PCA alinhado, Procrustes,
  RPA ou outro embedding;
- o Python produz a representação FINAL usada no feedback;
- o mesmo transform é aplicado às amostras de treino (para gerar o mapa)
  e ao ponto online (antes de enviá-lo ao Unity).

A implementação padrão é identity: usa as coordenadas PCA atuais sem alterar nada.
Para alinhamentos lineares/afins futuros, use representation_feedback.transform
no config ou escreva o transform equivalente no meta do modelo.

Exemplo opcional no config.yaml:

representation_feedback:
  dimensions: 2
  name: "PCA"
  pad_frac: 0.10
  grid_size: 240
  density_masses: [0.50, 0.80, 0.95]
  show_labels: true
  transform:
    type: "identity"

Exemplo de transform afim (inclui rotações/Procrustes quando expresso em
coordenadas do embedding):
  transform:
    type: "affine"
    matrix: [[1.0, 0.0], [0.0, 1.0]]
    offset: [0.0, 0.0]
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


DEFAULT_DENSITY_MASSES = (0.50, 0.80, 0.95)


def _raw_config(cfg) -> Dict[str, Any]:
    return getattr(cfg, "_raw_config", {}) or {}


def feedback_config(cfg) -> Dict[str, Any]:
    raw = _raw_config(cfg)
    block = raw.get("representation_feedback", {})
    return block if isinstance(block, dict) else {}


def _normalize_transform_spec(spec: Any) -> Dict[str, Any]:
    if spec is None:
        return {"type": "identity"}

    if isinstance(spec, str):
        return {"type": spec}

    if not isinstance(spec, dict):
        raise TypeError("representation_feedback.transform deve ser string ou dict.")

    out = dict(spec)
    out["type"] = str(out.get("type", "identity")).strip().lower()
    return out


def transform_spec_from_cfg(cfg) -> Dict[str, Any]:
    block = feedback_config(cfg)
    return _normalize_transform_spec(block.get("transform", {"type": "identity"}))


def apply_representation_transform(X: np.ndarray, spec: Mapping[str, Any] | None) -> np.ndarray:
    """
    Aplica o transform que define o espaço FINAL do feedback.

    O classificador pode continuar operando no espaço original. Este transform
    existe para que mapa de treino e ponto online usem exatamente a mesma
    representação visual.
    """
    arr = np.asarray(X, dtype=float)
    was_1d = arr.ndim == 1

    if was_1d:
        arr = arr[None, :]

    if arr.ndim != 2:
        raise ValueError(f"Representação deve ser vetor/matriz 2D; shape recebido={arr.shape}")

    tr = _normalize_transform_spec(spec)
    ttype = tr.get("type", "identity").lower()

    if ttype in {"identity", "none", "raw"}:
        out = arr.copy()

    elif ttype in {"affine", "linear", "procrustes"}:
        matrix = np.asarray(tr.get("matrix"), dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Transform afim exige 'matrix' 2D.")

        # Convenção: matrix = (dim_saida, dim_entrada)
        if matrix.shape[1] != arr.shape[1]:
            raise ValueError(
                "Dimensão incompatível no transform: "
                f"X tem {arr.shape[1]} colunas, matrix espera {matrix.shape[1]}."
            )

        offset = np.asarray(tr.get("offset", np.zeros(matrix.shape[0])), dtype=float).reshape(-1)
        if offset.size != matrix.shape[0]:
            raise ValueError(
                f"offset deve ter {matrix.shape[0]} valores; recebeu {offset.size}."
            )

        scale = float(tr.get("scale", 1.0))
        out = scale * (arr @ matrix.T) + offset

    else:
        raise ValueError(
            f"Transform de representação não reconhecido: {ttype!r}. "
            "Implemente-o em representation_feedback.apply_representation_transform()."
        )

    return out[0] if was_1d else out


def _padded_limits(x: np.ndarray, pad_frac: float) -> list[float]:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return [-1.0, 1.0]

    lo = float(np.min(finite))
    hi = float(np.max(finite))

    if lo == hi:
        lo -= 1.0
        hi += 1.0

    pad = float(pad_frac) * max(hi - lo, 1e-6)
    return [lo - pad, hi + pad]


def _density_thresholds(z: np.ndarray, masses) -> list[float]:
    """
    Limiares HDR aproximados. Para cada massa (ex. 0.50, 0.80, 0.95),
    encontra o nível de densidade que contém aproximadamente aquela fração
    da massa da grade.
    """
    flat = np.asarray(z, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat >= 0)]

    if flat.size == 0 or np.sum(flat) <= 0:
        return []

    order = np.sort(flat)[::-1]
    cum = np.cumsum(order)
    cum /= cum[-1]

    thresholds = []
    for mass in masses:
        m = float(np.clip(mass, 1e-4, 0.9999))
        idx = int(np.searchsorted(cum, m, side="left"))
        idx = min(max(idx, 0), len(order) - 1)
        thresholds.append(float(order[idx]))

    # contour exige níveis estritamente crescentes.
    return sorted(set(thresholds))



def _density_threshold_by_mass(z: np.ndarray, masses) -> dict[float, float]:
    """Retorna o limiar HDR correspondente a cada massa solicitada."""
    flat = np.asarray(z, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat >= 0)]
    if flat.size == 0 or np.sum(flat) <= 0:
        return {}

    order = np.sort(flat)[::-1]
    cum = np.cumsum(order)
    cum /= cum[-1]
    out: dict[float, float] = {}
    for mass in masses:
        m = float(np.clip(mass, 1e-4, 0.9999))
        idx = int(np.searchsorted(cum, m, side="left"))
        idx = min(max(idx, 0), len(order) - 1)
        out[m] = float(order[idx])
    return out


def _decimate_polygon(points: np.ndarray, max_points: int = 180) -> list[list[float]]:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return []
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=int)
        pts = pts[idx]
    return [[round(float(x), 7), round(float(y), 7)] for x, y in pts[:, :2]]


def _contour_polygons(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, threshold: float) -> list[list[list[float]]]:
    """Extrai os polígonos do nível de densidade para uso numérico online."""
    fig, ax = plt.subplots(figsize=(2, 2))
    try:
        cs = ax.contour(xx, yy, zz, levels=[float(threshold)])
        segs = cs.allsegs[0] if getattr(cs, "allsegs", None) else []
        polygons = []
        for seg in segs:
            poly = _decimate_polygon(seg)
            if len(poly) >= 3:
                polygons.append(poly)
        return polygons
    finally:
        plt.close(fig)


def _fallback_ellipse_regions(points: np.ndarray, masses) -> tuple[list[float], list[dict[str, Any]]]:
    """Fallback robusto quando a KDE 2D é singular: elipses por distância Mahalanobis empírica."""
    pts = np.asarray(points, dtype=float)[:, :2]
    mu = np.nanmean(pts, axis=0)
    cov = np.cov(pts.T) if pts.shape[0] > 1 else np.eye(2)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        cov = np.eye(2)
    scale = max(float(np.trace(cov)) / 2.0, 1e-6)
    cov = cov + np.eye(2) * (1e-6 * scale)
    inv = np.linalg.pinv(cov)
    centered = pts - mu
    d2 = np.einsum("ni,ij,nj->n", centered, inv, centered)
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.maximum(eigval, 1e-10)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=True)
    unit = np.vstack([np.cos(theta), np.sin(theta)])

    regions = []
    for mass in masses:
        m = float(np.clip(mass, 1e-4, 0.9999))
        q = float(np.quantile(d2, m)) if d2.size else 1.0
        axes = np.sqrt(eigval * max(q, 1e-9))
        ellipse = (eigvec @ (axes[:, None] * unit)).T + mu
        regions.append({
            "mass": m,
            "density_threshold": None,
            "polygons": [_decimate_polygon(ellipse, max_points=120)],
        })
    return [float(mu[0]), float(mu[1])], regions


def build_density_region_manifest(
    X_rep: np.ndarray,
    y: np.ndarray,
    xlim,
    ylim,
    dimensions: int,
    density_masses,
    class_details: list[dict[str, Any]] | None = None,
    grid_size: int = 240,
) -> list[dict[str, Any]]:
    """Constrói regiões HDR numéricas que o controlador online pode consultar.

    Em 2D, cada região é salva como um conjunto de polígonos no mesmo sistema
    de coordenadas rep1/rep2 usado pelo Unity. Em 1D, salva intervalos.
    """
    X = np.asarray(X_rep, dtype=float)
    yy = np.asarray(y)
    details_by_class = {
        int(row.get("model_class")): dict(row)
        for row in (class_details or [])
        if row.get("model_class") is not None
    }
    masses = [float(np.clip(v, 1e-4, 0.9999)) for v in density_masses]
    out: list[dict[str, Any]] = []

    for c in np.unique(yy):
        ci = int(c)
        pts = X[yy == c, :max(1, dimensions)]
        detail = details_by_class.get(ci, {"model_class": ci})
        row = {
            "model_class": ci,
            "event_label": str(detail.get("event_label", f"CLASS_{ci}")),
            "display_label": str(detail.get("display_label", f"CLASS {ci}")),
            "control": str(detail.get("control", "")),
            "decoder_channel": str(detail.get("decoder_channel", "")),
            "marker_code": detail.get("marker_code"),
            "n_training_points": int(pts.shape[0]),
            "method": "none",
            "center": [],
            "regions": [],
        }

        if pts.shape[0] == 0:
            out.append(row)
            continue

        if dimensions <= 1:
            values = pts[:, 0]
            row["center"] = [float(np.nanmean(values)), 0.5]
            kde = _safe_kde_1d(values)
            if kde is not None:
                xs = np.linspace(float(xlim[0]), float(xlim[1]), max(160, int(grid_size)))
                dens = np.asarray(kde(xs), dtype=float)
                thresholds = _density_threshold_by_mass(dens, masses)
                row["method"] = "kde_1d"
                for mass in masses:
                    thr = thresholds.get(mass)
                    intervals = []
                    if thr is not None:
                        mask = dens >= thr
                        start = None
                        for i, flag in enumerate(mask):
                            if flag and start is None:
                                start = i
                            if start is not None and ((not flag) or i == len(mask) - 1):
                                end = i if flag and i == len(mask) - 1 else i - 1
                                if end >= start:
                                    intervals.append([round(float(xs[start]), 7), round(float(xs[end]), 7)])
                                start = None
                    row["regions"].append({
                        "mass": mass,
                        "density_threshold": None if thr is None else float(thr),
                        "intervals": intervals,
                        "polygons": [],
                    })
            out.append(row)
            continue

        pts2 = pts[:, :2]
        row["center"] = [float(np.nanmean(pts2[:, 0])), float(np.nanmean(pts2[:, 1]))]
        kde = _safe_kde_2d(pts2)
        if kde is None:
            center, regions = _fallback_ellipse_regions(pts2, masses)
            row["center"] = center
            row["method"] = "mahalanobis_ellipse_fallback"
            row["regions"] = regions
            out.append(row)
            continue

        gx = np.linspace(float(xlim[0]), float(xlim[1]), max(80, int(grid_size)))
        gy = np.linspace(float(ylim[0]), float(ylim[1]), max(80, int(grid_size)))
        xx, yy_grid = np.meshgrid(gx, gy)
        grid = np.vstack([xx.ravel(), yy_grid.ravel()])
        zz = np.asarray(kde(grid), dtype=float).reshape(xx.shape)
        thresholds = _density_threshold_by_mass(zz, masses)
        row["method"] = "gaussian_kde"
        row["peak_density"] = float(np.nanmax(zz)) if zz.size else None

        for mass in masses:
            threshold = thresholds.get(mass)
            polygons = [] if threshold is None else _contour_polygons(xx, yy_grid, zz, threshold)
            row["regions"].append({
                "mass": mass,
                "density_threshold": None if threshold is None else float(threshold),
                "polygons": polygons,
            })
        out.append(row)

    return out


def _safe_kde_2d(points: np.ndarray):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 4:
        return None

    xy = pts[:, :2].T
    if np.linalg.matrix_rank(np.cov(xy)) < 2:
        return None

    try:
        return gaussian_kde(xy)
    except Exception:
        return None


def _safe_kde_1d(values: np.ndarray):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3 or np.nanstd(v) < 1e-12:
        return None
    try:
        return gaussian_kde(v)
    except Exception:
        return None


def render_density_map(
    X_rep: np.ndarray,
    y: np.ndarray,
    out_png: str,
    xlim,
    ylim,
    dimensions: int,
    class_labels: Mapping[int, str] | None = None,
    grid_size: int = 240,
    density_masses=DEFAULT_DENSITY_MASSES,
    show_labels: bool = True,
    dpi: int = 180,
) -> None:
    """
    Gera uma textura PNG transparente, sem ticks/margens/eixos.

    A borda esquerda/direita da imagem corresponde exatamente a xlim;
    a borda inferior/superior corresponde exatamente a ylim.
    Isso permite ao Unity mapear rep1/rep2 diretamente para pixels/UI.
    """
    X = np.asarray(X_rep, dtype=float)
    yy = np.asarray(y)
    labels = class_labels or {}

    fig = plt.figure(figsize=(6.0, 6.0), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_axis_off()

    classes = np.unique(yy)
    cmap = plt.get_cmap("tab10")

    if dimensions <= 1:
        xs = np.linspace(float(xlim[0]), float(xlim[1]), max(100, int(grid_size)))
        max_density = 0.0
        curves = []

        for i, c in enumerate(classes):
            vals = X[yy == c, 0]
            kde = _safe_kde_1d(vals)
            if kde is None:
                continue
            dens = kde(xs)
            curves.append((i, c, dens))
            max_density = max(max_density, float(np.nanmax(dens)))

        max_density = max(max_density, 1e-12)
        for i, c, dens in curves:
            color = cmap(i % 10)
            yn = 0.15 + 0.70 * dens / max_density
            ax.plot(xs, yn, linewidth=2.0, color=color, alpha=0.95)
            ax.fill_between(xs, 0.15, yn, color=color, alpha=0.10)
            if show_labels:
                vals = X[yy == c, 0]
                ax.text(
                    float(np.nanmean(vals)),
                    0.90 - 0.10 * (i % 3),
                    str(labels.get(int(c), f"class {int(c)}")),
                    color=color,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    alpha=0.95,
                )
        # O eixo Y é apenas decorativo em 1D; o ponto online fica em y=0.5.
        ax.set_ylim(0.0, 1.0)

    else:
        gx = np.linspace(float(xlim[0]), float(xlim[1]), max(80, int(grid_size)))
        gy = np.linspace(float(ylim[0]), float(ylim[1]), max(80, int(grid_size)))
        xx, yy_grid = np.meshgrid(gx, gy)
        grid = np.vstack([xx.ravel(), yy_grid.ravel()])

        for i, c in enumerate(classes):
            pts = X[yy == c, :2]
            color = cmap(i % 10)
            kde = _safe_kde_2d(pts)

            if kde is not None:
                zz = kde(grid).reshape(xx.shape)
                levels = _density_thresholds(zz, density_masses)

                if levels:
                    widths = np.linspace(1.0, 2.4, len(levels))
                    ax.contour(
                        xx,
                        yy_grid,
                        zz,
                        levels=levels,
                        colors=[color],
                        linewidths=widths,
                        alpha=0.92,
                    )

            else:
                # Fallback visual se KDE ficar singular.
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=5,
                    color=color,
                    alpha=0.18,
                    linewidths=0,
                )

            if show_labels and pts.shape[0] > 0:
                mu = np.nanmean(pts[:, :2], axis=0)
                ax.text(
                    float(mu[0]),
                    float(mu[1]),
                    str(labels.get(int(c), f"class {int(c)}")),
                    color=color,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    alpha=0.95,
                )

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.savefig(out_png, transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)


def _representation_id(X_rep: np.ndarray, y: np.ndarray, transform_spec: Mapping[str, Any]) -> str:
    h = hashlib.sha1()
    x = np.asarray(X_rep, dtype=np.float64)
    yy = np.asarray(y)
    h.update(x.tobytes(order="C"))
    h.update(yy.tobytes(order="C"))
    h.update(json.dumps(dict(transform_spec), sort_keys=True, default=str).encode("utf-8"))
    return "rep_" + h.hexdigest()[:12]


def prepare_training_representation(
    X_base: np.ndarray,
    y: np.ndarray,
    cfg,
    out_dir: str,
    class_labels: Mapping[int, str] | None = None,
    class_details: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    1) aplica o transform final do feedback;
    2) calcula limites fixos;
    3) gera PNG de densidades;
    4) gera manifesto JSON para o Unity;
    5) retorna metadata a ser embutida no *_meta.json do modelo.
    """
    block = feedback_config(cfg)
    transform_spec = transform_spec_from_cfg(cfg)

    X_final = apply_representation_transform(np.asarray(X_base, dtype=float), transform_spec)
    if X_final.ndim == 1:
        X_final = X_final[:, None]

    requested_dims = int(block.get("dimensions", min(2, X_final.shape[1])))
    dimensions = max(1, min(requested_dims, X_final.shape[1], 2))

    X_display = X_final[:, :dimensions]
    pad_frac = float(block.get("pad_frac", 0.10))
    grid_size = int(block.get("grid_size", 240))
    density_masses = block.get("density_masses", list(DEFAULT_DENSITY_MASSES))
    show_labels = bool(block.get("show_labels", True))
    name = str(block.get("name", "PCA")).strip() or "Representation"

    xlim = _padded_limits(X_display[:, 0], pad_frac)
    if dimensions >= 2:
        ylim = _padded_limits(X_display[:, 1], pad_frac)
    else:
        ylim = [0.0, 1.0]

    os.makedirs(out_dir, exist_ok=True)
    map_png = os.path.join(out_dir, "pca_map.png")
    manifest_json = os.path.join(out_dir, "pca_map.json")

    labels = dict(class_labels or {0: "LEFT", 1: "RIGHT"})
    density_regions = build_density_region_manifest(
        X_display,
        y,
        xlim=xlim,
        ylim=ylim,
        dimensions=dimensions,
        density_masses=density_masses,
        class_details=list(class_details or []),
        grid_size=grid_size,
    )
    render_density_map(
        X_display,
        y,
        map_png,
        xlim=xlim,
        ylim=ylim,
        dimensions=dimensions,
        class_labels=labels,
        grid_size=grid_size,
        density_masses=density_masses,
        show_labels=show_labels,
    )

    rep_id = _representation_id(X_display, y, transform_spec)

    meta = {
        "schema_version": 1,
        "representation_id": rep_id,
        "representation_name": name,
        "dimensions": dimensions,
        "source_dimensions": int(np.asarray(X_base).shape[1]),
        "x_min": float(xlim[0]),
        "x_max": float(xlim[1]),
        "y_min": float(ylim[0]),
        "y_max": float(ylim[1]),
        "map_image": os.path.basename(map_png),
        "manifest_file": os.path.basename(manifest_json),
        "transform_type": str(transform_spec.get("type", "identity")),
        "transform": transform_spec,
        "class_labels": [str(labels.get(int(c), f"class {int(c)}")) for c in np.unique(y)],
        "classes": list(class_details or []),
        "density_masses": [float(v) for v in density_masses],
        "density_regions_in_manifest": True,
    }

    # pca_map.json é o contrato explícito entre o modelo e o feedback no Unity.
    # As chaves geométricas continuam planas para facilitar JsonUtility.
    details = list(class_details or [])
    manifest = {
        "schema_version": 3,
        "purpose": "PCA feedback map + numerical density regions for Unity/control",
        "representation_id": meta["representation_id"],
        "representation_name": meta["representation_name"],
        "dimensions": meta["dimensions"],
        "x_min": meta["x_min"],
        "x_max": meta["x_max"],
        "y_min": meta["y_min"],
        "y_max": meta["y_max"],
        "map_image": meta["map_image"],
        "transform_type": meta["transform_type"],
        "class_ids": [int(row.get("model_class", -1)) for row in details],
        "event_labels": [str(row.get("event_label", "")) for row in details],
        "class_labels": [str(row.get("display_label", "")) for row in details],
        "controls": [str(row.get("control", "")) for row in details],
        "decoder_channels": [str(row.get("decoder_channel", "")) for row in details],
        "marker_codes": [row.get("marker_code") for row in details],
        "lsl_rep_channels": ["rep1", "rep2"],
        "density_masses": [float(v) for v in density_masses],
        "density_regions": density_regions,
    }

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "X_rep": X_display,
        "meta": meta,
        "map_png": map_png,
        "manifest_json": manifest_json,
    }


def transform_online_representation(feature_vector: np.ndarray, model_meta: Mapping[str, Any] | None) -> np.ndarray:
    """
    Converte a feature/embedding online para o MESMO espaço usado no mapa.

    Para modelos antigos, sem bloco 'representation', cai automaticamente
    para identity e usa as duas primeiras coordenadas.
    """
    meta = dict(model_meta or {})
    rep_meta = meta.get("representation", {})
    if not isinstance(rep_meta, dict):
        rep_meta = {}

    spec = rep_meta.get("transform", {"type": "identity"})
    transformed = apply_representation_transform(np.asarray(feature_vector, dtype=float), spec)
    transformed = np.asarray(transformed, dtype=float).reshape(-1)

    dims = int(rep_meta.get("dimensions", min(2, transformed.size)))
    dims = max(1, min(dims, transformed.size, 2))

    out = np.zeros(2, dtype=float)
    out[:dims] = transformed[:dims]
    return out
