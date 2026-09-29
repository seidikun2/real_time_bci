#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dashboard leve: mapa PCA estático + probabilidades + GrazMI_Control.

O mapa de fundo é carregado do pca_map.json ativo. Quando o PNG associado
existe, ele é usado diretamente (curvas KDE exatamente como na calibração).
Se o PNG não existir, as regiões numéricas em density_regions são desenhadas
como fallback. O background é criado uma única vez e não participa do loop de
atualização, para não acrescentar latência ao decoder/controlador.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pylsl import StreamInlet, resolve_byprop

DECODER_STREAM_NAME = "Signal"
DECODER_STREAM_TYPE = "BCI"
CONTROL_STREAM_NAME = "GrazMI_Control"
CONTROL_STREAM_TYPE = "BCIControl"
MARKER_STREAM_NAME = "GrazMI_Markers"
MARKER_STREAM_TYPE = "Markers"

SIGNAL_CHANNELS = (
    "rep1", "rep2", "p_left", "p_rest", "p_right",
    "active_left", "active_rest", "active_right",
)
CODE_MAP = {
    1: "BASELINE", 2: "ATTENTION", 3: "LEFT_MI_STIM", 4: "RIGHT_MI_STIM",
    5: "ATTEMPT", 6: "REST", 7: "BOTH_MI_STIM", 8: "REST_STIM", 99: "BLOCK_END",
}
STATE_NAMES = {0: "REST", 1: "LEFT", 2: "BOTH (legado)", 3: "RIGHT"}


def resolve_stream_blocking(name, stype, timeout=1.0):
    print(f"Procurando stream: name={name}, type={stype}")
    while True:
        streams = resolve_byprop("name", name, timeout=timeout) if name else []
        if not streams and stype:
            streams = resolve_byprop("type", stype, timeout=timeout)
        if streams:
            si = streams[0]
            print(f"Conectado: {si.name()} | type={si.type()} | ch={si.channel_count()}")
            return StreamInlet(si, recover=True)
        time.sleep(0.5)


def resolve_stream_once(name, stype, timeout=0.03):
    streams = resolve_byprop("name", name, timeout=timeout) if name else []
    if not streams and stype:
        streams = resolve_byprop("type", stype, timeout=timeout)
    if not streams:
        return None
    si = streams[0]
    print(f"Conectado opcional: {si.name()} | type={si.type()} | ch={si.channel_count()}")
    return StreamInlet(si, recover=True)


def channel_names(inlet):
    info = inlet.info()
    count = int(info.channel_count())
    try:
        node = info.desc().child("channels").child("channel")
        names = []
        while node is not None and node.name() == "channel":
            names.append((node.child_value("label") or "").strip())
            node = node.next_sibling()
        return [names[i] if i < len(names) and names[i] else f"ch{i}" for i in range(count)]
    except Exception:
        return [f"ch{i}" for i in range(count)]


def parse_marker(sample):
    value = sample[0]
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip()
    try:
        return int(text)
    except Exception:
        inv = {v: k for k, v in CODE_MAP.items()}
        return int(inv.get(text.upper(), -1))


def padded(lims, frac=0.10):
    lo, hi = map(float, lims)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = frac * (hi - lo)
    return lo - pad, hi + pad


def load_pca_map(path):
    if not path:
        return None, None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        print(f"[dashboard] pca_map.json não encontrado: {p}")
        return None, None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        print(f"[dashboard] Mapa PCA: {p}")
        return data, p
    except Exception as exc:
        print(f"[dashboard] Falha ao ler {p}: {exc}")
        return None, None


def map_limits(map_data, xlim, ylim):
    if not isinstance(map_data, dict):
        return tuple(xlim), tuple(ylim)
    try:
        xx = (float(map_data["x_min"]), float(map_data["x_max"]))
        yy = (float(map_data["y_min"]), float(map_data["y_max"]))
        if xx[0] < xx[1] and yy[0] < yy[1]:
            return xx, yy
    except Exception:
        pass
    return tuple(xlim), tuple(ylim)


def _map_image_path(map_data, map_json_path):
    if not isinstance(map_data, dict) or map_json_path is None:
        return None
    value = str(map_data.get("map_image", "")).strip()
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = map_json_path.parent / p
        if p.exists():
            return p
    p = map_json_path.with_suffix(".png")
    return p if p.exists() else None


def _display_label(row):
    label = str(row.get("display_label", "")).strip()
    if label:
        return label
    event = str(row.get("event_label", "")).strip().upper()
    return {
        "LEFT_MI_STIM": "LEFT", "LEFT": "LEFT",
        "RIGHT_MI_STIM": "RIGHT", "RIGHT": "RIGHT",
        "REST_STIM": "REST", "REST": "REST",
    }.get(event, event or "CLASS")


def draw_density_fallback(ax, map_data):
    """Desenha as curvas HDR salvas no JSON quando o PNG não está disponível."""
    rows = map_data.get("density_regions", []) if isinstance(map_data, dict) else []
    cmap = plt.get_cmap("tab10")
    for i, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        color = cmap(i % 10)
        regions = [r for r in row.get("regions", []) or [] if isinstance(r, dict)]
        regions.sort(key=lambda r: float(r.get("mass", 0.0)), reverse=True)
        for j, region in enumerate(regions):
            mass = float(region.get("mass", 0.0))
            polygons = region.get("polygons", []) or []
            for poly in polygons:
                pts = np.asarray(poly, dtype=float)
                if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                    continue
                ax.plot(
                    pts[:, 0], pts[:, 1], color=color,
                    linewidth=1.1 + 0.6 * (1.0 - min(max(mass, 0.0), 1.0)),
                    alpha=0.42 + 0.18 * (1.0 - min(max(mass, 0.0), 1.0)),
                    zorder=0.8,
                )
                if j == 0:
                    ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.025, zorder=0.3)
        center = row.get("center", []) or []
        if len(center) >= 2:
            try:
                ax.text(
                    float(center[0]), float(center[1]), _display_label(row),
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=color, alpha=0.80, zorder=1.0,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.5),
                )
            except Exception:
                pass


def draw_pca_background(ax, map_data, map_json_path, xlim, ylim):
    """Cria o mapa estático uma única vez antes do loop de atualização."""
    if not isinstance(map_data, dict):
        return "sem mapa"
    image_path = _map_image_path(map_data, map_json_path)
    if image_path is not None:
        try:
            img = plt.imread(str(image_path))
            ax.imshow(
                img,
                extent=[float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1])],
                origin="upper", aspect="auto", interpolation="bilinear",
                alpha=0.72, zorder=0,
            )
            return image_path.name
        except Exception as exc:
            print(f"[dashboard] Não foi possível abrir {image_path}: {exc}. Usando polígonos do JSON.")
    draw_density_fallback(ax, map_data)
    return "density_regions do JSON"


def parse_args():
    p = argparse.ArgumentParser(description="Dashboard PCA + probabilidades + GrazMI_Control")
    p.add_argument("--decoder-name", default=DECODER_STREAM_NAME)
    p.add_argument("--decoder-type", default=DECODER_STREAM_TYPE)
    p.add_argument("--control-name", default=CONTROL_STREAM_NAME)
    p.add_argument("--control-type", default=CONTROL_STREAM_TYPE)
    p.add_argument("--marker-name", default=MARKER_STREAM_NAME)
    p.add_argument("--marker-type", default=MARKER_STREAM_TYPE)
    p.add_argument("--pca-map-json", default=None)
    p.add_argument("--pca-xlim", nargs=2, type=float, default=(-5.0, 5.0))
    p.add_argument("--pca-ylim", nargs=2, type=float, default=(-5.0, 5.0))
    p.add_argument("--plot-hz", type=float, default=15.0)
    p.add_argument("--time-window", type=float, default=20.0)
    p.add_argument("--no-markers", action="store_true")
    return p.parse_args()


def setup_plot(xlim, ylim, map_data=None, map_json_path=None):
    plt.ion()
    fig, (ax_pca, ax_prob, ax_ctrl) = plt.subplots(
        1, 3, figsize=(14.2, 5.15),
        gridspec_kw={"width_ratios": [1.6, 1.0, 1.0]},
    )

    background_name = draw_pca_background(ax_pca, map_data, map_json_path, xlim, ylim)
    trace, = ax_pca.plot([], [], "-", alpha=0.38, linewidth=1.4, zorder=3)
    point, = ax_pca.plot([], [], "o", markersize=9, zorder=4)
    ax_pca.set_title("Representação PCA", pad=30)
    ax_pca.set_xlabel("rep1")
    ax_pca.set_ylabel("rep2")
    ax_pca.set_xlim(*xlim)
    ax_pca.set_ylim(*ylim)
    ax_pca.grid(True, alpha=0.22, zorder=2)
    phase_txt = ax_pca.text(
        0.5, 1.015, "fase: aguardando", transform=ax_pca.transAxes,
        ha="center", va="bottom", fontsize=9,
    )
    if map_data is not None:
        ax_pca.text(
            0.01, 0.01, f"mapa: {background_name}", transform=ax_pca.transAxes,
            ha="left", va="bottom", fontsize=7, alpha=0.55,
        )

    prob_bars = ax_prob.bar(["LEFT", "REST", "RIGHT"], [0, 0, 0])
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("predict_proba")
    ax_prob.set_title("Feedback contínuo", pad=30)
    active_txt = ax_prob.text(
        0.5, 1.015, "ativas: -", transform=ax_prob.transAxes,
        ha="center", va="bottom", fontsize=9,
    )

    leg_bars = ax_ctrl.bar(["left_leg", "right_leg"], [0, 0])
    ax_ctrl.set_ylim(0, 1)
    ax_ctrl.set_title("Controle do avatar", pad=30)
    state_txt = ax_ctrl.text(
        0.5, 1.015, "state: aguardando", transform=ax_ctrl.transAxes,
        ha="center", va="bottom", fontsize=9,
    )
    detail_txt = ax_ctrl.text(
        0.5, 0.96, "confidence=-", transform=ax_ctrl.transAxes,
        ha="center", va="top", fontsize=9,
    )

    # Espaço superior maior: os três status ficam separados dos títulos.
    # Evita tight_layout, que recolocava os eixos para cima e gerava sobreposição.
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.14, top=0.80, wspace=0.34)
    return fig, trace, point, prob_bars, leg_bars, phase_txt, active_txt, state_txt, detail_txt


def main():
    args = parse_args()
    map_data, map_json_path = load_pca_map(args.pca_map_json)
    pca_xlim, pca_ylim = map_limits(map_data, args.pca_xlim, args.pca_ylim)

    decoder = resolve_stream_blocking(args.decoder_name, args.decoder_type)
    if int(decoder.info().channel_count()) < len(SIGNAL_CHANNELS):
        raise RuntimeError(f"Signal/BCI precisa de {len(SIGNAL_CHANNELS)} canais: {', '.join(SIGNAL_CHANNELS)}")

    control = None
    markers = None
    last_control_try = 0.0
    last_marker_try = 0.0
    control_idx = {}
    current_phase = "sem marcador"
    ctrl = {"left_leg": 0.0, "right_leg": 0.0, "state_id": 0.0, "confidence": 0.0, "density_gate": 0.0}

    t, rep1, rep2 = deque(), deque(), deque()
    latest_prob = [0.0, 0.0, 0.0]
    latest_active = [0.0, 0.0, 0.0]

    fig, trace, point, prob_bars, leg_bars, phase_txt, active_txt, state_txt, detail_txt = setup_plot(
        pca_xlim, pca_ylim, map_data=map_data, map_json_path=map_json_path,
    )
    last_draw = 0.0
    period = 1.0 / max(args.plot_hz, 1.0)
    print("Dashboard em tempo real. O background PCA é estático e o plot apenas consome LSL.")

    try:
        while plt.fignum_exists(fig.number):
            now = time.monotonic()
            if control is None and now - last_control_try > 1.0:
                control = resolve_stream_once(args.control_name, args.control_type)
                last_control_try = now
                if control is not None:
                    names = channel_names(control)
                    control_idx = {name: i for i, name in enumerate(names)}
            if not args.no_markers and markers is None and now - last_marker_try > 1.0:
                markers = resolve_stream_once(args.marker_name, args.marker_type)
                last_marker_try = now

            samples, ts = decoder.pull_chunk(timeout=0.005, max_samples=128)
            for samp, stamp in zip(samples, ts):
                if len(samp) < len(SIGNAL_CHANNELS):
                    continue
                t.append(float(stamp)); rep1.append(float(samp[0])); rep2.append(float(samp[1]))
                latest_prob = [float(samp[2]), float(samp[3]), float(samp[4])]
                latest_active = [float(samp[5]), float(samp[6]), float(samp[7])]
            if t:
                while t and t[-1] - t[0] > args.time_window:
                    t.popleft(); rep1.popleft(); rep2.popleft()

            if control is not None:
                cs, cts = control.pull_chunk(timeout=0.0, max_samples=128)
                if cts:
                    samp = cs[-1]
                    for key in ctrl:
                        idx = control_idx.get(key)
                        if idx is not None and idx < len(samp):
                            ctrl[key] = float(samp[idx])

            if markers is not None:
                ms, mts = markers.pull_chunk(timeout=0.0, max_samples=64)
                if mts:
                    code = parse_marker(ms[-1])
                    current_phase = CODE_MAP.get(code, f"UNKNOWN_{code}")

            if t and now - last_draw >= period:
                x = np.asarray(rep1, float); y = np.asarray(rep2, float)
                trace.set_data(x, y); point.set_data([x[-1]], [y[-1]])
                for bar, value, active in zip(prob_bars, latest_prob, latest_active):
                    bar.set_height(value)
                    bar.set_alpha(1.0 if active >= 0.5 else 0.25)
                for bar, value in zip(leg_bars, [ctrl["left_leg"], ctrl["right_leg"]]):
                    bar.set_height(value)

                active_names = [name for name, flag in zip(["LEFT", "REST", "RIGHT"], latest_active) if flag >= 0.5]
                phase_txt.set_text(f"fase: {current_phase}")
                active_txt.set_text("ativas: " + (" / ".join(active_names) if active_names else "-"))
                sid = int(round(ctrl["state_id"]))
                state_txt.set_text(f"state: {STATE_NAMES.get(sid, sid)}")
                detail_txt.set_text(f"confidence={ctrl['confidence']:.2f} | gate={ctrl['density_gate']:.0f}")

                fig.canvas.draw_idle(); fig.canvas.flush_events(); last_draw = now
            else:
                fig.canvas.flush_events()
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nDashboard encerrada.")


if __name__ == "__main__":
    main()
