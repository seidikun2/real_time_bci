#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dashboard leve do experimentador: PCA + probabilidades contínuas.

Esta janela é propositalmente secundária ao Unity. Ela apenas consome LSL e
mostra, com baixa taxa de atualização:
  - mapa PCA/KDE estático do modelo ativo;
  - posição ATUAL de rep1/rep2;
  - p_left, p_rest, p_right;
  - classes disponíveis no modelo;
  - fase/cue mais recente do PsychoPy (opcional).

GrazMI_Control NÃO é consumido aqui. O controlador continua rodando normalmente
para o Unity, mas left_leg/right_leg/state não são desenhados nesta dashboard.

Para manter a GUI responsiva no Windows/TkAgg, os elementos estáticos são
desenhados uma vez e as partes dinâmicas usam blitting quando o backend suporta.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pylsl import StreamInlet, resolve_byprop

DECODER_STREAM_NAME = "Signal"
DECODER_STREAM_TYPE = "BCI"
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


def resolve_stream_once(name, stype, timeout=0.02):
    streams = resolve_byprop("name", name, timeout=timeout) if name else []
    if not streams and stype:
        streams = resolve_byprop("type", stype, timeout=timeout)
    if not streams:
        return None
    si = streams[0]
    print(f"Conectado opcional: {si.name()} | type={si.type()} | ch={si.channel_count()}")
    return StreamInlet(si, recover=True)


def parse_marker(sample):
    value = sample[0]
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip()
    try:
        return int(text)
    except Exception:
        inv = {v: k for k, v in CODE_MAP.items()}
        return int(inv.get(text.upper(), -1))


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
    """Curvas HDR salvas no JSON, usadas só se o PNG do mapa não existir."""
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
            for poly in region.get("polygons", []) or []:
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
    """Mapa PCA/KDE estático. Não é redesenhado durante o loop normal."""
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
            print(f"[dashboard] Não foi possível abrir {image_path}: {exc}. Usando JSON.")
    draw_density_fallback(ax, map_data)
    return "density_regions do JSON"


def parse_args():
    p = argparse.ArgumentParser(description="Dashboard leve: PCA + probabilidades")
    p.add_argument("--decoder-name", default=DECODER_STREAM_NAME)
    p.add_argument("--decoder-type", default=DECODER_STREAM_TYPE)
    p.add_argument("--marker-name", default=MARKER_STREAM_NAME)
    p.add_argument("--marker-type", default=MARKER_STREAM_TYPE)
    p.add_argument("--pca-map-json", default=None)
    p.add_argument("--pca-xlim", nargs=2, type=float, default=(-5.0, 5.0))
    p.add_argument("--pca-ylim", nargs=2, type=float, default=(-5.0, 5.0))
    p.add_argument("--plot-hz", type=float, default=5.0)
    p.add_argument("--no-markers", action="store_true")
    return p.parse_args()


def setup_plot(xlim, ylim, map_data=None, map_json_path=None):
    fig, (ax_pca, ax_prob) = plt.subplots(
        1, 2, figsize=(9.2, 3.9),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    background_name = draw_pca_background(ax_pca, map_data, map_json_path, xlim, ylim)
    # Apenas o estado atual. Não há trilha temporal no PCA: o mapa KDE já
    # fornece o contexto espacial e a ausência da trilha reduz redraw/ruído visual.
    point, = ax_pca.plot([], [], "o", markersize=9, zorder=4, animated=True)
    ax_pca.set_title("Representação PCA", pad=25)
    ax_pca.set_xlabel("rep1")
    ax_pca.set_ylabel("rep2")
    ax_pca.set_xlim(*xlim)
    ax_pca.set_ylim(*ylim)
    ax_pca.grid(True, alpha=0.18, zorder=2)
    phase_txt = ax_pca.text(
        0.5, 1.012, "fase: aguardando", transform=ax_pca.transAxes,
        ha="center", va="bottom", fontsize=8.5, animated=True,
    )
    if map_data is not None:
        ax_pca.text(
            0.01, 0.01, f"mapa: {background_name}", transform=ax_pca.transAxes,
            ha="left", va="bottom", fontsize=6.8, alpha=0.48,
        )

    prob_bars = ax_prob.bar(["LEFT", "REST", "RIGHT"], [0, 0, 0])
    for bar in prob_bars:
        bar.set_animated(True)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("predict_proba")
    ax_prob.set_title("Feedback contínuo", pad=25)
    active_txt = ax_prob.text(
        0.5, 1.012, "ativas: -", transform=ax_prob.transAxes,
        ha="center", va="bottom", fontsize=8.5, animated=True,
    )

    # Janela compacta para ficar no canto; status acima dos títulos sem sobreposição.
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.76, wspace=0.34)
    return fig, ax_pca, ax_prob, point, prob_bars, phase_txt, active_txt


def main():
    args = parse_args()
    map_data, map_json_path = load_pca_map(args.pca_map_json)
    pca_xlim, pca_ylim = map_limits(map_data, args.pca_xlim, args.pca_ylim)

    # A janela é criada ANTES de procurar o stream Signal. Isso é importante
    # porque o main.py inicia a dashboard antes do decoder online. Assim a GUI
    # aparece imediatamente e a descoberta LSL ocorre sem bloquear o Tk mainloop.
    state = {
        "decoder": None,
        "markers": None,
        "last_decoder_try": 0.0,
        "last_marker_try": 0.0,
        "current_phase": "aguardando",
        "current_rep": None,
        "latest_prob": [0.0, 0.0, 0.0],
        "latest_active": [0.0, 0.0, 0.0],
        "closed": False,
        "bg_pca": None,
        "bg_prob": None,
        "decoder_status": "Signal: aguardando",
    }

    (
        fig, ax_pca, ax_prob, point, prob_bars, phase_txt, active_txt,
    ) = setup_plot(pca_xlim, pca_ylim, map_data=map_data, map_json_path=map_json_path)

    interval_ms = max(80, int(round(1000.0 / max(args.plot_hz, 1.0))))
    dynamic_pca = [point, phase_txt]
    dynamic_prob = [*prob_bars, active_txt]
    supports_blit = bool(getattr(fig.canvas, "supports_blit", False))

    def invalidate_backgrounds():
        state["bg_pca"] = None
        state["bg_prob"] = None

    def on_draw(_event):
        # O fundo só é capturado DEPOIS que a janela já foi desenhada pelo
        # backend gráfico. Evita draw() manual antes de plt.show() no TkAgg.
        if not supports_blit or state["closed"]:
            return
        try:
            state["bg_pca"] = fig.canvas.copy_from_bbox(ax_pca.bbox)
            state["bg_prob"] = fig.canvas.copy_from_bbox(ax_prob.bbox)
        except Exception:
            invalidate_backgrounds()

    def render_dynamic():
        if not supports_blit:
            fig.canvas.draw_idle()
            return
        if state["bg_pca"] is None or state["bg_prob"] is None:
            # Solicita um draw normal; on_draw() captura o fundo para os próximos
            # frames. Não bloqueia a GUI.
            fig.canvas.draw_idle()
            return
        try:
            fig.canvas.restore_region(state["bg_pca"])
            fig.canvas.restore_region(state["bg_prob"])
            for artist in dynamic_pca:
                ax_pca.draw_artist(artist)
            for artist in dynamic_prob:
                ax_prob.draw_artist(artist)
            fig.canvas.blit(ax_pca.bbox)
            fig.canvas.blit(ax_prob.bbox)
        except Exception:
            invalidate_backgrounds()
            fig.canvas.draw_idle()

    def try_connect_decoder(now):
        if state["decoder"] is not None:
            return
        if now - state["last_decoder_try"] < 0.5:
            return
        state["last_decoder_try"] = now
        inlet = resolve_stream_once(args.decoder_name, args.decoder_type, timeout=0.01)
        if inlet is None:
            return
        n_ch = int(inlet.info().channel_count())
        if n_ch < len(SIGNAL_CHANNELS):
            print(
                f"[dashboard] Signal encontrado com {n_ch} canais; "
                f"esperados >= {len(SIGNAL_CHANNELS)}. Tentando novamente."
            )
            return
        state["decoder"] = inlet
        nominal_rate = float(inlet.info().nominal_srate() or 64.0)
        state["decoder_status"] = "Signal: conectado"
        print(
            f"[dashboard] Signal conectado | ch={n_ch} | fs={nominal_rate:.1f} Hz"
        )

    def update_dashboard():
        if state["closed"]:
            return False

        now = time.monotonic()
        try_connect_decoder(now)

        if (
            not args.no_markers
            and state["markers"] is None
            and now - state["last_marker_try"] > 1.0
        ):
            state["markers"] = resolve_stream_once(
                args.marker_name, args.marker_type, timeout=0.01
            )
            state["last_marker_try"] = now

        decoder = state["decoder"]
        if decoder is not None:
            try:
                samples, _ts = decoder.pull_chunk(timeout=0.0, max_samples=256)
                # Dashboard = monitor do estado presente. Se chegaram várias
                # amostras desde o último frame, só a mais recente interessa.
                if samples:
                    samp = samples[-1]
                    if len(samp) >= len(SIGNAL_CHANNELS):
                        state["current_rep"] = (float(samp[0]), float(samp[1]))
                        state["latest_prob"] = [float(samp[2]), float(samp[3]), float(samp[4])]
                        state["latest_active"] = [float(samp[5]), float(samp[6]), float(samp[7])]
            except Exception as exc:
                print(f"[dashboard] Signal perdido: {exc}")
                state["decoder"] = None
                state["decoder_status"] = "Signal: reconectando"

        markers = state["markers"]
        if markers is not None:
            try:
                ms, mts = markers.pull_chunk(timeout=0.0, max_samples=32)
                if mts:
                    code = parse_marker(ms[-1])
                    state["current_phase"] = CODE_MAP.get(code, f"UNKNOWN_{code}")
            except Exception:
                state["markers"] = None

        if state["current_rep"] is not None:
            x, y = state["current_rep"]
            point.set_data([x], [y])

        for bar, value, active in zip(prob_bars, state["latest_prob"], state["latest_active"]):
            bar.set_height(value)
            bar.set_alpha(1.0 if active >= 0.5 else 0.20)

        active_names = [
            name for name, flag in zip(["LEFT", "REST", "RIGHT"], state["latest_active"])
            if flag >= 0.5
        ]
        if state["decoder"] is None:
            phase_txt.set_text(state["decoder_status"])
        else:
            phase_txt.set_text(f"fase: {state['current_phase']}")
        active_txt.set_text("ativas: " + (" / ".join(active_names) if active_names else "-"))
        render_dynamic()
        return True

    def on_close(_event):
        state["closed"] = True

    def on_resize(_event):
        invalidate_backgrounds()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("resize_event", on_resize)
    fig.canvas.mpl_connect("draw_event", on_draw)

    timer = fig.canvas.new_timer(interval=interval_ms)
    timer.add_callback(update_dashboard)
    timer.start()

    print(
        f"Dashboard leve iniciada: {args.plot_hz:.1f} Hz | PCA=estado atual | "
        f"blit={'ON' if supports_blit else 'OFF'} | aguardando Signal sem bloquear GUI."
    )
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\nDashboard encerrada.")
    finally:
        state["closed"] = True
        try:
            timer.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
