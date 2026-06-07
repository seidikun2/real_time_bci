#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot em tempo real para o stream GrazMI_DecoderDebug.

Mudanças principais em relação à versão anterior:
1) limites do PCA inicializados por range de treino, sem autoscale a cada frame;
2) leitura de marcadores LSL em paralelo ao decoder e exibição da fase atual no plot;
3) atualização mais leve: evita recalcular limites e converter buffers mais do que o necessário.
"""

import argparse
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from pylsl import resolve_byprop, StreamInlet


# ===================== CONFIG =====================

DECODER_STREAM_NAME = "GrazMI_DecoderDebug"
DECODER_STREAM_TYPE = "BCI"

MARKER_STREAM_NAME  = "GrazMI_Markers"
MARKER_STREAM_TYPE  = "Markers"

TIME_WINDOW_S       = 30.0
PLOT_HZ             = 30.0  # aumentei o default; reduza se o computador engasgar

LEFT_LABEL          = "LEFT"
RIGHT_LABEL         = "RIGHT"
THRESHOLD           = 0.60

# Preencha estes valores com o range observado no PCA do treino.
# Exemplo: PCA_TRAIN_XLIM = (-7.5, 6.2), PCA_TRAIN_YLIM = (-4.0, 5.8)
# Também é possível passar por linha de comando:
#   python plot_decoder_realtime_modified.py --pca-xlim -7.5 6.2 --pca-ylim -4.0 5.8
PCA_TRAIN_XLIM      = None
PCA_TRAIN_YLIM      = None

# Fallback se o range de treino não for informado.
FALLBACK_PCA_XLIM   = (-5.0, 5.0)
FALLBACK_PCA_YLIM   = (-5.0, 5.0)
PCA_LIMIT_PAD_FRAC  = 0.10

# Mantém o range praticamente fixo, mas expande se o ponto sair da tela.
EXPAND_LIMITS_ON_OVERFLOW = True
OVERFLOW_PAD_FRAC         = 0.10
OVERFLOW_MIN_PAD          = 0.25

# Códigos usados no protocolo. Ajuste aqui se seu PsychoPy/config usar outros nomes.
CODE_MAP = {
    1: "BASELINE",
    2: "ATTENTION",
    3: "LEFT_MI_STIM",
    4: "RIGHT_MI_STIM",
    5: "ATTEMPT",
    6: "REST",
    99: "BLOCK_END",
}

STOP_ON_BLOCK_END          = False
MARKER_RETRY_INTERVAL_S    = 2.0
MARKER_PULL_MAX_SAMPLES    = 64
DECODER_PULL_MAX_SAMPLES   = 64


# ===================== LSL =====================

def resolve_stream_blocking(name, stype, timeout=2.0):
    """Resolve stream obrigatório, tentando continuamente."""
    print(f"Procurando stream LSL obrigatório: name={name}, type={stype}")

    while True:
        streams = resolve_byprop("name", name, timeout=timeout)

        if not streams:
            streams = resolve_byprop("type", stype, timeout=timeout)

        if streams:
            si = streams[0]
            print(
                f"Conectado: name={si.name()}, type={si.type()}, "
                f"chn={si.channel_count()}, fs={si.nominal_srate():.2f}"
            )
            return StreamInlet(si, recover=True)

        print("Stream obrigatório não encontrado. Tentando novamente...")
        time.sleep(1.0)


def resolve_stream_once(name, stype, timeout=0.10):
    """Resolve stream opcional sem travar o loop por muito tempo."""
    streams = resolve_byprop("name", name, timeout=timeout)

    if not streams:
        streams = resolve_byprop("type", stype, timeout=timeout)

    if not streams:
        return None

    si = streams[0]
    print(
        f"Conectado ao stream de marcadores: name={si.name()}, type={si.type()}, "
        f"chn={si.channel_count()}, fs={si.nominal_srate():.2f}"
    )
    return StreamInlet(si, recover=True)


# ===================== MARCADORES =====================

def normalize_code_map(code_map):
    out = {}
    for k, v in code_map.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            out[k] = str(v)
    return out


def parse_marker(sample, code_map):
    """Aceita marcador numérico, string numérica ou label textual."""
    val = sample[0]

    if isinstance(val, (bytes, str)):
        text = val.decode() if isinstance(val, bytes) else val
        text = text.strip()

        try:
            return int(text)
        except Exception:
            inv = {
                str(v).strip().upper(): int(k)
                for k, v in code_map.items()
                if isinstance(k, int)
            }
            return int(inv.get(text.upper(), -1))

    return int(val)


def pull_latest_marker(marker_inlet, code_map, current_phase):
    """
    Lê todos os marcadores disponíveis sem bloquear e retorna apenas o último estado.
    Isso evita fila acumulada e mantém a fase exibida o mais atual possível.
    """
    if marker_inlet is None:
        return current_phase, None, None

    samples, timestamps = marker_inlet.pull_chunk(timeout=0.0, max_samples=MARKER_PULL_MAX_SAMPLES)

    if not timestamps:
        return current_phase, None, None

    last_code = None
    last_ts   = None

    for samp, ts in zip(samples, timestamps):
        last_code = parse_marker(samp, code_map)
        last_ts   = ts

    label = code_map.get(last_code, f"UNKNOWN_{last_code}")
    return label, last_code, last_ts


# ===================== DECISÃO =====================

def get_decision(left, right, threshold=0.60):
    if left >= threshold and left > right:
        return LEFT_LABEL

    if right >= threshold and right > left:
        return RIGHT_LABEL

    return "REST / indefinido"


# ===================== LIMITES DO PCA =====================

def padded_limits(lims, pad_frac=PCA_LIMIT_PAD_FRAC):
    lo, hi = float(lims[0]), float(lims[1])

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"Limites inválidos para PCA: {lims}")

    if lo == hi:
        lo -= 1.0
        hi += 1.0

    if lo > hi:
        lo, hi = hi, lo

    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def get_initial_pca_limits(args):
    xlim = tuple(args.pca_xlim) if args.pca_xlim is not None else PCA_TRAIN_XLIM
    ylim = tuple(args.pca_ylim) if args.pca_ylim is not None else PCA_TRAIN_YLIM

    if xlim is None:
        xlim = FALLBACK_PCA_XLIM

    if ylim is None:
        ylim = FALLBACK_PCA_YLIM

    return padded_limits(xlim), padded_limits(ylim)


def maybe_expand_pca_limits(ax_pca, x, y):
    """
    Não faz autoscale contínuo. Só expande se o ponto atual sair dos limites.
    """
    if not EXPAND_LIMITS_ON_OVERFLOW:
        return

    xmin, xmax = ax_pca.get_xlim()
    ymin, ymax = ax_pca.get_ylim()

    changed = False

    if x < xmin or x > xmax:
        width = xmax - xmin
        pad   = max(OVERFLOW_MIN_PAD, OVERFLOW_PAD_FRAC * width)
        xmin  = min(xmin, x - pad)
        xmax  = max(xmax, x + pad)
        changed = True

    if y < ymin or y > ymax:
        height = ymax - ymin
        pad    = max(OVERFLOW_MIN_PAD, OVERFLOW_PAD_FRAC * height)
        ymin   = min(ymin, y - pad)
        ymax   = max(ymax, y + pad)
        changed = True

    if changed:
        ax_pca.set_xlim(xmin, xmax)
        ax_pca.set_ylim(ymin, ymax)


# ===================== PLOT =====================

def setup_plot(pca_xlim, pca_ylim):
    plt.ion()

    fig, (ax_pca, ax_dec) = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        gridspec_kw={"width_ratios": [2.0, 1.0]},
    )

    pca_trace, = ax_pca.plot([], [], "-", alpha=0.35)
    pca_point, = ax_pca.plot([], [], "o", markersize=10)

    ax_pca.axhline(0, linewidth=0.8)
    ax_pca.axvline(0, linewidth=0.8)
    ax_pca.set_title("Posição no espaço PCA")
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.set_xlim(*pca_xlim)
    ax_pca.set_ylim(*pca_ylim)
    ax_pca.set_box_aspect(1)  # mantém o eixo PCA quadrado, mesmo com layout 2:1
    ax_pca.grid(True, alpha=0.3)

    txt_phase = ax_pca.text(
        0.5,
        0.62,
        "aguardando marcador...",
        transform=ax_pca.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        alpha=0.75,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.18),
    )

    bars = ax_dec.bar(["Left", "Both", "Right"], [0, 0, 0])
    ax_dec.axhline(THRESHOLD, linestyle="--", linewidth=1.0)
    ax_dec.set_ylim(0, 1)
    ax_dec.set_title("Decodificação")
    ax_dec.set_ylabel("Valor / probabilidade")

    txt_decision = ax_dec.text(
        0.5,
        1.10,
        "Decisão: aguardando...",
        transform=ax_dec.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    return fig, ax_pca, ax_dec, pca_point, pca_trace, bars, txt_decision, txt_phase


def update_plot(ax_pca, pca_point, pca_trace, bars, txt_decision, txt_phase,
                pca1, pca2, left, both, right, current_phase):
    x_last = float(pca1[-1])
    y_last = float(pca2[-1])

    # Atualiza ponto e trilha. A conversão para array acontece só no refresh visual.
    pca_point.set_data([x_last], [y_last])
    pca_trace.set_data(
        np.fromiter(pca1, dtype=float, count=len(pca1)),
        np.fromiter(pca2, dtype=float, count=len(pca2)),
    )

    maybe_expand_pca_limits(ax_pca, x_last, y_last)

    values = (float(left[-1]), float(both[-1]), float(right[-1]))

    for bar, val in zip(bars, values):
        bar.set_height(val)

    decision = get_decision(values[0], values[2], THRESHOLD)
    txt_decision.set_text(f"Decisão: {decision}")
    txt_phase.set_text(str(current_phase))


# ===================== ARGUMENTOS =====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot em tempo real do decoder + fase atual via marcadores LSL."
    )

    parser.add_argument("--decoder-name", default=DECODER_STREAM_NAME)
    parser.add_argument("--decoder-type", default=DECODER_STREAM_TYPE)
    parser.add_argument("--marker-name", default=MARKER_STREAM_NAME)
    parser.add_argument("--marker-type", default=MARKER_STREAM_TYPE)

    parser.add_argument("--pca-xlim", nargs=2, type=float, default=None,
                        metavar=("XMIN", "XMAX"),
                        help="Range de treino para PCA1 antes do padding.")
    parser.add_argument("--pca-ylim", nargs=2, type=float, default=None,
                        metavar=("YMIN", "YMAX"),
                        help="Range de treino para PCA2 antes do padding.")
    parser.add_argument("--plot-hz", type=float, default=PLOT_HZ)
    parser.add_argument("--time-window", type=float, default=TIME_WINDOW_S)
    parser.add_argument("--no-markers", action="store_true",
                        help="Roda sem tentar conectar ao stream de marcadores.")

    return parser.parse_args()


# ===================== MAIN =====================

def main():
    args = parse_args()
    code_map = normalize_code_map(CODE_MAP)

    decoder_inlet = resolve_stream_blocking(args.decoder_name, args.decoder_type)

    marker_inlet = None
    last_marker_resolve_try = 0.0

    pca_xlim, pca_ylim = get_initial_pca_limits(args)
    fig, ax_pca, ax_dec, pca_point, pca_trace, bars, txt_decision, txt_phase = setup_plot(
        pca_xlim,
        pca_ylim,
    )

    t      = deque()
    pca1   = deque()
    pca2   = deque()
    left   = deque()
    both   = deque()
    right  = deque()

    current_phase = "sem marcador"
    last_update   = 0.0
    dt_update     = 1.0 / max(float(args.plot_hz), 1e-6)

    print("Rodando plot em tempo real. Feche a janela ou pressione Ctrl+C para sair.")
    print(f"Limites iniciais PCA: x={pca_xlim}, y={pca_ylim}")

    try:
        while plt.fignum_exists(fig.number):
            now = time.time()

            # Conecta marcadores sem bloquear o decoder caso o stream ainda não exista.
            if (not args.no_markers) and marker_inlet is None and (now - last_marker_resolve_try) >= MARKER_RETRY_INTERVAL_S:
                marker_inlet = resolve_stream_once(args.marker_name, args.marker_type, timeout=0.05)
                last_marker_resolve_try = now

            # Lê marcadores disponíveis sem bloquear.
            if marker_inlet is not None:
                current_phase, last_code, marker_ts = pull_latest_marker(marker_inlet, code_map, current_phase)

                if last_code is not None:
                    print(f"Marcador recebido: code={last_code} label={current_phase} lsl_t={marker_ts:.6f}")

                if STOP_ON_BLOCK_END and last_code == 99:
                    print("BLOCK_END recebido. Encerrando plot.")
                    break

            # Lê decoder com timeout curto. O chunk maior reduz overhead sem acumular muita latência.
            samples, timestamps = decoder_inlet.pull_chunk(timeout=0.01, max_samples=DECODER_PULL_MAX_SAMPLES)

            if timestamps:
                for samp, ts in zip(samples, timestamps):
                    if len(samp) < 5:
                        raise RuntimeError(
                            "O stream precisa ter pelo menos 5 canais: "
                            "pca1, pca2, left, both, right."
                        )

                    t.append(float(ts))
                    pca1.append(float(samp[0]))
                    pca2.append(float(samp[1]))
                    left.append(float(samp[2]))
                    both.append(float(samp[3]))
                    right.append(float(samp[4]))

                t_now = t[-1]

                while t and (t_now - t[0]) > float(args.time_window):
                    t.popleft()
                    pca1.popleft()
                    pca2.popleft()
                    left.popleft()
                    both.popleft()
                    right.popleft()

            now = time.time()

            if len(t) > 0 and (now - last_update) >= dt_update:
                update_plot(
                    ax_pca,
                    pca_point,
                    pca_trace,
                    bars,
                    txt_decision,
                    txt_phase,
                    pca1,
                    pca2,
                    left,
                    both,
                    right,
                    current_phase,
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_update = now
            else:
                # Mantém a janela responsiva com custo menor que redesenhar sempre.
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")


if __name__ == "__main__":
    main()
