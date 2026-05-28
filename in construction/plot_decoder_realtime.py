#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from pylsl import resolve_byprop, StreamInlet


# ===================== CONFIG =====================

STREAM_NAME     = "GrazMI_DecoderDebug"
STREAM_TYPE     = "BCI"
TIME_WINDOW_S   = 30.0
PLOT_HZ         = 15.0

LEFT_LABEL      = "LEFT"
RIGHT_LABEL     = "RIGHT"
THRESHOLD       = 0.60


# ===================== LSL =====================

def resolve_stream(name, stype, timeout=2.0):
    print(f"Procurando stream LSL: name={name}, type={stype}")

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

        print("Stream não encontrado. Tentando novamente...")
        time.sleep(1.0)


# ===================== DECISÃO =====================

def get_decision(left, right, threshold=0.60):
    if left >= threshold and left > right:
        return LEFT_LABEL

    if right >= threshold and right > left:
        return RIGHT_LABEL

    return "REST / indefinido"


# ===================== PLOT =====================

def setup_plot():
    plt.ion()

    fig, (ax_pca, ax_dec) = plt.subplots(
        1,
        2,
        figsize=(11, 4),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )

    pca_point, = ax_pca.plot([], [], "o", markersize=10)
    pca_trace, = ax_pca.plot([], [], "-", alpha=0.4)

    ax_pca.axhline(0, linewidth=0.8)
    ax_pca.axvline(0, linewidth=0.8)
    ax_pca.set_title("Posição no espaço PCA")
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.grid(True, alpha=0.3)

    bars = ax_dec.bar(["Left", "Both", "Right"], [0, 0, 0])
    ax_dec.axhline(THRESHOLD, linestyle="--", linewidth=1.0)
    ax_dec.set_ylim(0, 1)
    ax_dec.set_title("Decodificação")
    ax_dec.set_ylabel("Valor / probabilidade")

    txt = ax_dec.text(
        0.5,
        1.08,
        "Aguardando...",
        transform=ax_dec.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()

    return fig, ax_pca, ax_dec, pca_point, pca_trace, bars, txt


def update_plot(ax_pca, pca_point, pca_trace, bars, txt, t, pca1, pca2, left, both, right):
    pca_point.set_data([pca1[-1]], [pca2[-1]])
    pca_trace.set_data(pca1, pca2)

    x_all = np.asarray(pca1)
    y_all = np.asarray(pca2)

    if len(x_all) > 1:
        x_pad = max(0.5, 0.15 * (np.nanmax(x_all) - np.nanmin(x_all) + 1e-9))
        y_pad = max(0.5, 0.15 * (np.nanmax(y_all) - np.nanmin(y_all) + 1e-9))

        ax_pca.set_xlim(np.nanmin(x_all) - x_pad, np.nanmax(x_all) + x_pad)
        ax_pca.set_ylim(np.nanmin(y_all) - y_pad, np.nanmax(y_all) + y_pad)

    values = [left[-1], both[-1], right[-1]]

    for bar, val in zip(bars, values):
        bar.set_height(float(val))

    decision = get_decision(left[-1], right[-1], THRESHOLD)
    txt.set_text(f"Decisão: {decision}")

    plt.pause(0.001)


# ===================== MAIN =====================

def main():
    inlet = resolve_stream(STREAM_NAME, STREAM_TYPE)

    t      = deque()
    pca1   = deque()
    pca2   = deque()
    left   = deque()
    both   = deque()
    right  = deque()

    fig, ax_pca, ax_dec, pca_point, pca_trace, bars, txt = setup_plot()

    last_update = 0.0
    dt_update   = 1.0 / PLOT_HZ

    print("Rodando plot em tempo real. Feche a janela ou pressione Ctrl+C para sair.")

    try:
        while plt.fignum_exists(fig.number):
            samples, timestamps = inlet.pull_chunk(timeout=0.05, max_samples=32)

            if timestamps:
                for samp, ts in zip(samples, timestamps):
                    if len(samp) < 5:
                        raise RuntimeError(
                            "O stream precisa ter 5 canais: "
                            "pca1, pca2, left, both, right."
                        )

                    t.append(float(ts))
                    pca1.append(float(samp[0]))
                    pca2.append(float(samp[1]))
                    left.append(float(samp[2]))
                    both.append(float(samp[3]))
                    right.append(float(samp[4]))

                t_now = t[-1]

                while t and (t_now - t[0]) > TIME_WINDOW_S:
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
                    txt,
                    list(t),
                    list(pca1),
                    list(pca2),
                    list(left),
                    list(both),
                    list(right),
                )
                last_update = now

            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")


if __name__ == "__main__":
    main()
