# graz_realtime_signal.py
import time
import math
import threading
from collections import deque

import numpy as np
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet, local_clock

# ==========================
# PARÂMETROS (ajuste se quiser)
# ==========================
# Entrada (marcadores)
MARKER_NAME = "GrazMI_Markers"
MARKER_TYPE = "Markers"

# Saída (sinal simulado)
SIGNAL_NAME = "GrazMI_SimEEG"
SIGNAL_TYPE = "EEG"

FS = 250.0          # Hz (taxa fixa de aquisição)
CHANNELS = 8        # número de canais
CHUNK = 10          # amostras por bloco (latência ~ CHUNK/FS)
NOISE_STD = 0.5     # desvio-padrão do ruído branco

# Padrão temporal (burst) disparado pelos marcadores
BURST_FREQ = 10.0   # Hz
BURST_AMP  = 1.2    # amplitude
BURST_DUR  = 1.0    # s
TAPER_FRAC = 0.2    # fração para janela Hann no início/final

# Ganhos por hemisfério (esq, dir) para cada mão (contralateral)
PROFILE_LEFT_MI  = (1.0, 0.3)  # evento RIGHT_MI_STIM (mão direita) -> hemisfério esquerdo mais forte
PROFILE_RIGHT_MI = (0.3, 1.0)  # evento LEFT_MI_STIM  (mão esquerda) -> hemisfério direito mais forte

# Códigos
CODE_LEFT_MI  = 3  # LEFT_MI_STIM
CODE_RIGHT_MI = 4  # RIGHT_MI_STIM
# ==========================


def make_signal_outlet(channels: int, fs: float, name: str, stype: str) -> StreamOutlet:
    info = StreamInfo(name, stype, channels, fs, 'float32', 'simEEG-Graz')
    chns = info.desc().append_child("channels")
    for i in range(channels):
        ch = chns.append_child("channel")
        ch.append_child_value("label", f"CH{i+1}")
        ch.append_child_value("unit", "a.u.")
        ch.append_child_value("type", "EEG")
    return StreamOutlet(info)


def resolve_marker_inlet(name: str, stype: str, retry_s: float = 1.0) -> StreamInlet:
    print(f"Procurando stream de marcadores LSL (name='{name}' ou type='{stype}')...")
    while True:
        streams = resolve_byprop('name', name, timeout=1.0)
        if not streams:
            streams = resolve_byprop('type', stype, timeout=1.0)
        if streams:
            si = streams[0]
            print(f"Conectado aos marcadores: name={si.name()}, type={si.type()}, chn={si.channel_count()}, fmt={si.channel_format()}")
            return StreamInlet(si, recover=True)
        print("  (não encontrado; tentando novamente...)")
        time.sleep(retry_s)


def build_burst(fs: float, dur: float, freq: float, amp: float, taper_frac: float) -> np.ndarray:
    """Seno com janela Hann nos primeiros/últimos 'taper_frac' da duração."""
    n = max(1, int(round(dur * fs)))
    t = np.arange(n, dtype=np.float32) / fs
    wave = amp * np.sin(2 * math.pi * freq * t)
    w = np.ones(n, dtype=np.float32)
    k = int(round(taper_frac * n))
    if k > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, k, dtype=np.float32))
        w[:k] *= ramp
        w[-k:] *= ramp[::-1]
    return (wave * w).astype(np.float32)


def hemispheric_weights(channels: int, left_gain: float, right_gain: float) -> np.ndarray:
    """Metade esquerda dos canais = left_gain; metade direita = right_gain."""
    mid = channels // 2
    w = np.empty(channels, dtype=np.float32)
    w[:mid] = left_gain
    w[mid:] = right_gain
    return w


class PatternEvent:
    """Evento ativo que adiciona um burst com pesos por canal."""
    def __init__(self, wave_1d: np.ndarray, weights_1d: np.ndarray):
        self.wave = wave_1d  # shape (T,)
        self.w = weights_1d  # shape (C,)
        self.i = 0           # índice de leitura

    def add_to(self, buf: np.ndarray):
        """Soma contribuição ao buffer (chunk, C)."""
        # quantas amostras vamos somar neste chunk
        take = min(len(self.wave) - self.i, buf.shape[0])
        if take <= 0:
            return 0
        # outer: (take, 1) * (1, C) => (take, C)
        buf[:take, :] += np.outer(self.wave[self.i:self.i + take], self.w)
        self.i += take
        return take

    def done(self) -> bool:
        return self.i >= len(self.wave)


def marker_thread(inlet: StreamInlet,
                  event_queue: deque,
                  fs: float,
                  dur: float,
                  freq: float,
                  amp: float,
                  channels: int,
                  profile_left: tuple,
                  profile_right: tuple):
    """Escuta marcadores e enfileira eventos de burst."""
    print("Listener de marcadores iniciado.")
    wave = build_burst(fs, dur, freq, amp, TAPER_FRAC)
    while True:
        try:
            samples, timestamps = inlet.pull_chunk(timeout=0.2, max_samples=32)
            if not timestamps:
                continue
            for samp, ts in zip(samples, timestamps):
                val = samp[0]
                try:
                    code = int(val) if not isinstance(val, (bytes, str)) else int(str(val).strip())
                except Exception:
                    continue

                if code == CODE_LEFT_MI:
                    # LEFT_MI_STIM -> contralateral (direita mais forte)
                    w = hemispheric_weights(channels, *profile_right)
                    event_queue.append(PatternEvent(wave, w))
                    print(f"[{ts:.3f}] LEFT_MI_STIM → evento (dir>esq).")
                elif code == CODE_RIGHT_MI:
                    # RIGHT_MI_STIM -> contralateral (esquerda mais forte)
                    w = hemispheric_weights(channels, *profile_left)
                    event_queue.append(PatternEvent(wave, w))
                    print(f"[{ts:.3f}] RIGHT_MI_STIM → evento (esq>dir).")
                # demais códigos ignorados
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Erro no listener:", e)
            time.sleep(0.1)


def streaming_loop(outlet: StreamOutlet,
                   event_queue: deque,
                   fs: float,
                   channels: int,
                   noise_std: float,
                   chunk: int):
    """Gera e envia o sinal em taxa fixa, somando ruído + eventos ativos."""
    rng = np.random.default_rng()
    active = []
    period = chunk / fs
    next_t = time.perf_counter()
    print(f"Streaming iniciado: fs={fs} Hz, canais={channels}, chunk={chunk} (latência ≈ {period*1000:.1f} ms). Ctrl+C para parar.")
    while True:
        try:
            # base: ruído branco (chunk, C)
            buf = rng.normal(0.0, noise_std, size=(chunk, channels)).astype(np.float32)

            # mover novos eventos para 'active'
            while event_queue:
                active.append(event_queue.popleft())

            # somar contribuições
            still = []
            for ev in active:
                ev.add_to(buf)
                if not ev.done():
                    still.append(ev)
            active = still

            # enviar
            outlet.push_chunk(buf.tolist())

            # agendamento com taxa fixa
            next_t += period
            dt_sleep = next_t - time.perf_counter()
            if dt_sleep > 0:
                time.sleep(dt_sleep)
            else:
                # atrasou; realinhar para evitar drift
                next_t = time.perf_counter()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Erro no streaming:", e)
            time.sleep(0.05)


def main():
    # Outlet do sinal
    outlet = make_signal_outlet(CHANNELS, FS, SIGNAL_NAME, SIGNAL_TYPE)
    print(f"Outlet pronto: name={SIGNAL_NAME}, type={SIGNAL_TYPE}, fs={FS}, ch={CHANNELS}")

    # Inlet dos marcadores (bloqueia até encontrar)
    inlet = resolve_marker_inlet(MARKER_NAME, MARKER_TYPE)

    # Fila de eventos
    event_queue = deque()

    # Thread do listener de marcadores
    t = threading.Thread(
        target=marker_thread,
        args=(inlet, event_queue, FS, BURST_DUR, BURST_FREQ, BURST_AMP, CHANNELS, PROFILE_LEFT_MI, PROFILE_RIGHT_MI),
        daemon=True
    )
    t.start()

    try:
        streaming_loop(outlet, event_queue, FS, CHANNELS, NOISE_STD, CHUNK)
    finally:
        print("Encerrando.")


if __name__ == "__main__":
    main()
