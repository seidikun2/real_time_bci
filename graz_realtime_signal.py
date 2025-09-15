# graz_realtime_signal.py
import os
import time
import math
import threading
from collections import deque
import datetime as dt

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

FS          = 250.0      # Hz (taxa fixa de aquisição)
CHANNELS    = 8          # número de canais
CHUNK       = 10         # amostras por bloco (latência ~ CHUNK/FS)
NOISE_STD   = 0.5        # desvio-padrão do ruído branco

# Padrão temporal (burst) disparado pelos marcadores
BURST_FREQ = 10.0        # Hz
BURST_AMP  = 1.5         # amplitude
BURST_DUR  = 3.5         # s
TAPER_FRAC = 0.2         # fração para janela Hann no início/final

# Ganhos por hemisfério (esq, dir) para cada mão (contralateral)
PROFILE_LEFT_MI  = (1.0, 0.3)  # evento RIGHT_MI_STIM (mão direita)  -> hemisfério ESQUERDO mais forte
PROFILE_RIGHT_MI = (0.3, 1.0)  # evento LEFT_MI_STIM  (mão esquerda) -> hemisfério DIREITO  mais forte

# Códigos
CODE_LEFT_MI   = 3  # LEFT_MI_STIM
CODE_RIGHT_MI  = 4  # RIGHT_MI_STIM
CODE_ATTEMPT   = 5  # <<< ALTERADO: constante explícita

# Mapa de rótulos (para log legível)
CODE_MAP = {
    1: "BASELINE",
    2: "ATTENTION",
    3: "LEFT_MI_STIM",
    4: "RIGHT_MI_STIM",
    5: "ATTEMPT",
    6: "REST",
}

# ===== Logs em disco =====
LOG_DIR = r"C:\Users\Unifesp\Desktop\Dados Seidi"   # pasta de saída
FLUSH_EVERY_N_SIGNAL_ROWS = 100            # flush periódico do CSV do sinal
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
        take = min(len(self.wave) - self.i, buf.shape[0])
        if take <= 0:
            return 0
        buf[:take, :] += np.outer(self.wave[self.i:self.i + take], self.w)
        self.i += take
        return take

    def done(self) -> bool:
        return self.i >= len(self.wave)


def open_csv_writers():
    import csv
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sig_path = os.path.join(LOG_DIR, f"graz_sim_signal_{stamp}.csv")
    mrk_path = os.path.join(LOG_DIR, f"graz_sim_markers_{stamp}.csv")
    # sinal
    fS = open(sig_path, "w", newline="", encoding="utf-8")
    wS = csv.writer(fS)
    header_sig = ["iso_time", "lsl_time_s"] + [f"ch{i+1}" for i in range(CHANNELS)]
    wS.writerow(header_sig)
    # marcadores
    fM = open(mrk_path, "w", newline="", encoding="utf-8")
    wM = csv.writer(fM)
    wM.writerow(["iso_time", "lsl_time_s", "code", "label", "local_recv_iso", "local_recv_s"])
    print(f"Log (sinal):   {sig_path}")
    print(f"Log (markers): {mrk_path}")
    return fS, wS, sig_path, fM, wM, mrk_path


def marker_thread(inlet: StreamInlet, event_queue:deque, channels:int, profile_left:tuple,
                  profile_right: tuple, wM, lockM, unix_offset: float):
    """Escuta marcadores e enfileira eventos apenas quando chega ATTEMPT, usando a última pista (LEFT/RIGHT)."""
    print("Listener de marcadores iniciado.")
    wave = build_burst(FS, BURST_DUR, BURST_FREQ, BURST_AMP, TAPER_FRAC)

    last_stim = None  # <<< ALTERADO: lembra última pista ("LEFT" ou "RIGHT")

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

                # Atualiza memória com a última pista
                if code == CODE_LEFT_MI:
                    last_stim = "LEFT"   # <<< ALTERADO
                elif code == CODE_RIGHT_MI:
                    last_stim = "RIGHT"  # <<< ALTERADO

                # Dispara burst somente no ATTEMPT, com base na última pista
                elif code == CODE_ATTEMPT:  # <<< ALTERADO
                    if last_stim == "LEFT":
                        # ATTEMPT da mão ESQUERDA -> contralateral (direita mais forte)
                        w = hemispheric_weights(channels, *profile_right)
                        event_queue.append(PatternEvent(wave, w))
                        print(f"[{ts:.3f}] ATTEMPT após LEFT_MI_STIM → evento (dir>esq).")
                    elif last_stim == "RIGHT":
                        # ATTEMPT da mão DIREITA -> contralateral (esquerda mais forte)
                        w = hemispheric_weights(channels, *profile_left)
                        event_queue.append(PatternEvent(wave, w))
                        print(f"[{ts:.3f}] ATTEMPT após RIGHT_MI_STIM → evento (esq>dir).")
                    else:
                        # ATTEMPT sem pista anterior: ignora ou use default (aqui: ignora)
                        print(f"[{ts:.3f}] ATTEMPT sem pista anterior — ignorado.")

                # Log do marcador (mais precisão)
                label = CODE_MAP.get(code, "UNKNOWN")
                local_now = time.time()
                iso_mrk  = dt.datetime.fromtimestamp(ts + unix_offset).isoformat(timespec="microseconds")
                local_iso = dt.datetime.fromtimestamp(local_now).isoformat(timespec="microseconds")
                with lockM:
                    wM.writerow([iso_mrk, f"{ts:.9f}", code, label, local_iso, f"{local_now:.9f}"])

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Erro no listener:", e)
            time.sleep(0.1)


def streaming_loop(outlet: StreamOutlet, event_queue: deque, wS, lockS, unix_offset: float):
    """
    Gera e envia o sinal em taxa fixa, somando ruído + eventos ativos, e salva no CSV do sinal.
    """
    rng = np.random.default_rng()
    active = []
    period = CHUNK / FS

    # timestamps estáveis: ancora no início e usa contador de amostras (float64)
    start_lsl = local_clock()
    n_sent = 0
    next_t = time.perf_counter()

    n_rows_since_flush = 0
    print(f"Streaming iniciado: fs={FS} Hz, canais={CHANNELS}, chunk={CHUNK} (latência ≈ {period*1000:.1f} ms). Ctrl+C para parar.")
    while True:
        try:
            # ruído base
            buf = rng.normal(0.0, NOISE_STD, size=(CHUNK, CHANNELS)).astype(np.float32)

            # adicionar novos eventos
            while event_queue:
                active.append(event_queue.popleft())

            # somar contribuições dos eventos ativos
            still = []
            for ev in active:
                ev.add_to(buf)
                if not ev.done():
                    still.append(ev)
            active = still

            # timestamps LSL por amostra (alta resolução)
            idx = n_sent + np.arange(CHUNK, dtype=np.float64)
            ts_vec = start_lsl + (idx / FS)
            n_sent += CHUNK

            # envia chunk com timestamps
            outlet.push_chunk(buf.tolist(), ts_vec.tolist())

            # log do sinal
            rows = []
            for i in range(CHUNK):
                iso = dt.datetime.fromtimestamp(ts_vec[i] + unix_offset).isoformat(timespec="microseconds")
                row = [iso, f"{ts_vec[i]:.9f}"] + [f"{float(v):.6f}" for v in buf[i, :]]
                rows.append(row)
            with lockS:
                wS.writerows(rows)
            n_rows_since_flush += CHUNK
            if n_rows_since_flush >= FLUSH_EVERY_N_SIGNAL_ROWS:
                try:
                    wS.writerow([])
                except Exception:
                    pass
                n_rows_since_flush = 0

            # agendamento com taxa fixa (perf_counter)
            next_t += period
            dt_sleep = next_t - time.perf_counter()
            if dt_sleep > 0:
                time.sleep(dt_sleep)
            else:
                next_t = time.perf_counter()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Erro no streaming:", e)
            time.sleep(0.05)

def main():
    # Outlet do sinal
    outlet      = make_signal_outlet(CHANNELS, FS, SIGNAL_NAME, SIGNAL_TYPE)
    print(f"Outlet pronto: name={SIGNAL_NAME}, type={SIGNAL_TYPE}, fs={FS}, ch={CHANNELS}")

    # Inlet dos marcadores (bloqueia até encontrar)
    inlet       = resolve_marker_inlet(MARKER_NAME, MARKER_TYPE)

    # Conversão LSL -> Unix (aprox.): Unix ≈ LSL + offset
    unix_offset = time.time() - local_clock()

    # Abertura de CSVs
    fS, wS, sig_path, fM, wM, mrk_path = open_csv_writers()
    lockS       = threading.Lock()
    lockM       = threading.Lock()

    # Fila de eventos
    event_queue = deque()

    # Thread do listener de marcadores (gera eventos + loga)
    t = threading.Thread(
        target=marker_thread,
        args=(inlet, event_queue, CHANNELS, PROFILE_LEFT_MI, PROFILE_RIGHT_MI, wM, lockM, unix_offset),
        daemon=True
    )
    t.start()

    try:
        streaming_loop(outlet, event_queue, wS, lockS, unix_offset)
    finally:
        print("\nEncerrando.")
        try: fS.close()
        except Exception: pass
        try: fM.close()
        except Exception: pass
        print(f"Arquivos salvos:\n - {sig_path}\n - {mrk_path}")

if __name__ == "__main__":
    main()
