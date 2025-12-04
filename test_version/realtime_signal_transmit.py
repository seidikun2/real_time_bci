# transmit.py
import threading
from collections import deque
import time
import math
import numpy as np
import os
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet

from config_models import AppConfig


def make_outlet(cfg: AppConfig) -> StreamOutlet:
    info = StreamInfo(
        cfg.lsl.signal_name,
        cfg.lsl.signal_type,
        cfg.sim_signal.channels,
        cfg.sim_signal.fs,
        'float32',
        'simEEG'
    )
    ch = info.desc().append_child("channels")
    for i in range(cfg.sim_signal.channels):
        ch.append_child("channel").append_child_value("label", f"CH{i+1}")
    return StreamOutlet(info)


def find_markers(cfg: AppConfig) -> StreamInlet:
    while True:
        s = (resolve_byprop('name', cfg.lsl.marker_name, timeout=1.0) or
             resolve_byprop('type', cfg.lsl.marker_type, timeout=1.0))
        if s:
            return StreamInlet(s[0], recover=True)
        print("Aguardando marcadores...")

def burst(cfg: AppConfig) -> np.ndarray:
    FS = cfg.sim_signal.fs
    n  = int(cfg.sim_burst.dur * FS)
    t  = np.arange(n) / FS
    w  = cfg.sim_burst.amp * np.sin(2 * math.pi * cfg.sim_burst.freq * t)
    k  = int(cfg.sim_burst.taper_frac * n)

    if k > 0:
        ramp   = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, k))
        w[:k] *= ramp
        w[-k:] *= ramp[::-1]
    return w.astype(np.float32)


def weights(cfg: AppConfig, l: float, r: float) -> np.ndarray:
    CHANNELS = cfg.sim_signal.channels
    mid = CHANNELS // 2
    v   = np.empty(CHANNELS, np.float32)
    v[:mid]  = l
    v[mid:]  = r
    return v


class Event:
    def __init__(self, wave: np.ndarray, w: np.ndarray):
        self.wave = wave
        self.w    = w
        self.i    = 0

    def add(self, x: np.ndarray):
        k = min(len(self.wave) - self.i, x.shape[0])
        if k > 0:
            x[:k] += np.outer(self.wave[self.i:self.i + k], self.w)
        self.i += k

    def done(self) -> bool:
        return self.i >= len(self.wave)


def open_logs(cfg: AppConfig, mode: str):
    """
    mode: "train" ou "realtime" (define nomes dos arquivos).
    """
    subject = cfg.experiment.subject_id
    session = cfg.experiment.session_id
    session_type = cfg.experiment.session_type

    # ex.: C:/.../Dados/SY100/S1/train/
    log_dir = os.path.join(
        cfg.experiment.log_root,
        subject,
        f"S{session}",
        session_type,
        mode
    )
    os.makedirs(log_dir, exist_ok=True)

    signal_csv = os.path.join(log_dir, f"{mode}_signal.csv")
    marker_csv = os.path.join(log_dir, f"{mode}_markers.csv")

    sig = open(signal_csv, "w")
    mrk = open(marker_csv, "w")

    sig.write("t," + ",".join(f"ch{i+1}" for i in range(cfg.sim_signal.channels)) + "\n")
    mrk.write("t,code,label\n")
    return sig, mrk


def marker_thread(cfg: AppConfig, inlet, q, mrk, stop_event: threading.Event):
    last = None
    wave = burst(cfg)
    codes = cfg.codes
    code_map = cfg.codes.code_map

    while not stop_event.is_set():
        samples, timestamps = inlet.pull_chunk(timeout=0.2)
        if not timestamps:
            continue

        for samp, ts in zip(samples, timestamps):
            try:
                c = int(samp[0])
            except Exception:
                continue

            if c == codes.left_mi:
                last = "L"
            if c == codes.right_mi:
                last = "R"
            if c == codes.attempt and last:
                if last == "L":
                    l, r = cfg.sim_profiles.right_mi
                else:
                    l, r = cfg.sim_profiles.left_mi
                w = weights(cfg, l, r)
                q.append(Event(wave, w))

            label = code_map.get(c, "UNK")
            try:
                mrk.write(f"{time.time():.3f},{c},{label}\n")
            except ValueError:
                print("[marker_thread] Arquivo de marcadores já fechado. Encerrando thread.")
                return

def stream(cfg: AppConfig, outlet, q, sig, stop_event: threading.Event):
    rng = np.random.default_rng()
    ev  = []
    FS     = cfg.sim_signal.fs
    CHUNK  = cfg.sim_signal.chunk
    NOISE  = cfg.sim_signal.noise_std
    dt  = CHUNK / FS

    while not stop_event.is_set():
        x = rng.normal(0, NOISE, (CHUNK, cfg.sim_signal.channels)).astype(np.float32)

        # injeta eventos pendentes
        while q:
            ev.append(q.popleft())
        keep = []
        for e in ev:
            e.add(x)
            if not e.done():
                keep.append(e)
        ev = keep

        outlet.push_chunk(x.tolist())

        t0 = time.time()
        for i in range(CHUNK):
            sig.write(f"{t0 + i/FS:.3f}," + ",".join(f"{v:.4f}" for v in x[i]) + "\n")

        time.sleep(dt)



def run_transmission(cfg: AppConfig, mode: str = "train", stop_event: threading.Event | None = None):
    """
    Função de alto nível chamada pelo programa principal.
    `mode` diferencia treino/tempo-real nos nomes de arquivos.
    `stop_event` controla quando parar.
    """
    if stop_event is None:
        stop_event = threading.Event()

    outlet   = make_outlet(cfg)
    inlet    = find_markers(cfg)
    sig, mrk = open_logs(cfg, mode=mode)
    q        = deque()

    th = threading.Thread(
        target=marker_thread,
        args=(cfg, inlet, q, mrk, stop_event),
        daemon=True
    )
    th.start()

    try:
        stream(cfg, outlet, q, sig, stop_event)
    finally:
        # Quando stop_event for setado, saímos do loop e fechamos os arquivos
        try:
            sig.close()
        except Exception:
            pass
        try:
            mrk.close()
        except Exception:
            pass