# receive.py
import os
import csv
import time
import datetime as dt
from typing import Tuple
import threading


from pylsl import resolve_byprop, StreamInlet, local_clock, StreamInfo

from config_models import AppConfig


def resolve_stream(name: str, stype: str, timeout: float = 4.0) -> "StreamInfo":
    """Resolve por name e, se falhar, por type."""
    streams = resolve_byprop('name', name, timeout=timeout)
    if not streams:
        streams = resolve_byprop('type', stype, timeout=timeout)
    if not streams:
        raise RuntimeError(
            f"Nenhum stream LSL encontrado (name='{name}' / type='{stype}')."
        )
    si = streams[0]
    print(
        f"Conectado: name={si.name()}, type={si.type()}, "
        f"chn={si.channel_count()}, fmt={si.channel_format()}, "
        f"fs={si.nominal_srate():.2f}"
    )
    return si


def _make_log_dir(cfg: AppConfig, mode: str) -> str:
    """
    Define diretório padrão para logs com base em:
      log_root / subject / S{session} / session_type / mode
    Ex.: C:/.../Dados/SY100/S1/IM_treino/train/
    """
    return os.path.join(
        cfg.experiment.log_root,
        cfg.experiment.subject_id,
        f"S{cfg.experiment.session_id}",
        cfg.experiment.session_type,
        mode,
    )


def open_marker_logger(cfg: AppConfig, mode: str) -> Tuple:
    log_dir = _make_log_dir(cfg, mode)
    os.makedirs(log_dir, exist_ok=True)

    ts_str     = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_base = (
        f"{cfg.experiment.subject_id}_"
        f"{cfg.experiment.exp_name}_"
        f"S{cfg.experiment.session_id}_"
        f"{cfg.experiment.session_type}_"
        f"{mode}_markers_{ts_str}.csv"
    )
    fname = os.path.join(log_dir, fname_base)

    f = open(fname, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(
        ["iso_time", "lsl_time_s", "code", "label", "local_recv_iso", "local_recv_s"]
    )
    return f, w, fname


def open_signal_logger(cfg: AppConfig, mode: str, channels: int) -> Tuple:
    log_dir = _make_log_dir(cfg, mode)
    os.makedirs(log_dir, exist_ok=True)

    ts_str     = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_base = (
        f"{cfg.experiment.subject_id}_"
        f"{cfg.experiment.exp_name}_"
        f"S{cfg.experiment.session_id}_"
        f"{cfg.experiment.session_type}_"
        f"{mode}_signal_{ts_str}.csv"
    )
    fname = os.path.join(log_dir, fname_base)

    f = open(fname, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    header = (
        ["iso_time", "lsl_time_s", "local_recv_s"]
        + [f"ch{i+1}" for i in range(channels)]
    )
    w.writerow(header)
    return f, w, fname


def run_receive(cfg: AppConfig, mode: str = "train", stop_event: threading.Event | None = None):
    """
    Etapa de recepção de dados (usada na Etapa 1 e Etapa 3).
    `mode`: "train" ou "realtime".
    `stop_event`: permite encerrar de forma controlada.
    """
    if stop_event is None:
        stop_event = threading.Event()

    # Resolve streams de entrada
    si_mark = resolve_stream(cfg.lsl.marker_name, cfg.lsl.marker_type)
    inlet_mark = StreamInlet(si_mark, recover=True)

    si_sig = resolve_stream(cfg.lsl.signal_name, cfg.lsl.signal_type)
    inlet_sig = StreamInlet(si_sig, recover=True)
    sig_channels = si_sig.channel_count()

    unix_offset = time.time() - local_clock()

    fM, wM, fnameM = open_marker_logger(cfg, mode)
    fS, wS, fnameS = open_signal_logger(cfg, mode, sig_channels)

    code_map = cfg.codes.code_map

    print("Aguardando dados... (stop_event controla o fim)")
    print("Códigos:", ", ".join(f"{k}={v}" for k, v in code_map.items()))

    last_info = time.time()
    n_mark = 0
    n_sig = 0

    try:
        while not stop_event.is_set():
            # --- Marcadores ---
            samples, timestamps = inlet_mark.pull_chunk(timeout=0.1, max_samples=64)
            if timestamps:
                for samp, ts in zip(samples, timestamps):
                    val = samp[0]
                    if isinstance(val, (bytes, str)):
                        try:
                            code = int(val)
                        except Exception:
                            inv = {v: k for k, v in code_map.items()}
                            code = inv.get(str(val).strip(), -1)
                    else:
                        code = int(val)

                    label = code_map.get(code, "UNKNOWN")

                    iso = dt.datetime.fromtimestamp(
                        ts + unix_offset
                    ).isoformat(timespec="milliseconds")
                    local_now = time.time()
                    local_iso = dt.datetime.fromtimestamp(
                        local_now
                    ).isoformat(timespec="milliseconds")

                    wM.writerow(
                        [iso, f"{ts:.6f}", code, label, local_iso, f"{local_now:.6f}"]
                    )
                    fM.flush()
                    n_mark += 1
                    print(f"[{iso}] code={code:>2} label={label:<14} (lsl_t={ts:.6f})")

            # --- Sinal contínuo ---
            sig_samples, sig_timestamps = inlet_sig.pull_chunk(
                timeout=0.05, max_samples=256
            )
            if sig_timestamps:
                rows = []
                now_local = time.time()
                for samp, ts in zip(sig_samples, sig_timestamps):
                    iso = dt.datetime.fromtimestamp(
                        ts + unix_offset
                    ).isoformat(timespec="milliseconds")
                    row = (
                        [iso, f"{ts:.6f}", f"{now_local:.6f}"]
                        + [f"{float(v):.6f}" for v in samp]
                    )
                    rows.append(row)
                wS.writerows(rows)
                fS.flush()
                n_sig += len(rows)

            tnow = time.time()
            if tnow - last_info > 4.0:
                print(f"(acumulado) marcadores={n_mark}, amostras_sinal={n_sig}")
                last_info = tnow

    finally:
        print(
            f"\nFinalizado.\n - Log de marcadores: {fnameM}\n - Log de sinal: {fnameS}"
        )
        try:
            fM.close()
        except Exception:
            pass
        try:
            fS.close()
        except Exception:
            pass