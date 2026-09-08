# -*- coding: utf-8 -*-
"""
input_hiamp.py

Entrada real g.HIamp -> LSL.

Este módulo é usado pelo main.py quando runtime.simulate_signal = false.
Ele configura o g.HIamp, adquire os canais físicos definidos em cfg.hiamp
e publica no stream LSL o sinal já re-referenciado.

Configuração adotada para o g.Scarabeo/g.HIamp com 16 entradas:
- adquire 16 canais físicos;
- usa o canal 16 como referência de software;
- remove o canal 16 do stream de saída;
- aplica Common Average Reference (CAR) nos canais EEG restantes;
- publica CH1..CH15 no LSL, já prontos para gravação, treino, check_data e online.

A gravação sincronizada de sinal + marcadores continua sendo feita por receive_data_log.py,
para manter exatamente o mesmo formato CSV usado no treino, check_data e online.
"""

from __future__ import annotations

import contextlib
import io
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from pylsl import StreamInfo, StreamOutlet, cf_float32

from config_models import AppConfig


DEFAULT_GTEC_ROOT = Path(r"D:\Documentos\gtec\gNEEDaccessClientAPI")


def _raw(cfg: AppConfig) -> dict[str, Any]:
    return getattr(cfg, "_raw_config", {}) or {}


def _raw_get(cfg: AppConfig, *keys: str, default=None):
    obj = _raw(cfg)
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def _raw_bool(cfg: AppConfig, *keys: str, default: bool = False) -> bool:
    value = _raw_get(cfg, *keys, default=default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "sim", "s"}

    return bool(value)


def get_gtec_root(cfg: AppConfig) -> Path:
    value = (
        _raw_get(cfg, "gtec_root", default=None)
        or _raw_get(cfg, "hiamp", "gtec_root", default=None)
        or os.environ.get("GTEC_ROOT")
        or DEFAULT_GTEC_ROOT
    )
    return Path(value)


def get_hiamp_acquisition_channels(cfg: AppConfig) -> int:
    """
    Número de entradas físicas adquiridas do g.HIamp.

    Aqui ele deve ser 16: canais 1-15 são EEG úteis e o canal 16 é usado como
    referência de software antes do CAR.
    """
    return int(_raw_get(cfg, "hiamp", "acquisition_channels", default=cfg.n_channels))


def get_reference_channel(cfg: AppConfig, acquisition_channels: int) -> int | None:
    """
    Canal de referência em base 1.

    reference_channel: 16 significa usar CH16 como referência.
    """
    value = _raw_get(cfg, "hiamp", "reference_channel", default=None)

    if value in (None, "", "none", "None", "null"):
        return None

    ref_ch = int(value)

    if not 1 <= ref_ch <= acquisition_channels:
        raise ValueError(
            f"hiamp.reference_channel={ref_ch} inválido para "
            f"hiamp.acquisition_channels={acquisition_channels}."
        )

    return ref_ch


def get_output_indices(cfg: AppConfig, acquisition_channels: int) -> list[int]:
    ref_ch   = get_reference_channel(cfg, acquisition_channels)
    drop_ref = _raw_bool(cfg, "hiamp", "drop_reference_channel", default=True)

    indices = list(range(acquisition_channels))

    if ref_ch is not None and drop_ref:
        ref_idx = ref_ch - 1
        indices = [i for i in indices if i != ref_idx]

    return indices


def get_output_channel_labels(cfg: AppConfig, acquisition_channels: int) -> list[str]:
    indices = get_output_indices(cfg, acquisition_channels)
    suffix  = "_CAR" if get_reference_mode(cfg) == "car" else ""

    return [f"CH{i + 1}{suffix}" for i in indices]


def get_reference_mode(cfg: AppConfig) -> str:
    """
    Modo de re-referência aplicado antes de publicar no LSL.

    Opções:
    - "none": publica os canais sem re-referência de software;
    - "channel": subtrai hiamp.reference_channel e não aplica CAR;
    - "car": subtrai hiamp.reference_channel, remove o canal de referência e aplica CAR.
    """
    mode = str(_raw_get(cfg, "hiamp", "reference_mode", default="car")).strip().lower()

    aliases = {
        "none": "none",
        "off": "none",
        "raw": "none",
        "channel": "channel",
        "reference": "channel",
        "ref": "channel",
        "car": "car",
        "common_average": "car",
        "common_average_reference": "car",
        "channel_then_car": "car",
        "ref_then_car": "car",
    }

    if mode not in aliases:
        raise ValueError(
            "hiamp.reference_mode inválido. Use: none, channel ou car."
        )

    return aliases[mode]


def init_pygds(root: Path):
    c_dir   = root / "C"
    dll_dir = c_dir / "x64"
    dll     = dll_dir / "GDSClientAPI.dll"
    headers = [
        c_dir / "GDSClientAPI.h",
        c_dir / "GDSClientAPI_gHIamp.h",
        c_dir / "GDSClientAPI_gNautilus.h",
        c_dir / "GDSClientAPI_gUSBamp.h",
    ]

    missing = [p for p in [dll, *headers] if not p.exists()]
    if missing:
        txt = "\n  - ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Arquivos da API g.tec não encontrados:\n  - {txt}")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import pygds

    pygds.Initialize([str(h) for h in headers], str(dll))
    return pygds


def find_hiamp(pygds):
    devices = pygds.ConnectedDevices()

    for serial, devtype, inuse in devices:
        print(f"serial={serial} | type={devtype} | in_use={inuse}")

    for serial, devtype, inuse in devices:
        if devtype == pygds.DEVICE_TYPE_GHIAMP and not inuse:
            return serial

    raise RuntimeError("Nenhum g.HIamp livre encontrado.")


def configure_hiamp(d, cfg: AppConfig):
    fs_req               = int(round(float(cfg.fs_hz)))
    acquisition_channels = get_hiamp_acquisition_channels(cfg)
    supported            = d.GetSupportedSamplingRates()[0]

    if fs_req in supported:
        d.SamplingRate  = fs_req
        d.NumberOfScans = supported[fs_req]
    else:
        d.SamplingRate, d.NumberOfScans = sorted(supported.items())[0]
        print(f"[g.HIamp] Fs solicitada={fs_req} Hz indisponível; usando {d.SamplingRate} Hz.")

    d.Counter = 0
    d.Trigger = 0

    if hasattr(d, "HoldEnabled"):
        d.HoldEnabled = 0

    if hasattr(d, "InternalSignalGenerator"):
        d.InternalSignalGenerator.Enabled = 0

    for i, ch in enumerate(d.Channels):
        ch.Acquire             = int(i < acquisition_channels)
        ch.BandpassFilterIndex = -1
        ch.NotchFilterIndex    = -1

        # Não forçamos ReferenceChannel do driver aqui.
        # O CH16 é usado como referência de software abaixo, de forma explícita
        # e rastreável, antes da publicação em LSL.
        if hasattr(ch, "ReferenceChannel"):
            ch.ReferenceChannel = 0

    d.SetConfiguration()

    n_calc = int(d.N_ch_calc())
    if n_calc < acquisition_channels:
        raise RuntimeError(
            f"g.HIamp retornou {n_calc} canais, mas hiamp.acquisition_channels={acquisition_channels}."
        )

    return acquisition_channels, float(d.SamplingRate)


def transform_reference(cfg: AppConfig, x: np.ndarray) -> np.ndarray:
    """
    Aplica referência por canal 16 e CAR.

    Com a configuração padrão:
    entrada : CH1..CH16
    saída   : CH1..CH15, cada amostra com média espacial removida.
    """
    acquisition_channels = x.shape[1]
    ref_ch               = get_reference_channel(cfg, acquisition_channels)
    mode                 = get_reference_mode(cfg)
    out_indices          = get_output_indices(cfg, acquisition_channels)

    y = x.astype(np.float32, copy=False)

    if mode in {"channel", "car"} and ref_ch is not None:
        ref_idx = ref_ch - 1
        y       = y - y[:, [ref_idx]]

    y = y[:, out_indices]

    if mode == "car":
        y = y - np.mean(y, axis=1, keepdims=True)

    return y.astype(np.float32, copy=False)


def make_outlet(cfg: AppConfig, fs: float, channel_labels: list[str], serial: str) -> StreamOutlet:
    info = StreamInfo(
        cfg.lsl.signal_name,
        cfg.lsl.signal_type,
        len(channel_labels),
        fs,
        cf_float32,
        f"gHIamp_{serial}",
    )
    desc = info.desc()
    desc.append_child_value("manufacturer", "g.tec")
    desc.append_child_value("device", "g.HIamp")
    desc.append_child_value("serial", serial)
    desc.append_child_value("reference_mode", get_reference_mode(cfg))
    desc.append_child_value("reference_channel", str(_raw_get(cfg, "hiamp", "reference_channel", default="")))
    desc.append_child_value("drop_reference_channel", str(_raw_get(cfg, "hiamp", "drop_reference_channel", default=True)))

    channels = desc.append_child("channels")
    for label in channel_labels:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("type", "EEG")
        ch.append_child_value("unit", "uV")

    return StreamOutlet(info)


def get_block_size(cfg: AppConfig, d) -> int:
    chunk        = int(_raw_get(cfg, "hiamp", "chunk", default=getattr(cfg.sim_signal, "chunk", 10)))
    device_block = int(d.NumberOfScans)
    block        = max(device_block, chunk)
    return ((block + device_block - 1) // device_block) * device_block


def stream_hiamp(cfg: AppConfig, d, outlet: StreamOutlet, stop_event: threading.Event) -> None:
    fs                   = float(d.SamplingRate)
    block                = get_block_size(cfg, d)
    acquisition_channels = get_hiamp_acquisition_channels(cfg)
    channel_labels       = get_output_channel_labels(cfg, acquisition_channels)

    print(f"Transmitindo g.HIamp por LSL: {cfg.lsl.signal_name} [{cfg.lsl.signal_type}]")
    print(
        f"fs={fs:.1f} Hz | canais adquiridos={acquisition_channels} | "
        f"canais publicados={len(channel_labels)} | bloco={block} amostras"
    )
    print(
        f"referência={_raw_get(cfg, 'hiamp', 'reference_channel', default=None)} | "
        f"modo={get_reference_mode(cfg)} | labels={channel_labels[:4]}..."
    )

    state = {"samples": 0, "blocks": 0, "t0": time.time(), "last": time.time()}

    def on_block(samples):
        if stop_event.is_set():
            return False

        x = np.asarray(samples, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] < acquisition_channels:
            raise RuntimeError(
                f"Bloco recebido com {x.shape[1]} canais, mas eram esperados {acquisition_channels}."
            )

        x = x[:, :acquisition_channels]
        y = transform_reference(cfg, x)

        outlet.push_chunk(y.tolist())

        state["samples"] += len(y)
        state["blocks"]  += 1

        now = time.time()
        if now - state["last"] >= 1.0:
            eff_fs = state["samples"] / max(now - state["t0"], 1e-9)
            print(
                f"blocos={state['blocks']:05d} | "
                f"amostras={state['samples']:07d} | "
                f"fs efetiva={eff_fs:.1f} Hz | "
                f"última CAR ch1-4={np.round(y[-1, :4], 3)}"
            )
            state["last"] = now

        return True

    d.GetData(block, more=on_block)


def run_transmission(cfg: AppConfig, mode: str = "train", stop_event: threading.Event | None = None) -> None:
    del mode  # a gravação do bloco é feita por receive_data_log.py

    stop_event = stop_event or threading.Event()
    pygds      = None
    d          = None

    try:
        root   = get_gtec_root(cfg)
        pygds  = init_pygds(root)
        serial = find_hiamp(pygds)

        print(f"[g.HIamp] Conectando ao dispositivo: {serial}")
        d = pygds.GDS(gds_device=serial)

        acquisition_channels, fs = configure_hiamp(d, cfg)
        channel_labels           = get_output_channel_labels(cfg, acquisition_channels)
        outlet                   = make_outlet(cfg, fs, channel_labels, serial)

        stream_hiamp(cfg, d, outlet, stop_event)

    except KeyboardInterrupt:
        stop_event.set()

    except Exception as exc:
        print(f"[g.HIamp] Falha na transmissão: {type(exc).__name__}: {exc}")
        stop_event.set()

    finally:
        stop_event.set()

        if d is not None:
            try:
                d.Close()
            except Exception:
                pass

        if pygds is not None:
            try:
                pygds.Uninitialize()
            except Exception:
                pass

        print("Transmissão g.HIamp finalizada.")
