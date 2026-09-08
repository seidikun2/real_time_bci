from pathlib import Path
import os, time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, cf_float32


BASE = Path(r"D:\Documentos\gtec\gNEEDaccessClientAPI")
C_DIR = BASE / "C"
DLL_DIR = C_DIR / "x64"

HEADERS = [
    C_DIR / "GDSClientAPI.h",
    C_DIR / "GDSClientAPI_gHIamp.h",
    C_DIR / "GDSClientAPI_gNautilus.h",
    C_DIR / "GDSClientAPI_gUSBamp.h",
]

DLL = DLL_DIR / "GDSClientAPI.dll"

N_CHANNELS = 16
FS = 256
BLOCK_SEC = 0.1

LSL_STREAM_NAME = "gHIamp_EEG"
LSL_STREAM_TYPE = "EEG"


def run_gtec_transmission(cfg=None, mode="train", stop_event=None):
    dll_handle = os.add_dll_directory(str(DLL_DIR))

    import pygds

    pygds.Initialize([str(h) for h in HEADERS], str(DLL))

    d = None

    try:
        devices = pygds.ConnectedDevices()

        hiamp_serials = [
            serial for serial, devtype, inuse in devices
            if devtype == pygds.DEVICE_TYPE_GHIAMP and not inuse
        ]

        if not hiamp_serials:
            raise RuntimeError("Nenhum g.HIamp disponível encontrado.")

        serial = hiamp_serials[0]
        print(f"[g.HIamp] Conectando ao dispositivo: {serial}")

        d = pygds.GDS(gds_device=serial)

        supported = d.GetSupportedSamplingRates()[0]

        if FS in supported:
            d.SamplingRate = FS
            d.NumberOfScans = supported[FS]
        else:
            d.SamplingRate, d.NumberOfScans = sorted(supported.items())[0]

        d.Counter = 0
        d.Trigger = 0

        if hasattr(d, "HoldEnabled"):
            d.HoldEnabled = 0

        if hasattr(d, "InternalSignalGenerator"):
            d.InternalSignalGenerator.Enabled = 0

        for i, ch in enumerate(d.Channels):
            ch.Acquire = int(i < N_CHANNELS)
            ch.BandpassFilterIndex = -1
            ch.NotchFilterIndex = -1

            if hasattr(ch, "ReferenceChannel"):
                ch.ReferenceChannel = 0

        d.SetConfiguration()

        n_ch = d.N_ch_calc()

        info = StreamInfo(
            LSL_STREAM_NAME,
            LSL_STREAM_TYPE,
            n_ch,
            float(d.SamplingRate),
            cf_float32,
            f"gHIamp_{serial}",
        )

        channels = info.desc().append_child("channels")
        for i in range(n_ch):
            ch = channels.append_child("channel")
            ch.append_child_value("label", f"Ch{i + 1}")
            ch.append_child_value("unit", "uV")
            ch.append_child_value("type", "EEG")

        outlet = StreamOutlet(info)

        desired_block = int(d.SamplingRate * BLOCK_SEC)
        device_block = int(d.NumberOfScans)
        block_scans = ((desired_block + device_block - 1) // device_block) * device_block

        print(f"[g.HIamp] Sampling rate: {d.SamplingRate} Hz")
        print(f"[g.HIamp] Canais adquiridos: {n_ch}")
        print(f"[g.HIamp] Block scans: {block_scans}")
        print(f"[g.HIamp] Publicando LSL como name='{LSL_STREAM_NAME}', type='{LSL_STREAM_TYPE}'")
        print("[g.HIamp] Aquisição iniciada.\n")

        t0 = time.time()
        state = {"block": 0, "sample": 0}

        def on_block(samples):
            if stop_event is not None and stop_event.is_set():
                return False

            x = np.asarray(samples, dtype=np.float32)

            if x.ndim == 1:
                x = x.reshape(-1, n_ch)

            outlet.push_chunk(x.tolist())

            state["block"] += 1
            state["sample"] += len(x)

            if state["block"] % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"[g.HIamp] bloco={state['block']:05d} | "
                    f"amostras={state['sample']:08d} | "
                    f"t={elapsed:7.3f}s | "
                    f"shape={x.shape}"
                )

            return True

        d.GetData(block_scans, more=on_block)

    except KeyboardInterrupt:
        print("\n[g.HIamp] Aquisição interrompida.")

    finally:
        if d is not None:
            d.Close()

        pygds.Uninitialize()
        dll_handle.close()

        print("[g.HIamp] Finalizado.")