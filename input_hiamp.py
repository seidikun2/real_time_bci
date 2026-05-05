# transmit.py
import contextlib, io, os, threading, time
import numpy         as np
from   pathlib       import Path
from   pylsl         import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
from   config_models import AppConfig

GTEC_ROOT                                = Path(r"D:\Documentos\gtec\gNEEDaccessClientAPI")
def init_pygds(root=GTEC_ROOT):
    c_dir, dll_dir                       = root / "C", root / "C" / "x64"
    headers                              = [c_dir / "GDSClientAPI.h", c_dir / "GDSClientAPI_gHIamp.h", c_dir / "GDSClientAPI_gNautilus.h", c_dir / "GDSClientAPI_gUSBamp.h",]

    os.add_dll_directory(str(dll_dir))

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import pygds

    pygds.Initialize([str(h) for h in headers], str(dll_dir / "GDSClientAPI.dll"))
    return pygds

def find_hiamp(pygds):
    devices                              = pygds.ConnectedDevices()

    for serial, devtype, inuse in devices:
        print(f"serial={serial} | type={devtype} | in_use={inuse}")

    for serial, devtype, inuse in devices:
        if devtype == pygds.DEVICE_TYPE_GHIAMP and not inuse:
            return serial

    raise RuntimeError("Nenhum g.HIamp livre encontrado.")

def configure_hiamp(d, cfg: AppConfig):
    fs                                    = int(cfg.sim_signal.fs)
    n_channels                            = int(cfg.sim_signal.channels)
    supported                             = d.GetSupportedSamplingRates()[0]

    d.SamplingRate, d.NumberOfScans       = ((fs, supported[fs]) if fs in supported else sorted(supported.items())[0])
    d.Counter                             = 0
    d.Trigger                             = 0

    if hasattr(d, "HoldEnabled"):
        d.HoldEnabled                     = 0

    if hasattr(d, "InternalSignalGenerator"):
        d.InternalSignalGenerator.Enabled = 0

    for i, ch in enumerate(d.Channels):
        ch.Acquire                        = int(i < n_channels)
        ch.BandpassFilterIndex            = -1
        ch.NotchFilterIndex               = -1

        if hasattr(ch, "ReferenceChannel"):
            ch.ReferenceChannel           = 0

    d.SetConfiguration()
    return d.N_ch_calc(), float(d.SamplingRate)

def make_outlet(cfg: AppConfig, fs: float, n_channels: int, serial: str):
    info                                  = StreamInfo(cfg.lsl.signal_name, cfg.lsl.signal_type, n_channels, fs, "float32", f"gHIamp_{serial}",)
    desc                                  = info.desc()
    desc.append_child_value("manufacturer", "g.tec")
    desc.append_child_value("device", "g.HIamp")
    desc.append_child_value("serial", serial)

    channels                              = desc.append_child("channels")
    for i in range(n_channels):
        ch                                = channels.append_child("channel")
        ch.append_child_value("label", f"CH{i + 1}")
        ch.append_child_value("type", "EEG")
        ch.append_child_value("unit", "uV")

    return StreamOutlet(info)

def find_markers(cfg: AppConfig):
    streams                               = (resolve_byprop("name", cfg.lsl.marker_name, timeout=1.0) or resolve_byprop("type", cfg.lsl.marker_type, timeout=1.0))

    if not streams:
        print("Nenhum stream de marcadores encontrado.")
        return None

    print(f"Marcadores encontrados: {streams[0].name()}")
    return StreamInlet(streams[0], recover=True)

def open_logs(cfg: AppConfig, mode: str, n_channels: int):
    path                                  = os.path.join(cfg.experiment.log_root, cfg.experiment.subject_id, f"S{cfg.experiment.session_id}", cfg.experiment.session_type, mode,)
    os.makedirs(path, exist_ok=True)
    sig_path                              = os.path.join(path, f"{mode}_signal.csv")
    mrk_path                              = os.path.join(path, f"{mode}_markers.csv")                   
    sig                                   = open(sig_path, "w", buffering=1)
    mrk                                   = open(mrk_path, "w", buffering=1)
    sig.write("t," + ",".join(f"ch{i + 1}" for i in range(n_channels)) + "\n")
    mrk.write("t,code,label\n")

    return sig, mrk

def marker_thread(cfg: AppConfig, inlet, mrk, stop_event: threading.Event):
    if inlet is None:
        return

    code_map                              = getattr(cfg.codes, "code_map", {})

    while not stop_event.is_set():
        samples, timestamps               = inlet.pull_chunk(timeout=0.2)

        for sample, _ in zip(samples, timestamps):
            try:
                code                      = int(sample[0])
                label                     = code_map.get(code, code_map.get(str(code), "UNK"))
                mrk.write(f"{time.time():.3f},{code},{label}\n")
            except Exception:
                pass

def stream_hiamp(cfg: AppConfig, d, outlet, sig, stop_event: threading.Event):
    fs                                     = float(d.SamplingRate)
    chunk                                  = int(cfg.sim_signal.chunk)
    device_block                           = int(d.NumberOfScans)
    block                                  = max(device_block, chunk)
    block                                  = ((block + device_block - 1) // device_block) * device_block

    print(f"Transmitindo g.HIamp por LSL: {cfg.lsl.signal_name} [{cfg.lsl.signal_type}]")
    print(f"fs={fs} Hz | canais={d.N_ch_calc()} | bloco={block} amostras")

    state                                  = {"samples": 0, "blocks": 0, "t0": time.time(), "last": time.time()}

    def on_block(samples):
        if stop_event.is_set():
            return False

        x                                   = np.asarray(samples, dtype=np.float32)
        outlet.push_chunk(x.tolist())

        t0                                  = time.time() - (len(x) - 1) / fs
        for i, row in enumerate(x):
            sig.write(f"{t0 + i / fs:.3f}," + ",".join(f"{v:.4f}" for v in row) + "\n")

        state["samples"]                    += len(x)
        state["blocks"]                     += 1

        now                                  = time.time()
        if now - state["last"] >= 1.0:
            eff_fs                           = state["samples"] / (now - state["t0"])
            print(f"blocos={state['blocks']:05d} | " f"amostras={state['samples']:07d} | " f"fs efetiva={eff_fs:.1f} Hz | " f"última ch1-4={np.round(x[-1, :4], 3)}")
            state["last"]                    = now

        return True

    d.GetData(block, more=on_block)

def run_transmission(cfg: AppConfig, mode: str = "train", stop_event: threading.Event | None = None):
    stop_event                               = stop_event or threading.Event()
    pygds, d, sig, mrk                       = init_pygds(), None, None, None

    try:
        serial                               = find_hiamp(pygds)
        d                                    = pygds.GDS(gds_device=serial)
        n_channels, fs                       = configure_hiamp(d, cfg)
        outlet                               = make_outlet(cfg, fs, n_channels, serial)
        inlet                                = find_markers(cfg)
        sig, mrk                             = open_logs(cfg, mode, n_channels)
        threading.Thread(target=marker_thread, args=(cfg, inlet, mrk, stop_event), daemon=True,).start()
        stream_hiamp(cfg, d, outlet, sig, stop_event)

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        stop_event.set()
        for obj in (sig, mrk, d):
            try:
                obj.Close() if obj is d and d is not None else obj.close()
            except Exception:
                pass
        try:
            pygds.Uninitialize()
        except Exception:
            pass
        print("Transmissão finalizada.")