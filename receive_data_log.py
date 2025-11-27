# lsl_receive_graz_markers.py
import os
import csv
import time
import datetime as     dt
from   pylsl    import resolve_byprop, StreamInlet, local_clock


# --- Saída (Windows) ---
LOG_DIR            = r"C:\Users\User\Desktop\Dados" # Mudar dependendo do computador

# --- Identificadores da sessão ---
SUBJECT_ID         = "SY100"        # código do sujeito
EXP_NAME           = "TEST"         # experimento (string)
SESSION_ID         = 1              # número da sessão
SESSION_TYPE       = "IM_treino"    # tipo (IM_treino, EM_treino etc.)

# --- Streams de entrada ---
STREAM_NAME        = "GrazMI_Markers"          # Marcações do Psychopy
STREAM_TYPE        = "Markers"
SIGNAL_NAME        = "Cognionics Wireless EEG" # "GrazMI_SimEEG"   # Sinal que chega
SIGNAL_TYPE        = "EEG"
CODE_MAP           = {1:"BASELINE", 2:"ATTENTION", 3:"LEFT_MI_STIM", 4:"RIGHT_MI_STIM", 5:"ATTEMPT", 6:"REST",}


def resolve_stream(name: str, stype: str, timeout: float = 4.0):
    """Resolve por name e, se falhar, por type."""
    streams        = resolve_byprop('name', name, timeout=timeout)
    if not streams:
        streams    = resolve_byprop('type', stype, timeout=timeout)
    if not streams:
        raise RuntimeError(f"Nenhum stream LSL encontrado (name='{name}' / type='{stype}').")
    si             = streams[0]
    print(f"Conectado: name={si.name()}, type={si.type()}, chn={si.channel_count()}, fmt={si.channel_format()}, fs={si.nominal_srate():.2f}")
    return si

def open_marker_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts_str         = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_base     = f"{SUBJECT_ID}_{EXP_NAME}_S{SESSION_ID}_{SESSION_TYPE}_markers_{ts_str}.csv"
    fname          = os.path.join(LOG_DIR, fname_base)
    f              = open(fname, "w", newline="", encoding="utf-8")
    w              = csv.writer(f)
    w.writerow(["iso_time", "lsl_time_s", "code", "label", "local_recv_iso", "local_recv_s"])
    return f, w, fname

def open_signal_logger(channels: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts_str         = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_base     = f"{SUBJECT_ID}_{EXP_NAME}_S{SESSION_ID}_{SESSION_TYPE}_signal_{ts_str}.csv"
    fname          = os.path.join(LOG_DIR, fname_base)
    f              = open(fname, "w", newline="", encoding="utf-8")
    w              = csv.writer(f)
    header         = ["iso_time", "lsl_time_s", "local_recv_s"] + [f"ch{i+1}" for i in range(channels)]
    w.writerow(header)
    return f, w, fname

def main():
    # Resolve e abre inlets
    si_mark        = resolve_stream(STREAM_NAME, STREAM_TYPE)
    inlet_mark     = StreamInlet(si_mark, recover=True)
    si_sig         = resolve_stream(SIGNAL_NAME, SIGNAL_TYPE)
    inlet_sig      = StreamInlet(si_sig, recover=True)
    sig_channels   = si_sig.channel_count()

    # Conversão de tempo: Unix ≈ LSL + offset (aprox.)
    unix_offset    = time.time() - local_clock()

    # Abertura de arquivos
    fM, wM, fnameM = open_marker_logger()
    fS, wS, fnameS = open_signal_logger(sig_channels)

    print("Aguardando dados... (Ctrl+C para sair)")
    print("Códigos:", ", ".join(f"{k}={v}" for k, v in CODE_MAP.items()))

    last_info      = time.time()
    n_mark         = 0
    n_sig          = 0
    try:
        while True:
            # --- Marcadores ---
            samples, timestamps  = inlet_mark.pull_chunk(timeout=0.0, max_samples=64)
            if timestamps:
                for samp, ts in zip(samples, timestamps):
                    val          = samp[0]
                    if isinstance(val, (bytes, str)):
                        try:
                            code = int(val)
                        except Exception:
                            inv  = {v: k for k, v in CODE_MAP.items()}
                            code = inv.get(str(val).strip(), -1)
                    else:
                        code     = int(val)
                    label        = CODE_MAP.get(code, "UNKNOWN")

                    iso          = dt.datetime.fromtimestamp(ts + unix_offset).isoformat(timespec="milliseconds")
                    local_now    = time.time()
                    local_iso    = dt.datetime.fromtimestamp(local_now).isoformat(timespec="milliseconds")
                    wM.writerow([iso, f"{ts:.6f}", code, label, local_iso, f"{local_now:.6f}"])
                    fM.flush()
                    n_mark      += 1
                    print(f"[{iso}] code={code:>2} label={label:<14} (lsl_t={ts:.6f})")

            # --- Sinal contínuo ---
            sig_samples, sig_timestamps = inlet_sig.pull_chunk(timeout=0.05, max_samples=256)
            if sig_timestamps:
                rows             = []
                now_local        = time.time()
                for samp, ts in zip(sig_samples, sig_timestamps):
                    iso          = dt.datetime.fromtimestamp(ts + unix_offset).isoformat(timespec="milliseconds")
                    row          = [iso, f"{ts:.6f}", f"{now_local:.6f}"] + [f"{float(v):.6f}" for v in samp]
                    rows.append(row)
                wS.writerows(rows)
                fS.flush()
                n_sig           += len(rows)

            tnow              = time.time()
            if tnow - last_info > 4.0:
                print(f"(acumulado) marcadores={n_mark}, amostras_sinal={n_sig}")
                last_info      = tnow

    except KeyboardInterrupt:
        print(f"\nFinalizado.\n - Log de marcadores: {fnameM}\n - Log de sinal: {fnameS}")
    finally:
        try:
            fM.close()
        except Exception:
            pass
        try:
            fS.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
