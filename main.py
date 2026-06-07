# main.py
import copy, glob, os, re, tempfile, threading, time
from pathlib import Path

import numpy as np
import yaml
from pylsl import StreamInlet, resolve_byprop
from config_models            import load_config, AppConfig
from online_inference         import run_realtime_decoder
from realtime_signal_transmit import run_transmission
from receive_data_log         import run_receive
from decoder_calibration      import run_calibration as run_decoder_calibration
from check_data               import run_check_data

YES = {"s", "sim", "y", "yes"}

def ask(msg: str, default: bool = False) -> bool:
    suf = "[S/n]" if default else "[s/N]"
    while True:
        ans = input(f"{msg} {suf}: ").strip().lower()
        if ans == "":
            return default
        if ans in YES:
            return True
        if ans in {"n", "nao", "não", "no"}:
            return False
        print("Responda apenas com s ou n.")

def load_cfg(path: Path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return load_config(path), raw
    except Exception:
        clean = {k: v for k, v in raw.items() if k != "protocol"}
        tmp = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)
                tmp = f.name
            return load_config(Path(tmp)), raw
        finally:
            if tmp:
                os.remove(tmp)

def set_session_type(cfg: AppConfig, session_type: str) -> AppConfig:
    cfg = copy.deepcopy(cfg)
    cfg.experiment.session_type = session_type
    return cfg

def stop_targets(raw: dict) -> set[str]:
    p = raw.get("protocol", {}) or {}
    values = [
        p.get("block_end_code"),
        p.get("block_end_label"),
        (raw.get("codes", {}) or {}).get("block_end"),
    ]
    return {str(v).strip() for v in values if v is not None}

def marker_is_stop(value, targets: set[str]) -> bool:
    txt = str(value).strip()
    candidates = {txt}
    try:
        candidates.add(str(int(float(txt))))
    except Exception:
        pass
    return bool(candidates & targets)

def wait_psychopy_stop(cfg: AppConfig, targets: set[str], stop_event: threading.Event) -> None:
    name  = getattr(cfg.lsl, "marker_name", "")
    stype = getattr(cfg.lsl, "marker_type", "Markers")

    print(f"[main] Aguardando marcador de fim do PsychoPy: {sorted(targets)}")
    while not stop_event.is_set():
        streams = resolve_byprop("name", name, timeout=1.0) if name else []
        if not streams and stype:
            streams = resolve_byprop("type", stype, timeout=1.0)
        if streams:
            inlet = StreamInlet(streams[0], recover=True)
            break
        print("[main] Stream de marcadores ainda não encontrado...")
    else:
        return

    while not stop_event.is_set():
        sample, _ = inlet.pull_sample(timeout=0.2)
        if sample and marker_is_stop(sample[0], targets):
            print(f"[main] Marcador de fim recebido: {sample[0]}")
            stop_event.set()
            return

def start_thread(threads: list, target, *args, **kwargs) -> None:
    th = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    threads.append(th)
    th.start()

def run_block(cfg: AppConfig, raw: dict, label: str, mode: str, decoder: bool = False, model_prefix: str | None = None) -> None:
    print(f"\n=== {label} | {cfg.experiment.session_type} | mode={mode} ===")

    stop_event = threading.Event()
    threads    = []
    targets    = stop_targets(raw)

    if cfg.runtime.simulate_signal:
        print(">> MODO TESTE: iniciando transmissão simulada por LSL.")
        start_thread(threads, run_transmission, cfg, mode, stop_event)
    else:
        print(">> MODO REAL: esperando sinal EEG já disponível por LSL.")

    start_thread(threads, run_receive, cfg, mode, stop_event)

    if decoder:
        start_thread(threads, run_realtime_decoder, cfg, mode = mode, model_prefix = model_prefix, stop_event = stop_event,)

    if targets:
        start_thread(threads, wait_psychopy_stop, cfg, targets, stop_event)
    else:
        print("[main] Nenhum marcador de fim configurado. Use Ctrl+C para encerrar o bloco.")

    try:
        while not stop_event.wait(0.2):
            pass
    except KeyboardInterrupt:
        print("[main] Ctrl+C recebido. Encerrando bloco.")
        stop_event.set()

    for th in threads:
        th.join(timeout=5.0)

    print("[main] Bloco encerrado.")

def run_training_stage(cfg: AppConfig, raw: dict, label: str, session_type: str) -> None:
    cfg_stage = set_session_type(cfg, session_type)
    rep = 1
    while ask(f"Rodar {label} {rep}?", default=(rep == 1)):
        run_block(cfg_stage, raw, label=label, mode="train")
        rep += 1

def run_calibration(cfg: AppConfig, raw: dict, session_type: str) -> None:
    cfg_cal = set_session_type(cfg, session_type)
    print(f"\n=== Calibração | {session_type} ===")

    res      = run_decoder_calibration(cfg_cal)
    acc_mean = float(np.mean(res["accs_cv"])) if res.get("accs_cv") else float("nan")
    print(f"[main] Acurácia média CV ~ {acc_mean:.3f}")

    if ask("Visualizar os dados com check_data?", default=False):
        run_check_data(cfg_cal)

def latest_model_prefix(cfg: AppConfig, session_type: str) -> str:
    train_dir = Path(cfg.experiment.log_root) / cfg.experiment.subject_id / f"S{cfg.experiment.session_id}" / session_type / "train"
    cands = glob.glob(str(train_dir / "*_classifier.pkl"))
    if not cands:
        raise FileNotFoundError(f"Não encontrei classificador em:\n  {train_dir}")
    return re.sub(r"_classifier\.pkl$", "", max(cands, key=os.path.getmtime))

def run_online_stage(cfg: AppConfig, raw: dict, online_type: str, model_type: str) -> None:
    cfg_online = set_session_type(cfg, online_type)
    rep = 1
    while ask(f"Rodar online {rep}?", default=(rep == 1)):
        model_prefix = latest_model_prefix(cfg, model_type)
        run_block(cfg_online, raw, label="Online", mode="realtime", decoder=True, model_prefix=model_prefix)
        rep += 1

def main() -> None:
    cfg, raw    = load_cfg(Path(__file__).resolve().parent / "config.yaml")
    p           = raw.get("protocol", {}) or {}
    em_type     = p.get("motor_session_type",   "EM_treino")
    im_type     = p.get("imagery_session_type", "IM_treino")
    online_type = p.get("online_session_type",  "IM_online")
    model_type  = p.get("model_session_type",   im_type)

    print("\n===== PROTOCOLO =====")
    print(f"Sujeito: {cfg.experiment.subject_id} | Sessão: S{cfg.experiment.session_id}")
    print(f"Marcador de fim: {sorted(stop_targets(raw))}\n")

    run_training_stage(cfg, raw, "Treino EM", em_type)

    if not ask("Continuar para treino IM?", default=True):
        return
    run_training_stage(cfg, raw, "Treino IM", im_type)

    if ask("Calibrar modelo agora?", default=True):
        run_calibration(cfg, raw, im_type)

    if not ask("Continuar para online?", default=True):
        return
    run_online_stage(cfg, raw, online_type, model_type)

    print("\n[main] Protocolo finalizado.")

if __name__ == "__main__":
    main()
