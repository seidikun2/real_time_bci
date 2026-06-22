# main.py
import copy, glob, inspect, json, os, re, subprocess, sys, tempfile, threading, time
from pathlib import Path

import numpy as np
import yaml
from pylsl import StreamInlet, resolve_byprop
from config_models            import load_config, AppConfig
from online_inference         import run_realtime_decoder
from realtime_signal_transmit import run_transmission as run_sim_transmission
from input_hiamp              import run_transmission as run_hiamp_transmission
from receive_data_log         import run_receive
from decoder_calibration      import run_calibration as run_decoder_calibration
from check_data               import run_check_data

YES = {"s", "sim", "y", "yes"}
NO  = {"n", "nao", "não", "no"}

PHASE_ORDER   = ["execution", "imagery", "online"]
PHASE_ALIASES = {
    "execution": "execution", "execucao": "execution", "execução": "execution", "motor": "execution", "em": "execution", "treino_em": "execution",
    "imagery": "imagery", "imagetica": "imagery", "imagética": "imagery", "mi": "imagery", "im": "imagery", "treino_im": "imagery",
    "online": "online", "realtime": "online", "tempo_real": "online",
}


def ask(msg: str, default: bool = False) -> bool:
    suf = "[S/n]" if default else "[s/N]"
    while True:
        ans = input(f"{msg} {suf}: ").strip().lower()
        if ans == "":
            return default
        if ans in YES:
            return True
        if ans in NO:
            return False
        print("Responda apenas com s ou n.")


def ask_choice(msg: str, choices: dict[str, str], default: str) -> str:
    opts = "/".join([k.upper() if k == default else k for k in choices])
    while True:
        print(msg)
        for k, v in choices.items():
            print(f"  [{k}] {v}")
        ans = input(f"Escolha [{opts}]: ").strip().lower()
        if ans == "":
            ans = default
        if ans in choices:
            return ans
        print("Opção inválida.")


def load_cfg(path: Path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return load_config(path), raw
    except Exception:
        clean = {k: v for k, v in raw.items() if k != "protocol"}
        tmp   = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)
                tmp = f.name
            return load_config(Path(tmp)), raw
        finally:
            if tmp:
                os.remove(tmp)


def protocol(raw: dict) -> dict:
    return raw.get("protocol", {}) or {}


def set_session_type(cfg: AppConfig, session_type: str) -> AppConfig:
    cfg                         = copy.deepcopy(cfg)
    cfg.experiment.session_type = session_type
    return cfg


def normalize_phase(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in PHASE_ALIASES:
        raise ValueError(f"start_phase inválido: {value}. Use execution, imagery ou online.")
    return PHASE_ALIASES[key]


def data_dir(cfg: AppConfig, session_type: str, mode: str) -> Path:
    return Path(cfg.experiment.log_root) / cfg.experiment.subject_id / f"S{cfg.experiment.session_id}" / session_type / mode


def _split_marker_name(fname: str) -> tuple[str, str] | None:
    """
    Retorna (prefixo, run_id) para nomes no formato:
      <prefixo>_markers_YYYYMMDD_HHMMSS.csv
    """
    m = re.match(r"^(?P<prefix>.+)_markers_(?P<run_id>\d{8}_\d{6})\.csv$", fname)
    if not m:
        return None
    return m.group("prefix"), m.group("run_id")


def find_marker_signal_pairs(folder: Path) -> list[tuple[str, str]]:
    if not folder.exists():
        return []

    exact_pairs: list[tuple[str, str]] = []
    legacy_pairs: list[tuple[str, str]] = []

    markers = glob.glob(str(folder / "*markers_*.csv"))

    for m in markers:
        marker_path = Path(m)
        parsed      = _split_marker_name(marker_path.name)

        if parsed is not None:
            prefix, run_id = parsed
            signal_path    = marker_path.parent / f"{prefix}_signal_{run_id}.csv"
            if signal_path.exists():
                exact_pairs.append((str(marker_path), str(signal_path)))
            continue

        # Compatibilidade com arquivos antigos fora do padrão run_id.
        # Não é usado quando existem pares exatos, para evitar pareamento cruzado.
        base_m = marker_path.name
        if "_markers_" not in base_m:
            continue
        prefix, _ = base_m.split("_markers_", 1)
        sig_files = glob.glob(str(marker_path.parent / f"{prefix}_signal_*.csv"))
        if sig_files:
            s = max(sig_files, key=os.path.getmtime)
            legacy_pairs.append((str(marker_path), s))

    pairs = exact_pairs if exact_pairs else legacy_pairs
    return sorted(pairs, key=lambda p: os.path.getmtime(p[1]), reverse=True)

def pair_key(pair: tuple[str, str]) -> tuple[str, str]:
    return (os.path.abspath(pair[0]), os.path.abspath(pair[1]))


def detect_current_pair(folder: Path, before: list[tuple[str, str]], started_at: float) -> tuple[str, str] | None:
    after       = find_marker_signal_pairs(folder)
    before_keys = {pair_key(p) for p in before}
    new_pairs   = [p for p in after if pair_key(p) not in before_keys]

    if new_pairs:
        return new_pairs[0]

    fresh_pairs = [p for p in after if os.path.getmtime(p[1]) >= started_at - 2.0]
    if fresh_pairs:
        return fresh_pairs[0]

    return after[0] if after else None


def stop_targets(raw: dict) -> set[str]:
    p = protocol(raw)
    values = [
        p.get("block_end_code"),
        p.get("block_end_label"),
        (raw.get("codes", {}) or {}).get("block_end"),
    ]
    return {str(v).strip() for v in values if v is not None}


def marker_is_stop(value, targets: set[str]) -> bool:
    txt        = str(value).strip()
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


def run_block(cfg: AppConfig, raw: dict, label: str, mode: str, decoder: bool = False, model_prefix: str | None = None) -> dict:
    print(f"\n=== {label} | {cfg.experiment.session_type} | mode={mode} ===")

    started_at = time.time()
    stop_event = threading.Event()
    threads    = []
    targets    = stop_targets(raw)

    if cfg.runtime.simulate_signal:
        print(">> MODO TESTE: iniciando transmissão simulada por LSL.")
        start_thread(threads, run_sim_transmission, cfg, mode, stop_event)
    else:
        print(">> MODO REAL: iniciando aquisição g.HIamp via input_hiamp.py e publicando em LSL.")
        start_thread(threads, run_hiamp_transmission, cfg, mode, stop_event)

    start_thread(threads, run_receive, cfg, mode, stop_event)

    if decoder:
        start_thread(threads, run_realtime_decoder, cfg, mode=mode, model_prefix=model_prefix, stop_event=stop_event)

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

    ended_at = time.time()
    print("[main] Bloco encerrado.")
    return {"started_at": started_at, "ended_at": ended_at}


def cv_mean(res: dict) -> float:
    accs = res.get("accs_cv", []) if isinstance(res, dict) else []
    return float(np.mean(accs)) if len(accs) else float("nan")


def run_check_same_pair(cfg: AppConfig, raw: dict, markers_file: str, signal_file: str) -> None:
    if not protocol(raw).get("auto_check_data", True):
        return

    print("\n[main] Gerando checks automáticos no mesmo bloco...")
    sig    = inspect.signature(run_check_data)
    params = sig.parameters
    kwargs = {}

    for key in ("markers_file", "marker_file", "markers_csv", "mark_explicit"):
        if key in params:
            kwargs[key] = markers_file
            break

    for key in ("signal_file", "signal_csv", "sig_explicit"):
        if key in params:
            kwargs[key] = signal_file
            break

    try:
        if kwargs:
            run_check_data(cfg, **kwargs)
        else:
            print("[main] run_check_data não expõe parâmetros de arquivo; usando o comportamento padrão do módulo.")
            run_check_data(cfg)
    except Exception as exc:
        print(f"[main] check_data falhou: {type(exc).__name__}: {exc}")


def train_and_check_current_block(cfg: AppConfig, raw: dict, pair: tuple[str, str] | None) -> dict:
    if pair is None:
        print("[main] Não encontrei o par markers/signal recém-gravado; calibração não executada.")
        return {"acc_mean": float("nan"), "res": None}

    markers_file, signal_file = pair
    print("\n[main] Treinando classificador no bloco recém-gravado:")
    print(f"  markers: {os.path.basename(markers_file)}")
    print(f"  signal : {os.path.basename(signal_file)}")

    try:
        res      = run_decoder_calibration(cfg, markers_file=markers_file, signal_file=signal_file)
        acc_mean = cv_mean(res)
        print(f"[main] Acurácia média CV do bloco = {acc_mean:.3f}")
    except Exception as exc:
        print(f"[main] Calibração falhou: {type(exc).__name__}: {exc}")
        return {"acc_mean": float("nan"), "res": None}

    run_check_same_pair(cfg, raw, markers_file, signal_file)
    return {"acc_mean": acc_mean, "res": res}


def default_training_action(raw: dict, acc_mean: float) -> str:
    threshold = protocol(raw).get("min_cv_accuracy", None)
    if threshold is None or not np.isfinite(acc_mean):
        return "r" if not np.isfinite(acc_mean) else "s"
    return "s" if acc_mean >= float(threshold) else "r"


def post_training_action(label: str, raw: dict, acc_mean: float) -> str:
    threshold = protocol(raw).get("min_cv_accuracy", None)
    crit      = f" | critério={float(threshold):.2f}" if threshold is not None else ""
    acc_txt   = "nan" if not np.isfinite(acc_mean) else f"{acc_mean:.3f}"
    default   = default_training_action(raw, acc_mean)

    return ask_choice(
        f"\nResultado {label}: CV={acc_txt}{crit}. O que fazer agora?",
        {"r": "refazer este bloco", "s": "seguir para a próxima fase", "f": "finalizar a sessão"},
        default=default,
    )


def run_training_stage(cfg: AppConfig, raw: dict, label: str, session_type: str) -> str:
    cfg_stage = set_session_type(cfg, session_type)
    folder    = data_dir(cfg_stage, session_type, "train")
    existing  = find_marker_signal_pairs(folder)

    if existing:
        print(f"\n[main] Detectei {len(existing)} bloco(s) prévio(s) em {folder}.")
        print("[main] O próximo bloco será tratado como continuação/repetição desta sessão.")

    block_n = len(existing) + 1

    while True:
        if not ask(f"Iniciar {label} bloco {block_n}?", default=True):
            return "stop"

        before = find_marker_signal_pairs(folder)
        info   = run_block(cfg_stage, raw, label=f"{label} {block_n}", mode="train")
        pair   = detect_current_pair(folder, before, info["started_at"])
        out    = train_and_check_current_block(cfg_stage, raw, pair)
        action = post_training_action(label, raw, out["acc_mean"])

        if action == "r":
            block_n += 1
            continue
        if action == "s":
            return "next"
        return "stop"


def read_model_meta(prefix: str) -> dict:
    meta_path = prefix + "_meta.json"
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def model_session_types(raw: dict, em_type: str, im_type: str) -> list[str]:
    p   = protocol(raw)
    val = p.get("model_session_types", None)

    if val is None:
        val = [p.get("model_session_type", im_type), em_type, im_type]
    elif isinstance(val, str):
        val = [val]

    out = []
    for item in val:
        if item and item not in out:
            out.append(item)
    return out


def list_model_prefixes(cfg: AppConfig, session_types: list[str]) -> list[dict]:
    rows = []
    for session_type in session_types:
        folder = data_dir(cfg, session_type, "train")
        for clf_path in glob.glob(str(folder / "*_classifier.pkl")):
            prefix = re.sub(r"_classifier\.pkl$", "", clf_path)
            meta   = read_model_meta(prefix)
            acc    = ((meta.get("cv", {}) or {}).get("acc_mean", None)) if meta else None
            rows.append({
                "session_type": session_type,
                "prefix":       prefix,
                "clf_path":     clf_path,
                "acc":          acc,
                "mtime":        os.path.getmtime(clf_path),
            })

    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def choose_model_prefix(cfg: AppConfig, raw: dict, em_type: str, im_type: str) -> str:
    p = protocol(raw)

    explicit = p.get("online_model_prefix", None)
    if explicit and str(explicit).lower() not in {"ask", "latest", "auto"}:
        return str(explicit)

    rows = list_model_prefixes(cfg, model_session_types(raw, em_type, im_type))
    if not rows:
        raise FileNotFoundError("Não encontrei nenhum *_classifier.pkl para iniciar o online.")

    if explicit in {"latest", "auto"}:
        chosen = rows[0]
        print(f"[main] Classificador online automático: {chosen['prefix']}")
        return chosen["prefix"]

    print("\n===== CLASSIFICADORES DISPONÍVEIS PARA O ONLINE =====")
    for i, row in enumerate(rows, start=1):
        acc_txt = "sem CV" if row["acc"] is None else f"CV={float(row['acc']):.3f}"
        stamp   = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["mtime"]))
        print(f"  [{i}] {row['session_type']} | {acc_txt} | {stamp} | {os.path.basename(row['prefix'])}")

    while True:
        ans = input("Escolha o classificador [1 = mais recente]: ").strip()
        idx = 1 if ans == "" else int(ans) if ans.isdigit() else -1
        if 1 <= idx <= len(rows):
            return rows[idx - 1]["prefix"]
        print("Número inválido.")


def start_realtime_plot(cfg: AppConfig, raw: dict, model_prefix: str | None = None) -> subprocess.Popen | None:
    p = protocol(raw)
    if not p.get("realtime_plot", False):
        return None

    script = Path(__file__).resolve().parent / p.get("realtime_plot_script", "plot_decoder_realtime.py")
    if not script.exists():
        print(f"[main] Plot realtime habilitado, mas script não encontrado: {script}")
        return None

    cmd = [
        sys.executable,
        str(script),
        "--decoder-name", p.get("decoder_debug_name", getattr(cfg.decoder, "outlet_name", "Signal")),
        "--decoder-type", p.get("decoder_debug_type", getattr(cfg.decoder, "outlet_type", "BCI")),
        "--marker-name", getattr(cfg.lsl, "marker_name", "GrazMI_Markers"),
        "--marker-type", getattr(cfg.lsl, "marker_type", "Markers"),
    ]

    if p.get("realtime_plot_no_markers", False):
        cmd.append("--no-markers")
    if p.get("realtime_plot_hz", None) is not None:
        cmd += ["--plot-hz", str(p["realtime_plot_hz"])]
    if p.get("realtime_plot_window_s", None) is not None:
        cmd += ["--time-window", str(p["realtime_plot_window_s"])]

    meta = read_model_meta(model_prefix) if model_prefix else {}
    xlim = meta.get("pca_train_xlim", None)
    ylim = meta.get("pca_train_ylim", None)
    if xlim and ylim:
        cmd += ["--pca-xlim", str(xlim[0]), str(xlim[1]), "--pca-ylim", str(ylim[0]), str(ylim[1])]

    print("[main] Abrindo plot_decoder_realtime em processo Python próprio.")
    creationflags = 0
    if os.name == "nt" and p.get("realtime_plot_new_console", True):
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return subprocess.Popen(cmd, creationflags=creationflags)


def stop_realtime_plot(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def post_online_action(label: str) -> str:
    return ask_choice(
        f"\n{label} encerrado. O que fazer agora?",
        {"r": "refazer online", "f": "finalizar a sessão"},
        default="r",
    )


def run_online_stage(cfg: AppConfig, raw: dict, online_type: str, model_prefix: str) -> str:
    cfg_online = set_session_type(cfg, online_type)
    folder     = data_dir(cfg_online, online_type, "realtime")
    existing   = find_marker_signal_pairs(folder)

    if existing:
        print(f"\n[main] Detectei {len(existing)} bloco(s) online prévio(s) em {folder}.")
        print("[main] O próximo bloco será tratado como continuação/repetição desta sessão.")

    block_n = len(existing) + 1

    while True:
        if not ask(f"Iniciar online bloco {block_n}?", default=True):
            return "stop"

        plot_proc = start_realtime_plot(cfg_online, raw, model_prefix=model_prefix)
        try:
            run_block(cfg_online, raw, label=f"Online {block_n}", mode="realtime", decoder=True, model_prefix=model_prefix)
        finally:
            stop_realtime_plot(plot_proc)

        action = post_online_action("Online")
        if action == "r":
            block_n += 1
            continue
        return "stop"


def main() -> None:
    cfg, raw    = load_cfg(Path(__file__).resolve().parent / "config.yaml")
    p           = protocol(raw)
    em_type     = p.get("motor_session_type",   "EM_treino")
    im_type     = p.get("imagery_session_type", "IM_treino")
    online_type = p.get("online_session_type",  "IM_online")
    start_phase = normalize_phase(p.get("start_phase", "execution"))

    print("\n===== PROTOCOLO =====")
    print(f"Sujeito: {cfg.experiment.subject_id} | Sessão: S{cfg.experiment.session_id}")
    print(f"Fase inicial: {start_phase}")
    print(f"Marcador de fim: {sorted(stop_targets(raw))}")
    print("Fluxo: execução motora → imagética motora → online\n")

    phases = PHASE_ORDER[PHASE_ORDER.index(start_phase):]

    if "execution" in phases:
        if run_training_stage(cfg, raw, "Execução motora", em_type) == "stop":
            print("\n[main] Sessão finalizada após execução motora.")
            return

    if "imagery" in phases:
        if run_training_stage(cfg, raw, "Imagética motora", im_type) == "stop":
            print("\n[main] Sessão finalizada após imagética motora.")
            return

    if "online" in phases:
        model_prefix = choose_model_prefix(cfg, raw, em_type, im_type)
        run_online_stage(cfg, raw, online_type, model_prefix)

    print("\n[main] Protocolo finalizado.")


if __name__ == "__main__":
    main()
