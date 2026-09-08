from __future__ import annotations

import copy
import glob
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from pylsl import StreamInlet, resolve_byprop

from bci_core.config_models import AppConfig, load_config
from bci_core.class_schema import classes_from_sequence, display_name
from bci_core.online_inference import run_realtime_decoder
from bci_core.intention_control import run_intention_controller
from bci_core.realtime_signal_transmit import run_transmission as run_sim_transmission
from bci_core.input_hiamp import run_transmission as run_hiamp_transmission
from bci_core.receive_data_log import run_receive
from bci_core.decoder_calibration import run_calibration as run_decoder_calibration
from bci_core.check_data import run_check_data
from bci_core.psychopy_process import start_psychopy, stop_psychopy, timeout_value


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"

YES = {"s", "sim", "y", "yes"}
NO = {"n", "nao", "não", "no"}

PHASE_ORDER = ["execution", "imagery", "online"]
PHASE_ALIASES = {
    "execution": "execution", "execucao": "execution", "execução": "execution", "motor": "execution", "em": "execution",
    "imagery": "imagery", "imagetica": "imagery", "imagética": "imagery", "mi": "imagery", "im": "imagery",
    "online": "online", "realtime": "online", "tempo_real": "online",
}


def ask(msg: str, default: bool = False) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    while True:
        ans = input(f"{msg} {suffix}: ").strip().lower()
        if not ans:
            return default
        if ans in YES:
            return True
        if ans in NO:
            return False
        print("Responda apenas com s ou n.")


def ask_choice(msg: str, choices: dict[str, str], default: str) -> str:
    opts = "/".join(k.upper() if k == default else k for k in choices)
    while True:
        print(msg)
        for key, label in choices.items():
            print(f"  [{key}] {label}")
        ans = input(f"Escolha [{opts}]: ").strip().lower() or default
        if ans in choices:
            return ans
        print("Opção inválida.")


def load_cfg() -> tuple[AppConfig, dict]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return load_config(str(CONFIG_PATH)), raw


def protocol(raw: dict) -> dict:
    value = raw.get("protocol", {}) or {}
    return value if isinstance(value, dict) else {}


def psychopy_cfg(raw: dict) -> dict:
    value = raw.get("psychopy", {}) or {}
    return value if isinstance(value, dict) else {}


def debug_plot_cfg(raw: dict) -> dict:
    value = raw.get("debug_plot", {}) or {}
    return value if isinstance(value, dict) else {}


def set_session_type(cfg: AppConfig, session_type: str) -> AppConfig:
    out = copy.deepcopy(cfg)
    out.experiment.session_type = session_type
    return out


def normalize_phase(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in PHASE_ALIASES:
        raise ValueError(f"start_phase inválido: {value}. Use execution, imagery ou online.")
    return PHASE_ALIASES[key]


def session_root(cfg: AppConfig) -> Path:
    return Path(cfg.experiment.log_root) / cfg.experiment.subject_id / f"S{cfg.experiment.session_id}"


def data_dir(cfg: AppConfig, session_type: str, mode: str) -> Path:
    """Pasta de dados. Treinos ficam por fase; todo online fica em S#/online."""
    if str(mode).lower() in {"online", "realtime"} or str(session_type).startswith("IM_online"):
        return session_root(cfg) / "online"
    return session_root(cfg) / session_type / mode


def _split_marker_name(fname: str) -> tuple[str, str] | None:
    m = re.match(r"^(?P<prefix>.+)_markers_(?P<run_id>\d{8}_\d{6})\.csv$", fname)
    return (m.group("prefix"), m.group("run_id")) if m else None


def find_marker_signal_pairs(folder: Path) -> list[tuple[str, str]]:
    if not folder.exists():
        return []
    exact: list[tuple[str, str]] = []
    legacy: list[tuple[str, str]] = []
    for marker in glob.glob(str(folder / "*markers_*.csv")):
        mp = Path(marker)
        parsed = _split_marker_name(mp.name)
        if parsed:
            prefix, run_id = parsed
            sig = mp.parent / f"{prefix}_signal_{run_id}.csv"
            if sig.exists():
                exact.append((str(mp), str(sig)))
            continue
        if "_markers_" in mp.name:
            prefix = mp.name.split("_markers_", 1)[0]
            sigs = glob.glob(str(mp.parent / f"{prefix}_signal_*.csv"))
            if sigs:
                legacy.append((str(mp), max(sigs, key=os.path.getmtime)))
    pairs = exact if exact else legacy
    return sorted(pairs, key=lambda pair: os.path.getmtime(pair[1]), reverse=True)


def pair_key(pair: tuple[str, str]) -> tuple[str, str]:
    return os.path.abspath(pair[0]), os.path.abspath(pair[1])


def detect_current_pair(folder: Path, before: list[tuple[str, str]], started_at: float) -> tuple[str, str] | None:
    after = find_marker_signal_pairs(folder)
    before_keys = {pair_key(p) for p in before}
    new_pairs = [p for p in after if pair_key(p) not in before_keys]
    if new_pairs:
        return new_pairs[0]
    fresh = [p for p in after if os.path.getmtime(p[1]) >= started_at - 2.0]
    return fresh[0] if fresh else (after[0] if after else None)


def stop_targets(raw: dict) -> set[str]:
    p = protocol(raw)
    values = [p.get("block_end_code"), p.get("block_end_label"), (raw.get("codes", {}) or {}).get("block_end")]
    return {str(v).strip() for v in values if v is not None}


def marker_is_stop(value, targets: set[str]) -> bool:
    txt = str(value).strip()
    candidates = {txt}
    try:
        candidates.add(str(int(float(txt))))
    except Exception:
        pass
    return bool(candidates & targets)


def wait_psychopy_stop(cfg: AppConfig, targets: set[str], stop_event: threading.Event, marker_state: dict) -> None:
    name = getattr(cfg.lsl, "marker_name", "")
    stype = getattr(cfg.lsl, "marker_type", "Markers")
    print(f"[main] Aguardando marcadores PsychoPy; fim={sorted(targets)}")

    while not stop_event.is_set():
        streams = resolve_byprop("name", name, timeout=1.0) if name else []
        if not streams and stype:
            streams = resolve_byprop("type", stype, timeout=1.0)
        if streams:
            inlet = StreamInlet(streams[0], recover=True)
            marker_state["stream_connected_at"] = time.monotonic()
            print("[main] Stream de marcadores PsychoPy conectado.")
            break
    else:
        return

    while not stop_event.is_set():
        sample, _ = inlet.pull_sample(timeout=0.2)
        if not sample:
            continue
        now = time.monotonic()
        marker_state.setdefault("first_marker_at", now)
        marker_state["last_marker_at"] = now
        marker_state["last_marker"] = sample[0]
        if marker_is_stop(sample[0], targets):
            marker_state["block_end_received"] = True
            print(f"[main] BLOCK_END recebido: {sample[0]}")
            stop_event.set()
            return


def start_thread(threads: list[threading.Thread], target, *args, **kwargs) -> None:
    th = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    threads.append(th)
    th.start()


def run_block(
    cfg: AppConfig,
    raw: dict,
    label: str,
    mode: str,
    decoder: bool = False,
    model_ref: str | None = None,
    feedback_mode: str = "none",
) -> dict:
    print(f"\n=== {label} | {cfg.experiment.session_type} | mode={mode} ===")

    started_at = time.time()
    started_mono = time.monotonic()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    marker_state: dict = {}
    targets = stop_targets(raw)
    psychopy_proc = None
    psychopy_started_mono: float | None = None
    psychopy_failed = False
    stop_reason = "block_end_or_external_stop"

    if cfg.runtime.simulate_signal:
        print(">> MODO TESTE: sinal simulado LSL.")
        start_thread(threads, run_sim_transmission, cfg, mode, stop_event)
    else:
        print(">> MODO REAL: g.HIamp → LSL.")
        start_thread(threads, run_hiamp_transmission, cfg, mode, stop_event)

    start_thread(threads, run_receive, cfg, mode, stop_event)

    if decoder:
        # Decoder: publica rep1/rep2 + probabilidades LEFT/BOTH/RIGHT.
        start_thread(
            threads,
            run_realtime_decoder,
            cfg,
            mode,
            model_ref,
            stop_event,
            feedback_mode,
        )
        # Controlador: usa densidade PCA + probabilidades + histerese temporal
        # e publica a posição contínua das pernas para o Unity.
        start_thread(
            threads,
            run_intention_controller,
            cfg,
            mode,
            stop_event,
            feedback_mode,
        )

    if targets:
        start_thread(threads, wait_psychopy_stop, cfg, targets, stop_event, marker_state)

    # Dá tempo para aquisição/logger/decoder criarem seus streams antes de abrir o PsychoPy.
    pcfg = psychopy_cfg(raw)
    pre_launch = float(pcfg.get("pre_launch_pause_s", 2.0) or 0.0)
    if pre_launch > 0:
        print(f"[main] Preparando streams por {pre_launch:.1f}s antes de abrir o PsychoPy...")
        if stop_event.wait(pre_launch):
            stop_reason = "external_stop_before_psychopy"

    if not stop_event.is_set():
        try:
            psychopy_proc = start_psychopy(cfg, raw, project_dir=PROJECT_DIR)
            if psychopy_proc is not None:
                psychopy_started_mono = time.monotonic()
                grace = float(pcfg.get("startup_grace_s", 10.0) or 0.0)
                print(
                    f"[main] PsychoPy iniciado. Ele pode levar alguns segundos para abrir. "
                    f"Período de carregamento: {grace:.0f}s."
                )
        except Exception as exc:
            print(f"[main] Falha ao iniciar PsychoPy: {type(exc).__name__}: {exc}")
            stop_reason = "psychopy_start_failed"
            psychopy_failed = True
            stop_event.set()

    startup_timeout = timeout_value(raw, "startup_timeout_s", 90.0)
    stall_timeout = timeout_value(raw, "marker_stall_timeout_s", 60.0)
    max_runtime = timeout_value(raw, "max_block_duration_s", None)
    startup_grace = float(pcfg.get("startup_grace_s", 10.0) or 0.0)
    status_every = max(1.0, float(pcfg.get("startup_status_every_s", 5.0) or 5.0))
    last_status = psychopy_started_mono or started_mono
    process_exit_seen_at = None

    try:
        while not stop_event.wait(0.2):
            now = time.monotonic()
            first_marker = marker_state.get("first_marker_at")
            last_marker = marker_state.get("last_marker_at")

            if psychopy_proc is not None:
                rc = psychopy_proc.poll()
                if rc is not None:
                    if marker_state.get("block_end_received"):
                        stop_reason = "block_end"
                        stop_event.set()
                        break
                    if process_exit_seen_at is None:
                        process_exit_seen_at = now
                    elif now - process_exit_seen_at > 1.0:
                        print(f"[main] PsychoPy encerrou antes de BLOCK_END (exit={rc}).")
                        stop_reason = "psychopy_exited_early"
                        psychopy_failed = True
                        stop_event.set()
                        break

                if first_marker is None and psychopy_started_mono is not None:
                    elapsed = now - psychopy_started_mono
                    if now - last_status >= status_every:
                        print(f"[main] PsychoPy carregando... {elapsed:.0f}s desde a abertura.")
                        last_status = now
                    if startup_timeout is not None and elapsed > startup_grace + startup_timeout:
                        print(f"[main] Nenhum marcador após {elapsed:.0f}s; encerrando bloco.")
                        stop_reason = "psychopy_startup_timeout"
                        psychopy_failed = True
                        stop_event.set()
                        break

                if stall_timeout is not None and last_marker is not None and now - last_marker > stall_timeout:
                    print(f"[main] Nenhum marcador PsychoPy há {stall_timeout:.1f}s. Assumindo travamento.")
                    stop_reason = "psychopy_marker_stall"
                    psychopy_failed = True
                    stop_event.set()
                    break

            if max_runtime is not None and now - started_mono > max_runtime:
                print(f"[main] Duração máxima do bloco excedida ({max_runtime:.1f}s).")
                stop_reason = "block_timeout"
                psychopy_failed = psychopy_proc is not None
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("[main] Ctrl+C recebido. Encerrando bloco.")
        stop_reason = "keyboard_interrupt"
        stop_event.set()
    finally:
        stop_event.set()
        stop_psychopy(psychopy_proc, raw)

    for th in threads:
        th.join(timeout=5.0)

    if not psychopy_failed and marker_state.get("block_end_received"):
        stop_reason = "block_end"

    ended_at = time.time()
    bad_reasons = {
        "psychopy_start_failed", "psychopy_exited_early", "psychopy_startup_timeout",
        "psychopy_marker_stall", "block_timeout", "keyboard_interrupt",
        "external_stop_before_psychopy",
    }
    ok = not psychopy_failed and stop_reason not in bad_reasons
    print(f"[main] Bloco encerrado. reason={stop_reason} | ok={ok}")
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "ok": ok,
        "stop_reason": stop_reason,
        "psychopy_returncode": None if psychopy_proc is None else psychopy_proc.poll(),
    }


def cv_mean(res: dict) -> float:
    accs = res.get("accs_cv", []) if isinstance(res, dict) else []
    return float(np.mean(accs)) if len(accs) else float("nan")


def run_check_same_pair(cfg: AppConfig, raw: dict, markers_file: str, signal_file: str) -> None:
    if not protocol(raw).get("auto_check_data", True):
        return
    print("\n[main] Gerando QC do bloco recém-gravado...")
    try:
        sig = inspect.signature(run_check_data)
        kwargs = {}
        if "markers_file" in sig.parameters:
            kwargs["markers_file"] = markers_file
        if "signal_file" in sig.parameters:
            kwargs["signal_file"] = signal_file
        run_check_data(cfg, **kwargs)
    except Exception as exc:
        print(f"[main] check_data falhou: {type(exc).__name__}: {exc}")


def train_and_check_current_block(cfg: AppConfig, raw: dict, pair: tuple[str, str] | None) -> dict:
    if pair is None:
        print("[main] Não encontrei o par markers/signal recém-gravado.")
        return {"acc_mean": float("nan"), "res": None}
    markers_file, signal_file = pair
    print("\n[main] Treinando classificador no bloco recém-gravado:")
    print(f"  markers: {os.path.basename(markers_file)}")
    print(f"  signal : {os.path.basename(signal_file)}")
    try:
        res = run_decoder_calibration(cfg, markers_file=markers_file, signal_file=signal_file)
        acc = cv_mean(res)
        print(f"[main] Acurácia média CV = {acc:.3f}")
        if res.get("model_dir"):
            print(f"[main] Pasta do modelo: {res['model_dir']}")
        if res.get("representation_manifest_path"):
            print(f"[main] JSON do mapa PCA para Unity: {res['representation_manifest_path']}")
    except Exception as exc:
        print(f"[main] Calibração falhou: {type(exc).__name__}: {exc}")
        return {"acc_mean": float("nan"), "res": None}
    run_check_same_pair(cfg, raw, markers_file, signal_file)
    return {"acc_mean": acc, "res": res}


def default_training_action(raw: dict, acc_mean: float) -> str:
    threshold = protocol(raw).get("min_cv_accuracy")
    if threshold is None or not np.isfinite(acc_mean):
        return "r" if not np.isfinite(acc_mean) else "s"
    return "s" if acc_mean >= float(threshold) else "r"


def run_training_stage(cfg: AppConfig, raw: dict, label: str, session_type: str) -> str:
    cfg_stage = set_session_type(cfg, session_type)
    folder = data_dir(cfg_stage, session_type, "train")
    existing = find_marker_signal_pairs(folder)
    if existing:
        print(f"\n[main] {len(existing)} bloco(s) prévio(s) em {folder}.")
    block_n = len(existing) + 1

    while True:
        if not ask(f"Iniciar {label} bloco {block_n}?", default=True):
            return "stop"
        before = find_marker_signal_pairs(folder)
        info = run_block(cfg_stage, raw, f"{label} {block_n}", "train")
        if not info.get("ok", True):
            print(f"[main] Bloco incompleto ({info.get('stop_reason')}); não será usado no treino.")
            action = ask_choice("O que fazer agora?", {"r": "refazer em novo bloco", "f": "finalizar sessão"}, "r")
            if action == "r":
                block_n += 1
                continue
            return "stop"

        pair = detect_current_pair(folder, before, info["started_at"])
        out = train_and_check_current_block(cfg_stage, raw, pair)
        acc = out["acc_mean"]
        threshold = protocol(raw).get("min_cv_accuracy")
        crit = f" | critério={float(threshold):.2f}" if threshold is not None else ""
        acc_txt = "nan" if not np.isfinite(acc) else f"{acc:.3f}"
        action = ask_choice(
            f"\nResultado {label}: CV={acc_txt}{crit}. O que fazer?",
            {"r": "refazer este bloco", "s": "seguir para a próxima fase", "f": "finalizar sessão"},
            default_training_action(raw, acc),
        )
        if action == "r":
            block_n += 1
            continue
        return "next" if action == "s" else "stop"


def _model_meta_path(model_ref: str) -> str:
    if os.path.isdir(model_ref) and os.path.exists(os.path.join(model_ref, "classifier.pkl")):
        return os.path.join(model_ref, "model_meta.json")
    return model_ref + "_meta.json"


def read_model_meta(model_ref: str) -> dict:
    try:
        with open(_model_meta_path(model_ref), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def model_session_types(raw: dict, em_type: str, im_type: str) -> list[str]:
    val = protocol(raw).get("model_session_types", [em_type, im_type])
    if isinstance(val, str):
        val = [val]
    out: list[str] = []
    for item in val or []:
        if item and item not in out:
            out.append(str(item))
    return out


def _map_path_from_meta(model_ref: str, meta: dict) -> str | None:
    value = meta.get("pca_map_json") if isinstance(meta, dict) else None
    if value:
        p = Path(str(value))
        if not p.is_absolute():
            p = (Path(model_ref) / p).resolve()
        if p.exists():
            return str(p)

    # Novo formato: _model/<run_id> e mapa visível na raiz de train/.
    model_path = Path(model_ref)
    if model_path.is_dir() and model_path.parent.name == "_model":
        candidate = model_path.parent.parent / f"pca_map_{model_path.name}.json"
        if candidate.exists():
            return str(candidate)

    # Compatibilidade com v3/v3.1: pca_map dentro da própria pasta do modelo.
    candidate = model_path / "pca_map.json"
    if candidate.exists():
        return str(candidate)
    return None




def _map_png_path_from_meta(model_ref: str, meta: dict, json_path: str | None = None) -> str | None:
    value = meta.get("pca_map_png") if isinstance(meta, dict) else None
    if value:
        p = Path(str(value))
        if not p.is_absolute():
            p = (Path(model_ref) / p).resolve()
        if p.exists():
            return str(p)

    # Prefer the PNG paired with the resolved JSON. This covers the current
    # pca_map_<run_id>.json/.png convention and legacy pca_map.json/.png.
    if json_path:
        candidate = Path(json_path).with_suffix(".png")
        if candidate.exists():
            return str(candidate)

    model_path = Path(model_ref)
    candidate = model_path / "pca_map.png"
    if candidate.exists():
        return str(candidate)
    return None

def list_models(cfg: AppConfig, session_types: list[str]) -> list[dict]:
    rows: list[dict] = []
    root = session_root(cfg)
    seen: set[str] = set()

    def add_model(session_type: str, model_ref: Path, clf: Path) -> None:
        key = str(model_ref.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        meta = read_model_meta(str(model_ref))
        pca_map_json = _map_path_from_meta(str(model_ref), meta)
        rows.append({
            "session_type": session_type,
            "ref": str(model_ref),
            "clf_path": str(clf),
            "acc": ((meta.get("cv", {}) or {}).get("acc_mean")),
            "pca_map": pca_map_json,
            "pca_map_png": _map_png_path_from_meta(str(model_ref), meta, pca_map_json),
            "mtime": clf.stat().st_mtime,
        })

    for session_type in session_types:
        # Formato atual: artefatos técnicos ficam dentro da própria pasta de treino.
        train_dir = data_dir(cfg, session_type, "train")
        model_root = train_dir / "_model"
        if model_root.exists():
            for d in model_root.iterdir():
                clf = d / "classifier.pkl"
                if d.is_dir() and clf.exists():
                    add_model(session_type, d, clf)

        # Compatibilidade com v3/v3.1: S#/models/<fase>/<run_id>/.
        legacy_model_root = root / "models" / session_type
        if legacy_model_root.exists():
            for d in legacy_model_root.iterdir():
                clf = d / "classifier.pkl"
                if d.is_dir() and clf.exists():
                    add_model(session_type, d, clf)

        # Compatibilidade com formato ainda mais antigo: artefatos prefixados no train/.
        for clf_path in glob.glob(str(train_dir / "*_classifier.pkl")):
            prefix = re.sub(r"_classifier\.pkl$", "", clf_path)
            meta = read_model_meta(prefix)
            rows.append({
                "session_type": session_type,
                "ref": prefix,
                "clf_path": clf_path,
                "acc": ((meta.get("cv", {}) or {}).get("acc_mean")),
                "pca_map": None,
                "mtime": os.path.getmtime(clf_path),
            })

    rows.sort(key=lambda row: row["mtime"], reverse=True)
    return rows


def choose_model(cfg: AppConfig, raw: dict, em_type: str, im_type: str) -> str:
    explicit = protocol(raw).get("online_model_prefix", "ask")
    if explicit and str(explicit).lower() not in {"ask", "latest", "auto"}:
        return str(explicit)

    rows = list_models(cfg, model_session_types(raw, em_type, im_type))
    if not rows:
        raise FileNotFoundError("Não encontrei nenhum modelo treinado para iniciar o online.")
    if str(explicit).lower() in {"latest", "auto"}:
        chosen = rows[0]
        print(f"[main] Modelo online automático: {chosen['ref']}")
        if chosen.get("pca_map"):
            print(f"[main] pca_map.json: {chosen['pca_map']}")
        return chosen["ref"]

    print("\n===== MODELOS DISPONÍVEIS PARA O ONLINE =====")
    for i, row in enumerate(rows, 1):
        acc_txt = "sem CV" if row["acc"] is None else f"CV={float(row['acc']):.3f}"
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["mtime"]))
        print(f"  [{i}] {row['session_type']} | {acc_txt} | {stamp} | {os.path.basename(row['ref'])}")
        if row.get("pca_map"):
            print(f"      PCA map: {row['pca_map']}")
    while True:
        ans = input("Escolha o modelo [1 = mais recente]: ").strip()
        idx = 1 if not ans else int(ans) if ans.isdigit() else -1
        if 1 <= idx <= len(rows):
            return rows[idx - 1]["ref"]
        print("Número inválido.")


def publish_selected_pca_map(cfg: AppConfig, model_ref: str) -> Path | None:
    """Publica o par pca_map.json + pca_map.png em S#/online/."""
    meta = read_model_meta(model_ref)
    src = _map_path_from_meta(model_ref, meta)
    if not src:
        print(f"[main] Modelo sem pca_map.json localizável: {model_ref}")
        return None

    src_json = Path(src)
    src_png_value = _map_png_path_from_meta(model_ref, meta, str(src_json))
    if not src_png_value:
        print(f"[main] Modelo sem pca_map.png correspondente: {model_ref}")
        return None
    src_png = Path(src_png_value)

    online_root = session_root(cfg) / "online"
    online_root.mkdir(parents=True, exist_ok=True)
    dst_json = online_root / "pca_map.json"
    dst_png  = online_root / "pca_map.png"
    shutil.copy2(src_png, dst_png)

    # Ao publicar no caminho canônico, ajusta a referência interna para o PNG
    # canônico que está no mesmo diretório.
    try:
        manifest = json.loads(src_json.read_text(encoding="utf-8"))
        manifest["map_image"] = dst_png.name
        dst_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        shutil.copy2(src_json, dst_json)

    # A rastreabilidade fica fora da vista principal dos dados online.
    tech_dir = online_root / "_model"
    tech_dir.mkdir(parents=True, exist_ok=True)
    pointer = {
        "selected_model": str(model_ref),
        "pca_map_json": str(dst_json),
        "pca_map_png": str(dst_png),
        "source_pca_map_json": str(src_json),
        "source_pca_map_png": str(src_png),
    }
    (tech_dir / "selected_model.json").write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[main] Mapa PCA ativo para o Unity:")
    print(f"       JSON: {dst_json}")
    print(f"       PNG : {dst_png}")
    return dst_json


def start_debug_plot(cfg: AppConfig, raw: dict, model_ref: str | None = None) -> subprocess.Popen | None:
    dcfg = debug_plot_cfg(raw)
    if not bool(dcfg.get("enabled", False)):
        return None
    script = PROJECT_DIR / str(dcfg.get("script", "tools/plot_decoder_realtime.py"))
    if not script.exists():
        print(f"[main] Debug plot habilitado, mas não encontrado: {script}")
        return None
    cmd = [
        sys.executable, str(script),
        "--decoder-name", getattr(cfg.decoder, "outlet_name", "Signal"),
        "--decoder-type", getattr(cfg.decoder, "outlet_type", "BCI"),
        "--marker-name", getattr(cfg.lsl, "marker_name", "GrazMI_Markers"),
        "--marker-type", getattr(cfg.lsl, "marker_type", "Markers"),
    ]
    meta = read_model_meta(model_ref) if model_ref else {}
    xlim, ylim = meta.get("pca_train_xlim"), meta.get("pca_train_ylim")
    if xlim and ylim:
        cmd += ["--pca-xlim", str(xlim[0]), str(xlim[1]), "--pca-ylim", str(ylim[0]), str(ylim[1])]
    print("[main] Abrindo janela diagnóstica Python (debug_plot.enabled=true).")
    flags = 0
    if os.name == "nt" and bool(dcfg.get("new_console", False)):
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return subprocess.Popen(cmd, creationflags=flags)


def stop_debug_plot(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def online_modes(raw: dict) -> list[dict]:
    modes = protocol(raw).get("online_modes")
    if not isinstance(modes, list) or not modes:
        modes = [
            {"name": "pca_feedback", "label": "Online com feedback PCA", "session_type": "IM_online_PCA", "feedback_mode": "pca"},
            {"name": "no_feedback", "label": "Online sem feedback", "session_type": "IM_online_sem_feedback", "feedback_mode": "none"},
        ]
    out = []
    for i, mode in enumerate(modes, 1):
        if not isinstance(mode, dict):
            continue
        out.append({
            "name": str(mode.get("name", f"online_{i}")),
            "label": str(mode.get("label", mode.get("name", f"Online {i}"))),
            "session_type": str(mode.get("session_type", f"IM_online_{i}")),
            "feedback_mode": str(mode.get("feedback_mode", "none")).lower(),
        })
    return out


def run_online_mode(cfg: AppConfig, raw: dict, mode_cfg: dict, model_ref: str, is_last: bool) -> str:
    session_type = mode_cfg["session_type"]
    label = mode_cfg["label"]
    feedback_mode = mode_cfg["feedback_mode"]
    cfg_online = set_session_type(cfg, session_type)
    folder = data_dir(cfg_online, session_type, "online")
    # Ambos os modos compartilham S#/online; o session_type no nome distingue os arquivos.
    existing = [
        pair for pair in find_marker_signal_pairs(folder)
        if f"_{session_type}_" in os.path.basename(pair[0])
    ]
    if existing:
        print(f"\n[main] {len(existing)} bloco(s) prévio(s) de '{label}' em {folder}.")
    block_n = len(existing) + 1

    while True:
        if not ask(f"Iniciar {label} — bloco {block_n}?", default=True):
            return "stop"
        plot_proc = start_debug_plot(cfg_online, raw, model_ref)
        try:
            info = run_block(
                cfg_online, raw, f"{label} {block_n}", "online",
                decoder=True, model_ref=model_ref, feedback_mode=feedback_mode,
            )
        finally:
            stop_debug_plot(plot_proc)

        if not info.get("ok", True):
            print(f"[main] Bloco online incompleto: {info.get('stop_reason')}.")
            action = ask_choice("O que fazer?", {"r": "refazer este modo", "f": "finalizar sessão"}, "r")
            if action == "r":
                block_n += 1
                continue
            return "stop"

        next_label = "encerrar protocolo" if is_last else "seguir para o próximo modo online"
        action = ask_choice(
            f"\n{label} encerrado. O que fazer?",
            {"r": f"repetir {label}", "s": next_label, "f": "finalizar sessão"},
            "s",
        )
        if action == "r":
            block_n += 1
            continue
        return "next" if action == "s" else "stop"


def show_protocol_summary(cfg: AppConfig, raw: dict, start_phase: str) -> None:
    print("\n===== PROTOCOLO =====")
    print(f"Sujeito: {cfg.experiment.subject_id} | Sessão: S{cfg.experiment.session_id}")
    print(f"Fase inicial: {start_phase}")
    sequence_cfg = psychopy_cfg(raw).get("stim_sequence", "experiment/stims_sequence.csv")
    sequence_path = PROJECT_DIR / str(sequence_cfg)
    classes = classes_from_sequence(sequence_path)
    if classes:
        print("Classes na stims_sequence: " + ", ".join(display_name(c) for c in classes))
        print("  (remover BOTH_MI_STIM da stims_sequence volta automaticamente ao modo de 2 classes)")
    print("Modos online:")
    for mode in online_modes(raw):
        fb = "feedback PCA" if mode["feedback_mode"] == "pca" else "sem feedback"
        print(f"  - {mode['label']} | {mode['session_type']} | {fb}")
    print(f"Debug plot Python: {'ON' if debug_plot_cfg(raw).get('enabled', False) else 'OFF'}")
    ccfg = raw.get("control", {}) or {}
    if bool(ccfg.get("enabled", True)):
        print(
            "Controle Unity: densidade PCA + probabilidades | "
            f"entrada HDR={float(ccfg.get('entry_density_mass', 0.50)):.2f} | "
            f"hold HDR={float(ccfg.get('hold_density_mass', 0.80)):.2f}"
        )
        print(
            f"Stream controle: {ccfg.get('outlet_name', 'GrazMI_Control')} / "
            f"{ccfg.get('outlet_type', 'BCIControl')}"
        )
    else:
        print("Controle Unity: OFF")
    print()


def main() -> None:
    cfg, raw = load_cfg()
    p = protocol(raw)
    em_type = str(p.get("motor_session_type", "EM_treino"))
    im_type = str(p.get("imagery_session_type", "IM_treino"))
    start_phase = normalize_phase(p.get("start_phase", "execution"))
    show_protocol_summary(cfg, raw, start_phase)

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
        model_ref = choose_model(cfg, raw, em_type, im_type)
        publish_selected_pca_map(cfg, model_ref)
        modes = online_modes(raw)
        for i, mode_cfg in enumerate(modes):
            result = run_online_mode(cfg, raw, mode_cfg, model_ref, is_last=(i == len(modes) - 1))
            if result == "stop":
                print("\n[main] Sessão finalizada durante online.")
                return

    print("\n[main] Protocolo finalizado.")


if __name__ == "__main__":
    main()
