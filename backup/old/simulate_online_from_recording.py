# -*- coding: utf-8 -*-
"""
simulate_online_batch_validate.py

Fluxo simples e fixo:

1) escolher uma pasta com arquivos CSV ja coletados;
2) encontrar todos os pares *_markers_*.csv + *_signal_*.csv;
3) para CADA par encontrado:
   - rodar train_decoder / run_calibration;
   - mostrar e guardar a acuracia de validacao cruzada;
   - gerar os arquivos do classificador;
   - rodar check_data no mesmo par e salvar imagens;
4) listar todos os dados com numeracao, CV e status;
5) escolher um dado/modelo para classificador;
6) escolher um dado para replay/inferencia online;
7) esperar OK;
8) abrir plot_decoder_realtime.py em processo separado;
9) rodar o decoder e transmitir o dado escolhido via LSL com suas marcacoes.

Uso:
    python simulate_online_batch_validate.py
    python simulate_online_batch_validate.py --folder "C:\\dados\\sessao" --replay-fs 256
    python simulate_online_batch_validate.py --no-plot
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pylsl import StreamInfo, StreamOutlet, cf_float32, cf_int32, cf_string

from config_models       import load_config, AppConfig
from decoder_calibration import run_calibration as train_decoder
from check_data          import run_check_data
from online_inference    import run_realtime_decoder


YES = {"s", "sim", "y", "yes"}
NO  = {"n", "nao", "não", "no"}

DEFAULT_EEG_NAME     = "gHIamp_EEG"
DEFAULT_EEG_TYPE     = "EEG"
DEFAULT_MARKER_NAME  = "GrazMI_Markers"
DEFAULT_MARKER_TYPE  = "Markers"
DEFAULT_DECODER_NAME = "GrazMI_DecoderDebug"
DEFAULT_DECODER_TYPE = "BCI"

LABEL_TO_CODE = {
    "BASELINE":      1,
    "ATTENTION":     2,
    "LEFT_MI_STIM":  3,
    "RIGHT_MI_STIM": 4,
    "ATTEMPT":       5,
    "REST":          6,
    "BLOCK_END":     99,
}


def ask_yes_no(msg: str, default: bool = True) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    while True:
        ans = input(f"{msg} {suffix}: ").strip().lower()
        if ans == "":
            return default
        if ans in YES:
            return True
        if ans in NO:
            return False
        print("Responda apenas com s ou n.")


def choose_folder_gui() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Selecione a pasta com os CSVs coletados")
        root.destroy()
        return Path(folder) if folder else None
    except Exception:
        return None


def choose_folder_cli() -> Path:
    while True:
        txt = input("Pasta com os dados coletados: ").strip().strip('"')
        if txt and Path(txt).exists():
            return Path(txt)
        print("Pasta inválida.")


def choose_valid_index(rows: list[dict], prompt: str, require_model: bool = False, default: int = 1) -> int:
    valid = {row["idx"] for row in rows if (not require_model) or row.get("model_prefix")}
    while True:
        ans = input(f"{prompt} [default={default}]: ").strip()
        idx = default if ans == "" else int(ans) if ans.isdigit() else -1
        if idx in valid:
            return idx - 1
        if require_model:
            print("Número inválido ou sem classificador válido.")
        else:
            print("Número inválido.")


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


def set_session_type(cfg: AppConfig, session_type: str | None) -> AppConfig:
    if not session_type:
        return cfg
    out = copy.deepcopy(cfg)
    out.experiment.session_type = session_type
    return out


def first_value(*values, default=None):
    for value in values:
        if value is not None and value != "":
            return value
    return default


def get_attr(obj: Any, names: list[str], default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None and value != "":
                return value
    return default


def lsl_names(cfg: AppConfig, raw: dict, args) -> dict[str, str]:
    p = raw.get("protocol", {}) or {}
    return {
        "eeg_name":     str(first_value(args.eeg_name, p.get("eeg_stream_name"), get_attr(cfg.lsl, ["eeg_name", "signal_name", "stream_name", "eeg_stream_name"], None), default=DEFAULT_EEG_NAME)),
        "eeg_type":     str(first_value(args.eeg_type, p.get("eeg_stream_type"), get_attr(cfg.lsl, ["eeg_type", "signal_type", "stream_type", "eeg_stream_type"], None), default=DEFAULT_EEG_TYPE)),
        "marker_name":  str(first_value(args.marker_name, p.get("marker_stream_name"), get_attr(cfg.lsl, ["marker_name", "marker_stream_name"], None), default=DEFAULT_MARKER_NAME)),
        "marker_type":  str(first_value(args.marker_type, p.get("marker_stream_type"), get_attr(cfg.lsl, ["marker_type", "marker_stream_type"], None), default=DEFAULT_MARKER_TYPE)),
        "decoder_name": str(first_value(args.decoder_name, p.get("decoder_debug_name"), get_attr(getattr(cfg, "decoder", None), ["outlet_name"], None), default=DEFAULT_DECODER_NAME)),
        "decoder_type": str(first_value(args.decoder_type, p.get("decoder_debug_type"), default=DEFAULT_DECODER_TYPE)),
    }


def find_pairs(folder: Path, recursive: bool = True) -> list[tuple[Path, Path]]:
    pattern = "**/*markers_*.csv" if recursive else "*markers_*.csv"
    pairs: list[tuple[Path, Path]] = []

    for marker_file in sorted(folder.glob(pattern)):
        if "_markers_" not in marker_file.name:
            continue
        prefix = marker_file.name.split("_markers_", 1)[0]
        signal_candidates = glob.glob(str(marker_file.parent / f"{prefix}_signal_*.csv"))
        if not signal_candidates:
            continue
        signal_file = Path(max(signal_candidates, key=os.path.getmtime))
        pairs.append((marker_file.resolve(), signal_file.resolve()))

    return sorted(pairs, key=lambda pair: pair[1].stat().st_mtime, reverse=True)


def count_attempts(marker_file: Path) -> tuple[int | None, int | None]:
    try:
        df = pd.read_csv(marker_file)
    except Exception:
        return None, None

    if "label" in df.columns:
        labels = df["label"].astype(str).str.strip().str.upper().tolist()
    elif "event" in df.columns:
        labels = df["event"].astype(str).str.strip().str.upper().tolist()
    elif "code" in df.columns:
        codes = pd.to_numeric(df["code"], errors="coerce").fillna(-999).astype(int).tolist()
        cmap = {1: "BASELINE", 2: "ATTENTION", 3: "LEFT_MI_STIM", 4: "RIGHT_MI_STIM", 5: "ATTEMPT", 6: "REST", 99: "BLOCK_END"}
        labels = [cmap.get(c, str(c)).upper() for c in codes]
    else:
        return None, None

    left = 0
    right = 0
    last = None
    for lab in labels:
        if lab in {"LEFT_MI_STIM", "RIGHT_MI_STIM"}:
            last = lab
        elif lab == "ATTEMPT" and last == "LEFT_MI_STIM":
            left += 1
        elif lab == "ATTEMPT" and last == "RIGHT_MI_STIM":
            right += 1
    return left, right


def model_prefix_from_result(res: dict, signal_file: Path) -> str:
    clf_path = res.get("clf_path") if isinstance(res, dict) else None
    if clf_path:
        return re.sub(r"_classifier\.pkl$", "", str(clf_path))
    return str(signal_file.with_suffix(""))


def update_meta_with_pca_range(model_prefix: str | None, res: dict) -> dict:
    """Salva limites do PCA de treino no *_meta.json para o plot realtime usar o mesmo plano."""
    if not model_prefix or not isinstance(res, dict) or "Xp" not in res:
        return {}
    Xp = np.asarray(res["Xp"], dtype=float)
    if Xp.ndim != 2 or Xp.shape[0] == 0:
        return {}
    x = Xp[:, 0]
    y = Xp[:, 1] if Xp.shape[1] > 1 else np.zeros_like(x)
    pad_x = 0.10 * max(float(np.nanmax(x) - np.nanmin(x)), 1e-6)
    pad_y = 0.10 * max(float(np.nanmax(y) - np.nanmin(y)), 1e-6)
    out = {
        "pca_train_xlim": [float(np.nanmin(x) - pad_x), float(np.nanmax(x) + pad_x)],
        "pca_train_ylim": [float(np.nanmin(y) - pad_y), float(np.nanmax(y) + pad_y)],
    }
    meta_path = Path(model_prefix + "_meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.update(out)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[meta] Não consegui atualizar limites PCA em {meta_path}: {type(exc).__name__}: {exc}")
    return out


def read_model_meta(model_prefix: str | None) -> dict:
    if not model_prefix:
        return {}
    meta_path = Path(str(model_prefix) + "_meta.json")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        return {}


def run_train_decoder_for_pair(cfg: AppConfig, pair: tuple[Path, Path]) -> dict:
    marker_file, signal_file = pair
    print("\n===== TRAIN_DECODER =====")
    print(f"Markers: {marker_file}")
    print(f"Signal : {signal_file}")

    out = {
        "model_prefix": None,
        "accs": [],
        "acc_mean": float("nan"),
        "acc_std": float("nan"),
        "train_status": "falhou",
        "train_error": "",
    }

    try:
        res = train_decoder(cfg, markers_file=str(marker_file), signal_file=str(signal_file))
        accs = res.get("accs_cv", []) if isinstance(res, dict) else []
        accs = [float(a) for a in accs]
        out["accs"] = accs
        out["acc_mean"] = float(np.mean(accs)) if accs else float("nan")
        out["acc_std"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        out["model_prefix"] = model_prefix_from_result(res, signal_file)
        pca_limits = update_meta_with_pca_range(out["model_prefix"], res)
        out["pca_train_xlim"] = pca_limits.get("pca_train_xlim")
        out["pca_train_ylim"] = pca_limits.get("pca_train_ylim")
        out["train_status"] = "ok"

        folds = " | ".join(f"{a:.3f}" for a in accs) if accs else "sem valores"
        print("\n[CV] Folds:", folds)
        print(f"[CV] Média={out['acc_mean']:.3f} | DP={out['acc_std']:.3f}")
        print(f"[CV] Modelo={out['model_prefix']}")
    except Exception as exc:
        out["train_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[train_decoder] Falhou: {out['train_error']}")

    return out


def run_check_for_pair(cfg: AppConfig, pair: tuple[Path, Path]) -> dict:
    marker_file, signal_file = pair
    print("\n===== CHECK_DATA =====")
    print(f"Markers: {marker_file}")
    print(f"Signal : {signal_file}")

    out = {
        "check_status": "falhou",
        "check_error": "",
        "stack_png": str(signal_file.with_suffix("")) + "_stack_hp.png",
        "epochs_png": str(signal_file.with_suffix("")) + "_epochs_hp.png",
    }

    try:
        run_check_data(
            cfg,
            mode="train",
            markers_file=str(marker_file),
            signal_file=str(signal_file),
            save_png=True,
        )
        out["check_status"] = "ok"
        print("[check_data] Figuras:")
        print(f"  {out['stack_png']}")
        print(f"  {out['epochs_png']}")
    except Exception as exc:
        out["check_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[check_data] Falhou: {out['check_error']}")

    return out


def process_all_pairs(cfg: AppConfig, pairs: list[tuple[Path, Path]], root: Path, skip_check: bool = False) -> list[dict]:
    rows: list[dict] = []
    print("\n===== PROCESSAMENTO DE TODOS OS DADOS =====")
    print("Para cada arquivo: train_decoder -> CV/modelo -> check_data/imagens.")

    for idx, pair in enumerate(pairs, start=1):
        marker_file, signal_file = pair
        left, right = count_attempts(marker_file)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(signal_file.stat().st_mtime))
        try:
            rel_dir = str(signal_file.parent.relative_to(root))
        except Exception:
            rel_dir = str(signal_file.parent)

        print(f"\n\n########## DADO {idx}/{len(pairs)} ##########")
        train_info = run_train_decoder_for_pair(cfg, pair)
        check_info = {"check_status": "pulado", "check_error": "", "stack_png": "", "epochs_png": ""}
        if not skip_check:
            check_info = run_check_for_pair(cfg, pair)

        row = {
            "idx": idx,
            "marker_file": marker_file,
            "signal_file": signal_file,
            "signal_name": signal_file.name,
            "rel_dir": rel_dir,
            "mtime": signal_file.stat().st_mtime,
            "stamp": stamp,
            "left": left,
            "right": right,
            **train_info,
            **check_info,
        }
        rows.append(row)

    return rows


def print_validation_table(rows: list[dict]) -> None:
    print("\n===== DADOS PROCESSADOS =====")
    header = f"{'#':>3} | {'CV':>7} | {'DP':>6} | {'treino':<6} | {'check':<6} | {'L/R':>7} | {'data/hora':<19} | arquivo"
    print(header)
    print("-" * len(header))

    for row in rows:
        acc_txt = "nan" if not np.isfinite(row["acc_mean"]) else f"{row['acc_mean']:.3f}"
        std_txt = "nan" if not np.isfinite(row["acc_std"]) else f"{row['acc_std']:.3f}"
        lr_txt = "?/ ?" if row["left"] is None or row["right"] is None else f"{row['left']}/{row['right']}"
        prefix = "" if row["rel_dir"] in {"", "."} else f"{row['rel_dir']} | "
        print(f"{row['idx']:>3} | {acc_txt:>7} | {std_txt:>6} | {row['train_status']:<6} | {row['check_status']:<6} | {lr_txt:>7} | {row['stamp']:<19} | {prefix}{row['signal_name']}")

    errors = [row for row in rows if row["train_status"] != "ok" or row["check_status"] not in {"ok", "pulado"}]
    if errors:
        print("\n===== AVISOS =====")
        for row in errors:
            if row["train_status"] != "ok":
                print(f"[{row['idx']:02d}] train_decoder falhou: {row['train_error']}")
            if row["check_status"] not in {"ok", "pulado"}:
                print(f"[{row['idx']:02d}] check_data falhou: {row['check_error']}")


def read_signal(signal_file: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(signal_file, low_memory=False)
    if "lsl_time_s" in df.columns:
        t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)
    elif "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
    else:
        t = np.arange(len(df), dtype=float)

    ch_cols = [c for c in df.columns if re.match(r"(?i)^ch\d+$", str(c))]
    if not ch_cols:
        ignore = {"iso_time", "lsl_time_s", "time_s", "local_recv_s", "timestamp"}
        ch_cols = [c for c in df.columns if c not in ignore]
    if not ch_cols:
        raise KeyError("Não encontrei colunas de canais no CSV de sinal.")

    X = df[ch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    ok = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    return t[ok], X[ok], [str(c) for c in ch_cols]


def estimate_fs(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if len(dt) else 256.0


def read_markers(marker_file: Path) -> tuple[np.ndarray, list[Any], bool]:
    df = pd.read_csv(marker_file)
    if "lsl_time_s" in df.columns:
        t = pd.to_numeric(df["lsl_time_s"], errors="coerce").to_numpy(float)
    elif "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
    else:
        t = np.arange(len(df), dtype=float)

    numeric = False
    if "code" in df.columns:
        code = pd.to_numeric(df["code"], errors="coerce")
        if code.notna().all():
            values = code.astype(int).tolist()
            numeric = True
        else:
            values = df["code"].astype(str).str.strip().tolist()
    elif "label" in df.columns:
        labels = df["label"].astype(str).str.strip().str.upper().tolist()
        mapped = [LABEL_TO_CODE.get(lab, None) for lab in labels]
        if all(v is not None for v in mapped):
            values = [int(v) for v in mapped]
            numeric = True
        else:
            values = labels
    elif "event" in df.columns:
        labels = df["event"].astype(str).str.strip().str.upper().tolist()
        mapped = [LABEL_TO_CODE.get(lab, None) for lab in labels]
        if all(v is not None for v in mapped):
            values = [int(v) for v in mapped]
            numeric = True
        else:
            values = labels
    else:
        raise KeyError("CSV de marcadores precisa ter coluna code, label ou event.")

    ok = np.isfinite(t)
    return t[ok], [values[i] for i in np.where(ok)[0]], numeric


class CSVReplayLSL:
    def __init__(self, pair: tuple[Path, Path], names: dict[str, str], replay_fs: float | None, block_sec: float, speed: float, append_block_end: bool, cfg: AppConfig | None = None):
        self.marker_file, self.signal_file = pair
        self.names = names
        self.block_sec = float(block_sec)
        self.speed = max(float(speed), 1e-6)
        self.append_block_end = bool(append_block_end)

        self.t_sig, self.X, self.ch_names = read_signal(self.signal_file)
        self.t_mark, self.markers, self.markers_are_numeric = read_markers(self.marker_file)
        self.fs = float(replay_fs) if replay_fs is not None else float(getattr(cfg, "fs_hz", estimate_fs(self.t_sig)) if cfg is not None else estimate_fs(self.t_sig))

        if len(self.t_sig) < 2:
            raise ValueError("Sinal com amostras insuficientes para replay.")

        self.t0 = float(self.t_sig[0])
        self.sig_rel = np.maximum(0.0, self.t_sig - self.t0)
        self.mark_rel = np.maximum(0.0, self.t_mark - self.t0)
        self.marker_sample_idx = self._marker_indices()
        self.eeg_outlet = self._make_eeg_outlet()
        self.marker_outlet = self._make_marker_outlet()

    def _marker_indices(self) -> np.ndarray:
        rel_sig = self.t_sig - self.t0
        rel_mark = self.t_mark - self.t0
        rel_mark = np.maximum(rel_mark, 0.0)
        idx = np.searchsorted(rel_sig, rel_mark)
        return np.clip(idx, 0, len(self.X) - 1).astype(int)

    def _make_eeg_outlet(self) -> StreamOutlet:
        info = StreamInfo(
            self.names["eeg_name"],
            self.names["eeg_type"],
            int(self.X.shape[1]),
            float(self.fs),
            cf_float32,
            f"CSVReplayEEG_{os.getpid()}_{time.time_ns()}",
        )
        channels = info.desc().append_child("channels")
        for ch_name in self.ch_names:
            ch = channels.append_child("channel")
            ch.append_child_value("label", str(ch_name))
            ch.append_child_value("unit", "uV")
            ch.append_child_value("type", "EEG")
        return StreamOutlet(info)

    def _make_marker_outlet(self) -> StreamOutlet:
        fmt = cf_int32 if self.markers_are_numeric else cf_string
        info = StreamInfo(
            self.names["marker_name"],
            self.names["marker_type"],
            1,
            0,
            fmt,
            f"CSVReplayMarkers_{os.getpid()}_{time.time_ns()}",
        )
        return StreamOutlet(info)

    def _push_marker(self, value: Any) -> None:
        if self.markers_are_numeric:
            self.marker_outlet.push_sample([int(value)])
        else:
            self.marker_outlet.push_sample([str(value)])

    def replay_markers(self, start_wall: float, stop_event: threading.Event) -> None:
        # Usa o tempo real dos marcadores no CSV; speed só comprime/dilata o tempo.
        for idx, rel_t, value in zip(self.marker_sample_idx, self.mark_rel, self.markers):
            if stop_event.is_set():
                return
            target = start_wall + float(rel_t) / self.speed
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self._push_marker(value)
            print(f"[markers] sample={idx:08d} | t_csv={float(rel_t):7.3f}s | marker={value}")

    def replay_signal(self, start_wall: float, stop_event: threading.Event) -> None:
        # Envia chunks, mas agenda cada chunk pelo lsl_time_s do próprio CSV.
        # Isso preserva alinhamento sinal-marcador mesmo se houver jitter ou replay_fs diferente.
        n = len(self.X)
        chunk = max(1, int(round(self.fs * self.block_sec)))
        for i0 in range(0, n, chunk):
            if stop_event.is_set():
                return
            i1 = min(i0 + chunk, n)
            target = start_wall + float(self.sig_rel[i0]) / self.speed
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self.eeg_outlet.push_chunk(self.X[i0:i1].astype(np.float32).tolist())
            if (i0 // chunk) % 20 == 0:
                print(f"[signal ] samples={i1:08d}/{n:08d} | t_csv={float(self.sig_rel[i0]):7.3f}s")

    def run(self, stop_event: threading.Event, start_delay_s: float) -> None:
        print("\n===== REPLAY LSL =====")
        print(f"Sinal     : {self.signal_file}")
        print(f"Marcadores: {self.marker_file}")
        print(f"EEG LSL   : {self.names['eeg_name']} / {self.names['eeg_type']} | fs={self.fs:.2f} Hz | canais={self.X.shape[1]}")
        print(f"Marker LSL: {self.names['marker_name']} / {self.names['marker_type']} | numeric={self.markers_are_numeric}")
        print(f"Velocidade: {self.speed:.2f}x")

        start_wall = time.perf_counter() + float(start_delay_s)
        th_sig = threading.Thread(target=self.replay_signal, args=(start_wall, stop_event), daemon=True)
        th_mrk = threading.Thread(target=self.replay_markers, args=(start_wall, stop_event), daemon=True)
        th_sig.start()
        th_mrk.start()
        th_sig.join()
        th_mrk.join(timeout=2.0)

        if self.append_block_end and not stop_event.is_set():
            time.sleep(0.25)
            end_marker = 99 if self.markers_are_numeric else "BLOCK_END"
            self._push_marker(end_marker)
            print(f"[markers] final={end_marker}")

        stop_event.set()
        print("[replay] Encerrado.")


def start_plot(script: Path, names: dict[str, str], model_prefix: str | None = None, new_console: bool = True) -> subprocess.Popen | None:
    if not script.exists():
        print(f"[plot] Script não encontrado: {script}")
        return None

    cmd = [
        sys.executable,
        str(script),
        "--decoder-name", names["decoder_name"],
        "--decoder-type", names["decoder_type"],
        "--marker-name", names["marker_name"],
        "--marker-type", names["marker_type"],
    ]
    meta = read_model_meta(model_prefix)
    xlim = meta.get("pca_train_xlim")
    ylim = meta.get("pca_train_ylim")
    if xlim and ylim:
        cmd += ["--pca-xlim", str(xlim[0]), str(xlim[1]), "--pca-ylim", str(ylim[0]), str(ylim[1])]
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" and new_console else 0
    print("[plot] Abrindo plot_decoder_realtime.py em processo separado.")
    return subprocess.Popen(cmd, creationflags=flags)


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_online(cfg: AppConfig, replay_pair: tuple[Path, Path], model_prefix: str, names: dict[str, str], args) -> None:
    stop_event = threading.Event()
    player = CSVReplayLSL(
        pair=replay_pair,
        names=names,
        replay_fs=args.replay_fs,
        block_sec=args.block_sec,
        speed=args.speed,
        append_block_end=not args.no_append_block_end,
        cfg=cfg,
    )

    plot_proc = None if args.no_plot else start_plot(args.plot_script, names, model_prefix=model_prefix, new_console=not args.same_console_plot)
    decoder_thread = threading.Thread(
        target=run_realtime_decoder,
        args=(cfg,),
        kwargs={"mode": "realtime", "model_prefix": model_prefix, "stop_event": stop_event},
        daemon=True,
    )

    try:
        print("\n[online] Iniciando decoder...")
        decoder_thread.start()
        time.sleep(max(float(args.decoder_warmup_s), 0.0))
        player.run(stop_event, start_delay_s=max(float(args.start_delay_s), 0.0))
        decoder_thread.join(timeout=5.0)
    finally:
        stop_event.set()
        stop_process(plot_proc)


def parse_args():
    parser = argparse.ArgumentParser(description="Valida todos os CSVs, escolhe um modelo e simula inferencia online com outro CSV via LSL.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--train-session-type", default=None)
    parser.add_argument("--online-session-type", default=None)

    parser.add_argument("--replay-fs", type=float, default=None, help="Taxa nominal do replay. Se omitido, estima pelo lsl_time_s do CSV.")
    parser.add_argument("--block-sec", type=float, default=0.10, help="Tamanho dos chunks enviados por LSL.")
    parser.add_argument("--speed", type=float, default=1.0, help="Velocidade temporal do replay. 1.0 = tempo real; 2.0 = 2x.")
    parser.add_argument("--x2", action="store_true", help="Atalho para --speed 2.0.")
    parser.add_argument("--start-delay-s", type=float, default=1.0)
    parser.add_argument("--decoder-warmup-s", type=float, default=1.5)
    parser.add_argument("--no-append-block-end", action="store_true")

    parser.add_argument("--eeg-name", default=None)
    parser.add_argument("--eeg-type", default=None)
    parser.add_argument("--marker-name", default=None)
    parser.add_argument("--marker-type", default=None)
    parser.add_argument("--decoder-name", default=None)
    parser.add_argument("--decoder-type", default=None)

    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-script", type=Path, default=Path("plot_decoder_realtime.py"))
    parser.add_argument("--same-console-plot", action="store_true")
    parser.add_argument("--skip-check", action="store_true", help="Nao roda check_data. Use apenas para debug rapido.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "x2", False):
        args.speed = 2.0
    base_dir = Path(__file__).resolve().parent

    if not args.config.is_absolute():
        args.config = base_dir / args.config
    if not args.plot_script.is_absolute():
        args.plot_script = base_dir / args.plot_script

    cfg, raw = load_cfg(args.config)
    p = raw.get("protocol", {}) or {}

    train_type = args.train_session_type or p.get("imagery_session_type", getattr(cfg.experiment, "session_type", None))
    online_type = args.online_session_type or p.get("online_session_type", "IM_online")
    cfg_train = set_session_type(cfg, train_type)
    cfg_online = set_session_type(cfg, online_type)
    names = lsl_names(cfg_online, raw, args)

    folder = args.folder
    if folder is None:
        folder = choose_folder_gui()
    if folder is None:
        folder = choose_folder_cli()
    folder = folder.resolve()

    pairs = find_pairs(folder, recursive=not args.no_recursive)
    if not pairs:
        raise FileNotFoundError(f"Nenhum par *_markers_*.csv / *_signal_*.csv encontrado em: {folder}")

    print("\n===== CONFIG =====")
    print(f"Config      : {args.config}")
    print(f"Pasta       : {folder}")
    print(f"Treino tipo : {cfg_train.experiment.session_type}")
    print(f"Online tipo : {cfg_online.experiment.session_type}")
    print(f"EEG stream  : {names['eeg_name']} / {names['eeg_type']}")
    print(f"Markers     : {names['marker_name']} / {names['marker_type']}")
    print(f"Pares       : {len(pairs)}")

    rows = process_all_pairs(cfg_train, pairs, folder, skip_check=args.skip_check)
    print_validation_table(rows)

    if not any(row.get("model_prefix") for row in rows):
        raise RuntimeError("Nenhum classificador foi criado com sucesso. Verifique os erros acima.")

    model_idx = choose_valid_index(rows, "Escolha o número do arquivo/modelo que será usado como CLASSIFICADOR", require_model=True, default=1)
    replay_default = 2 if len(rows) > 1 and model_idx == 0 else 1
    replay_idx = choose_valid_index(rows, "Escolha o número do arquivo que será usado como REPLAY/INFERÊNCIA", require_model=False, default=replay_default)

    model_row = rows[model_idx]
    replay_row = rows[replay_idx]

    print("\n===== SELEÇÃO FINAL =====")
    print(f"Modelo/classificador: [{model_row['idx']:02d}] CV={model_row['acc_mean']:.3f} | {model_row['signal_name']}")
    print(f"Replay/inferência   : [{replay_row['idx']:02d}] {replay_row['signal_name']}")
    print(f"Modelo prefix       : {model_row['model_prefix']}")
    print(f"Replay fs           : {args.replay_fs if args.replay_fs is not None else getattr(cfg_online, 'fs_hz', 'config')}")
    print(f"Replay speed        : {args.speed:.2f}x")
    print("\nNada foi transmitido ainda. O replay so começa depois do OK abaixo.")

    if not ask_yes_no("OK para iniciar o bloco online simulado?", default=True):
        print("Cancelado antes do online.")
        return

    run_online(
        cfg_online,
        replay_pair=(replay_row["marker_file"], replay_row["signal_file"]),
        model_prefix=str(model_row["model_prefix"]),
        names=names,
        args=args,
    )
    print("\nFinalizado.")


if __name__ == "__main__":
    main()
