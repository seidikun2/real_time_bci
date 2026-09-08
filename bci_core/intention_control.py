# -*- coding: utf-8 -*-
"""Controle de intenção para o feedback motor no Unity.

Camada entre o decoder probabilístico e a animação:
1) recebe [rep1, rep2, P(left), P(both), P(right)] do decoder;
2) usa as regiões HDR salvas em online/pca_map.json como gate espacial;
3) usa probabilidades do SVM para resolver sobreposições entre classes;
4) aplica persistência temporal + histerese espacial;
5) publica estado discreto e posições contínuas das pernas.

Estado discreto:
    0 REST
    1 LEFT
    2 BOTH
    3 RIGHT

A posição de cada perna fica em [0, 1]. Enquanto uma classe está ativa a
perna correspondente sobe continuamente; quando a classe deixa de estar ativa,
a perna cai a partir da posição em que se encontra.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

from .config_models import AppConfig
from .class_schema import MOTOR_CLASS_ORDER, decoder_channel, display_name


STATE_ID = {
    "REST": 0,
    "LEFT_MI_STIM": 1,
    "BOTH_MI_STIM": 2,
    "RIGHT_MI_STIM": 3,
}

PROB_CHANNEL_INDEX = {
    "LEFT_MI_STIM": 2,
    "BOTH_MI_STIM": 3,
    "RIGHT_MI_STIM": 4,
}


def log(msg: str) -> None:
    print(f"[control] {msg}")


def _raw(cfg: AppConfig) -> dict[str, Any]:
    return getattr(cfg, "_raw_config", {}) or {}


def control_config(cfg: AppConfig) -> dict[str, Any]:
    block = _raw(cfg).get("control", {}) or {}
    return block if isinstance(block, dict) else {}


def _session_root(cfg: AppConfig) -> Path:
    return Path(cfg.experiment.log_root) / cfg.experiment.subject_id / f"S{cfg.experiment.session_id}"


def _online_dir(cfg: AppConfig) -> Path:
    return _session_root(cfg) / "online"


def _map_path(cfg: AppConfig) -> Path:
    configured = control_config(cfg).get("pca_map_json")
    if configured:
        p = Path(str(configured))
        if not p.is_absolute():
            p = _online_dir(cfg) / p
        return p
    return _online_dir(cfg) / "pca_map.json"


def _load_map(cfg: AppConfig) -> dict[str, Any]:
    path = _map_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Mapa PCA não encontrado para o controlador: {path}. "
            "O main deve publicar pca_map.json antes do online."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def _active_labels(map_data: dict[str, Any]) -> list[str]:
    regions = map_data.get("density_regions", [])
    labels = []
    if isinstance(regions, list):
        for row in regions:
            if not isinstance(row, dict):
                continue
            lab = str(row.get("event_label", "")).strip().upper()
            if lab in MOTOR_CLASS_ORDER and lab not in labels:
                labels.append(lab)
    if labels:
        return [lab for lab in MOTOR_CLASS_ORDER if lab in labels]

    # Compatibilidade com mapa v2 sem regiões numéricas.
    events = [str(v).strip().upper() for v in map_data.get("event_labels", [])]
    return [lab for lab in MOTOR_CLASS_ORDER if lab in events]


def _region_row(map_data: dict[str, Any], label: str) -> Optional[dict[str, Any]]:
    for row in map_data.get("density_regions", []) or []:
        if isinstance(row, dict) and str(row.get("event_label", "")).strip().upper() == label:
            return row
    return None


def _closest_region(row: Optional[dict[str, Any]], mass: float) -> Optional[dict[str, Any]]:
    if not row:
        return None
    regions = [r for r in row.get("regions", []) or [] if isinstance(r, dict) and r.get("mass") is not None]
    if not regions:
        return None
    return min(regions, key=lambda r: abs(float(r.get("mass", 0.0)) - float(mass)))


def _point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray casting. polygon = [[x0,y0], [x1,y1], ...]."""
    try:
        pts = [(float(p[0]), float(p[1])) for p in polygon if len(p) >= 2]
    except Exception:
        return False
    if len(pts) < 3:
        return False

    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        crosses = ((yi > y) != (yj > y))
        if crosses:
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-15 else 1e-15) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _inside_region(rep1: float, rep2: float, region: Optional[dict[str, Any]]) -> Optional[bool]:
    """True/False se existe geometria; None quando o mapa não contém a região."""
    if not region:
        return None

    polygons = region.get("polygons", []) or []
    if polygons:
        return any(_point_in_polygon(rep1, rep2, poly) for poly in polygons)

    intervals = region.get("intervals", []) or []
    if intervals:
        for interval in intervals:
            if len(interval) >= 2 and float(interval[0]) <= rep1 <= float(interval[1]):
                return True
        return False

    return None


def _probabilities(sample) -> dict[str, float]:
    arr = list(sample)
    out = {}
    for label, idx in PROB_CHANNEL_INDEX.items():
        out[label] = float(arr[idx]) if idx < len(arr) else 0.0
    return out


def _probability_margin_ok(label: str, probs: dict[str, float], active_labels: list[str], min_margin: float) -> bool:
    p = float(probs.get(label, 0.0))
    others = [float(probs.get(other, 0.0)) for other in active_labels if other != label]
    second = max(others) if others else 0.0
    return (p - second) >= float(min_margin)


def _approach_binary(value: float, target: float, dt_s: float, rise_s: float, fall_s: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    target = 1.0 if target >= 0.5 else 0.0
    if target > value:
        value += max(0.0, dt_s) / max(float(rise_s), 1e-6)
    elif target < value:
        value -= max(0.0, dt_s) / max(float(fall_s), 1e-6)
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class DecisionSnapshot:
    rep1: float = 0.0
    rep2: float = 0.0
    probs: dict[str, float] | None = None
    entry_inside: dict[str, bool | None] | None = None
    hold_inside: dict[str, bool | None] | None = None
    active_label: str | None = None
    confidence: float = 0.0
    density_gate: float = 0.0
    decoder_lsl_time: float = 0.0


class IntentionStateMachine:
    def __init__(self, map_data: dict[str, Any], cfg: dict[str, Any]):
        self.map_data = map_data
        self.cfg = cfg
        self.active_labels = _active_labels(map_data)
        self.entry_mass = float(cfg.get("entry_density_mass", 0.50))
        self.hold_mass = float(cfg.get("hold_density_mass", 0.80))
        self.enter_persist_s = float(cfg.get("enter_persist_s", 0.25))
        self.exit_persist_s = float(cfg.get("exit_persist_s", 0.20))
        self.min_probability = float(cfg.get("min_probability", 0.40))
        self.min_probability_margin = float(cfg.get("min_probability_margin", 0.05))
        self.allow_probability_fallback = bool(cfg.get("allow_probability_fallback", True))

        self.active_label: str | None = None
        self.pending_label: str | None = None
        self.pending_since: float | None = None
        self.exit_since: float | None = None
        self.snapshot = DecisionSnapshot(probs={lab: 0.0 for lab in MOTOR_CLASS_ORDER})

    def _inside(self, label: str, rep1: float, rep2: float, mass: float) -> bool | None:
        row = _region_row(self.map_data, label)
        region = _closest_region(row, mass)
        return _inside_region(rep1, rep2, region)

    def _entry_candidates(self, entry: dict[str, bool | None], probs: dict[str, float]) -> list[str]:
        candidates = []
        for label in self.active_labels:
            inside = entry.get(label)
            spatial_ok = inside is True
            if inside is None and self.allow_probability_fallback:
                spatial_ok = True
            if not spatial_ok:
                continue
            if float(probs.get(label, 0.0)) < self.min_probability:
                continue
            if not _probability_margin_ok(label, probs, self.active_labels, self.min_probability_margin):
                continue
            candidates.append(label)
        return sorted(candidates, key=lambda lab: float(probs.get(lab, 0.0)), reverse=True)

    def process(self, sample, decoder_lsl_time: float, now_mono: float) -> DecisionSnapshot:
        if len(sample) < 5:
            raise ValueError("Decoder deve publicar [rep1, rep2, left, both, right].")

        rep1, rep2 = float(sample[0]), float(sample[1])
        probs = _probabilities(sample)
        entry = {lab: self._inside(lab, rep1, rep2, self.entry_mass) for lab in self.active_labels}
        hold = {lab: self._inside(lab, rep1, rep2, self.hold_mass) for lab in self.active_labels}

        # 1) Estado já ativo: só o libera após permanecer fora da região de hold.
        if self.active_label is not None:
            current = self.active_label
            inside_hold = hold.get(current)
            if inside_hold is None and self.allow_probability_fallback:
                inside_hold = float(probs.get(current, 0.0)) >= self.min_probability

            if inside_hold:
                self.exit_since = None
            else:
                if self.exit_since is None:
                    self.exit_since = now_mono
                if now_mono - self.exit_since >= self.exit_persist_s:
                    self.active_label = None
                    self.exit_since = None
                    self.pending_label = None
                    self.pending_since = None

        # 2) REST: procura uma classe cujo núcleo foi atingido de forma persistente.
        if self.active_label is None:
            candidates = self._entry_candidates(entry, probs)
            candidate = candidates[0] if candidates else None
            if candidate is None:
                self.pending_label = None
                self.pending_since = None
            elif candidate != self.pending_label:
                self.pending_label = candidate
                self.pending_since = now_mono
            elif self.pending_since is not None and now_mono - self.pending_since >= self.enter_persist_s:
                self.active_label = candidate
                self.pending_label = None
                self.pending_since = None
                self.exit_since = None

        active = self.active_label
        confidence = float(probs.get(active, 0.0)) if active else (max(probs.values()) if probs else 0.0)
        density_gate = 0.0
        if active:
            inside = hold.get(active)
            density_gate = 1.0 if inside is True else 0.0
        elif self.pending_label:
            inside = entry.get(self.pending_label)
            density_gate = 1.0 if inside is True else 0.0

        self.snapshot = DecisionSnapshot(
            rep1=rep1,
            rep2=rep2,
            probs=probs,
            entry_inside=entry,
            hold_inside=hold,
            active_label=active,
            confidence=confidence,
            density_gate=density_gate,
            decoder_lsl_time=float(decoder_lsl_time),
        )
        return self.snapshot

    def force_rest(self) -> None:
        self.active_label = None
        self.pending_label = None
        self.pending_since = None
        self.exit_since = None
        self.snapshot.active_label = None
        self.snapshot.density_gate = 0.0


def _resolve_decoder(cfg: AppConfig, stop_event: threading.Event) -> StreamInlet | None:
    name = getattr(cfg.decoder, "outlet_name", "Signal")
    stype = getattr(cfg.decoder, "outlet_type", "BCI")
    log(f"Aguardando decoder LSL: name={name!r}, type={stype!r}")
    while not stop_event.is_set():
        streams = resolve_byprop("name", name, timeout=1.0) if name else []
        if not streams and stype:
            streams = resolve_byprop("type", stype, timeout=1.0)
        if streams:
            si = streams[0]
            log(f"Conectado ao decoder: {si.name()} | ch={si.channel_count()}")
            return StreamInlet(si, recover=True)
    return None


def _make_control_outlet(cfg: AppConfig, srate: float, feedback_mode: str) -> StreamOutlet:
    ccfg = control_config(cfg)
    name = str(ccfg.get("outlet_name", "GrazMI_Control"))
    stype = str(ccfg.get("outlet_type", "BCIControl"))
    labels = [
        "left_leg", "right_leg",
        "rest", "left", "both", "right",
        "confidence", "density_gate", "state_id",
    ]
    info = StreamInfo(name, stype, len(labels), float(srate), "float32")
    root = info.desc()
    channels = root.append_child("channels")
    for label in labels:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "a.u.")
        ch.append_child_value("type", "BCIControl")
    root.append_child_value("state_codes", "0=REST;1=LEFT;2=BOTH;3=RIGHT")
    root.append_child_value("feedback_mode", str(feedback_mode))
    return StreamOutlet(info)


def _open_logger(cfg: AppConfig, feedback_mode: str):
    folder = _online_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{cfg.experiment.subject_id}_{cfg.experiment.exp_name}_S{cfg.experiment.session_id}_{cfg.experiment.session_type}_control_{stamp}.csv"
    path = folder / stem
    f = path.open("w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow([
        "iso_time", "lsl_time_s", "decoder_lsl_time_s", "feedback_mode",
        "rep1", "rep2", "p_left", "p_both", "p_right",
        "entry_left", "entry_both", "entry_right",
        "hold_left", "hold_both", "hold_right",
        "state", "state_id", "confidence", "density_gate",
        "left_leg", "right_leg",
    ])
    log(f"Log de controle: {path}")
    return f, w


def _bool_num(value: bool | None) -> int:
    return -1 if value is None else int(bool(value))


def run_intention_controller(
    cfg: AppConfig,
    mode: str = "online",
    stop_event: Optional[threading.Event] = None,
    feedback_mode: str = "pca",
) -> None:
    if stop_event is None:
        stop_event = threading.Event()

    ccfg = control_config(cfg)
    if not bool(ccfg.get("enabled", True)):
        log("Controle desabilitado em config.control.enabled=false.")
        return

    map_data = _load_map(cfg)
    sm = IntentionStateMachine(map_data, ccfg)
    if not sm.active_labels:
        raise RuntimeError("pca_map.json não informa classes motoras ativas.")

    has_numeric_regions = bool(map_data.get("density_regions"))
    log(f"Mapa: {map_data.get('_path')}")
    log(f"Classes: {[display_name(l) for l in sm.active_labels]}")
    log(
        f"Gate: entra HDR={sm.entry_mass:.2f}, mantém HDR={sm.hold_mass:.2f} | "
        f"persistência ON={sm.enter_persist_s:.2f}s OFF={sm.exit_persist_s:.2f}s | "
        f"Pmin={sm.min_probability:.2f} margem={sm.min_probability_margin:.2f}"
    )
    if not has_numeric_regions:
        if sm.allow_probability_fallback:
            log("AVISO: mapa antigo sem regiões numéricas; usando fallback probabilístico.")
        else:
            raise RuntimeError("Mapa sem regiões numéricas e allow_probability_fallback=false.")

    inlet = _resolve_decoder(cfg, stop_event)
    if inlet is None:
        return

    output_hz = max(1.0, float(ccfg.get("output_rate_hz", 60.0)))
    rise_s = max(0.05, float(ccfg.get("movement_rise_s", 1.20)))
    fall_s = max(0.05, float(ccfg.get("movement_fall_s", 0.80)))
    stale_s = max(0.05, float(ccfg.get("decoder_stale_s", 0.60)))

    outlet = _make_control_outlet(cfg, output_hz, feedback_mode)
    fcsv, wcsv = _open_logger(cfg, feedback_mode)

    left_leg = 0.0
    right_leg = 0.0
    last_decoder_recv = time.monotonic()
    last_update = time.monotonic()
    next_publish = last_update
    period = 1.0 / output_hz
    last_announced_state = "REST"

    try:
        while not stop_event.is_set():
            samples, timestamps = inlet.pull_chunk(timeout=min(0.02, period), max_samples=64)
            if timestamps:
                now_mono = time.monotonic()
                for sample, t_dec in zip(samples, timestamps):
                    sm.process(sample, float(t_dec), now_mono)
                    last_decoder_recv = now_mono

            now_mono = time.monotonic()
            if now_mono - last_decoder_recv > stale_s:
                sm.force_rest()

            if now_mono < next_publish:
                time.sleep(min(0.002, max(0.0, next_publish - now_mono)))
                continue

            dt_s = max(0.0, now_mono - last_update)
            last_update = now_mono
            active = sm.active_label
            target_left = 1.0 if active in {"LEFT_MI_STIM", "BOTH_MI_STIM"} else 0.0
            target_right = 1.0 if active in {"RIGHT_MI_STIM", "BOTH_MI_STIM"} else 0.0
            left_leg = _approach_binary(left_leg, target_left, dt_s, rise_s, fall_s)
            right_leg = _approach_binary(right_leg, target_right, dt_s, rise_s, fall_s)

            state = "REST" if active is None else display_name(active)
            sid = STATE_ID.get(active or "REST", 0)
            onehot = {
                "REST": 1.0 if sid == 0 else 0.0,
                "LEFT": 1.0 if sid == 1 else 0.0,
                "BOTH": 1.0 if sid == 2 else 0.0,
                "RIGHT": 1.0 if sid == 3 else 0.0,
            }
            snap = sm.snapshot
            vec = [
                left_leg, right_leg,
                onehot["REST"], onehot["LEFT"], onehot["BOTH"], onehot["RIGHT"],
                float(snap.confidence), float(snap.density_gate), float(sid),
            ]
            t_lsl = local_clock()
            outlet.push_sample(vec, timestamp=t_lsl)

            probs = snap.probs or {}
            entry = snap.entry_inside or {}
            hold = snap.hold_inside or {}
            wcsv.writerow([
                dt.datetime.now().isoformat(timespec="microseconds"),
                f"{t_lsl:.9f}", f"{snap.decoder_lsl_time:.9f}", str(feedback_mode),
                f"{snap.rep1:.6f}", f"{snap.rep2:.6f}",
                f"{probs.get('LEFT_MI_STIM', 0.0):.6f}",
                f"{probs.get('BOTH_MI_STIM', 0.0):.6f}",
                f"{probs.get('RIGHT_MI_STIM', 0.0):.6f}",
                _bool_num(entry.get("LEFT_MI_STIM")),
                _bool_num(entry.get("BOTH_MI_STIM")),
                _bool_num(entry.get("RIGHT_MI_STIM")),
                _bool_num(hold.get("LEFT_MI_STIM")),
                _bool_num(hold.get("BOTH_MI_STIM")),
                _bool_num(hold.get("RIGHT_MI_STIM")),
                state, sid, f"{snap.confidence:.6f}", f"{snap.density_gate:.1f}",
                f"{left_leg:.6f}", f"{right_leg:.6f}",
            ])

            if state != last_announced_state:
                log(f"Estado: {last_announced_state} -> {state} | P={snap.confidence:.3f} | rep=({snap.rep1:.2f},{snap.rep2:.2f})")
                last_announced_state = state

            next_publish += period
            if next_publish < now_mono - period:
                next_publish = now_mono + period

    except KeyboardInterrupt:
        log("Ctrl+C recebido.")
    finally:
        try:
            fcsv.close()
        except Exception:
            pass
        log("Controlador encerrado.")
