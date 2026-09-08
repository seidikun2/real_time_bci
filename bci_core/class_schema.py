from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Any

# Ordem canônica usada apenas para manter IDs/saídas estáveis entre sessões.
# A presença efetiva de uma classe é determinada pelos marcadores observados
# (e, antes da aquisição, pode ser prevista pela stims_sequence.csv).
MOTOR_CLASS_ORDER = ("LEFT_MI_STIM", "BOTH_MI_STIM", "RIGHT_MI_STIM")

MOTOR_CLASS_SPECS: dict[str, dict[str, Any]] = {
    "LEFT_MI_STIM": {
        "display": "LEFT",
        "control": "left_leg",
        "default_code": 3,
        "decoder_channel": "left",
    },
    "BOTH_MI_STIM": {
        "display": "BOTH",
        "control": "both_legs",
        "default_code": 7,
        "decoder_channel": "both",
    },
    "RIGHT_MI_STIM": {
        "display": "RIGHT",
        "control": "right_leg",
        "default_code": 4,
        "decoder_channel": "right",
    },
}


def normalize_label(value: Any) -> str:
    return str(value).strip().upper()


def active_motor_labels(labels: Iterable[Any], require_at_least: int = 0) -> list[str]:
    present = {normalize_label(v) for v in labels}
    out = [label for label in MOTOR_CLASS_ORDER if label in present]
    if require_at_least and len(out) < require_at_least:
        raise ValueError(
            f"Foram encontradas apenas {len(out)} classe(s) motora(s): {out}. "
            f"São necessárias pelo menos {require_at_least}."
        )
    return out


def label_map_for(labels: Iterable[Any]) -> dict[str, int]:
    active = active_motor_labels(labels, require_at_least=2)
    return {label: idx for idx, label in enumerate(active)}


def display_name(label: str) -> str:
    key = normalize_label(label)
    return str(MOTOR_CLASS_SPECS.get(key, {}).get("display", key))


def control_name(label: str) -> str:
    key = normalize_label(label)
    return str(MOTOR_CLASS_SPECS.get(key, {}).get("control", key.lower()))


def decoder_channel(label: str) -> str:
    key = normalize_label(label)
    return str(MOTOR_CLASS_SPECS.get(key, {}).get("decoder_channel", ""))


def marker_code(label: str, code_map: Mapping[int, str] | None = None) -> int | None:
    key = normalize_label(label)
    if code_map:
        for code, mapped in code_map.items():
            if normalize_label(mapped) == key:
                try:
                    return int(code)
                except Exception:
                    pass
    spec = MOTOR_CLASS_SPECS.get(key)
    return int(spec["default_code"]) if spec else None


def class_details(label_map: Mapping[str, int], code_map: Mapping[int, str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in MOTOR_CLASS_ORDER:
        if label not in label_map:
            continue
        rows.append(
            {
                "model_class": int(label_map[label]),
                "event_label": label,
                "display_label": display_name(label),
                "control": control_name(label),
                "decoder_channel": decoder_channel(label),
                "marker_code": marker_code(label, code_map),
            }
        )
    return rows


def classes_from_sequence(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    labels: list[str] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = normalize_label(row.get("event_label", ""))
            if label and label not in labels:
                labels.append(label)
    return active_motor_labels(labels)
