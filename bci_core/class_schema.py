from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Any

# LEFT/RIGHT são as classes motoras principais do protocolo atual.
# BOTH é mantido apenas por compatibilidade com modelos/Unity antigos.
MOTOR_CLASS_ORDER = ("LEFT_MI_STIM", "BOTH_MI_STIM", "RIGHT_MI_STIM")
REST_STIM_LABEL = "REST_STIM"
STIM_CLASS_ORDER = ("LEFT_MI_STIM", "RIGHT_MI_STIM", REST_STIM_LABEL, "BOTH_MI_STIM")

CLASS_SPECS: dict[str, dict[str, Any]] = {
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
    "REST_STIM": {
        "display": "REST",
        "control": "rest",
        "default_code": 8,
        "decoder_channel": "rest",
    },
}
MOTOR_CLASS_SPECS = CLASS_SPECS


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


def active_stim_labels(labels: Iterable[Any]) -> list[str]:
    present = {normalize_label(v) for v in labels}
    return [label for label in STIM_CLASS_ORDER if label in present]


def normalize_target_label(value: Any) -> str:
    key = normalize_label(value)
    aliases = {
        "LEFT": "LEFT_MI_STIM",
        "L": "LEFT_MI_STIM",
        "LEFT_LEG": "LEFT_MI_STIM",
        "LEFT_MI_STIM": "LEFT_MI_STIM",
        "RIGHT": "RIGHT_MI_STIM",
        "R": "RIGHT_MI_STIM",
        "RIGHT_LEG": "RIGHT_MI_STIM",
        "RIGHT_MI_STIM": "RIGHT_MI_STIM",
        "BOTH": "BOTH_MI_STIM",
        "BOTH_LEGS": "BOTH_MI_STIM",
        "BOTH_MI_STIM": "BOTH_MI_STIM",
    }
    if key not in aliases:
        raise ValueError(f"online_target inválido: {value!r}. Use left ou right.")
    return aliases[key]


def label_map_for(labels: Iterable[Any]) -> dict[str, int]:
    """Modo legado: usa apenas classes motoras presentes."""
    active = active_motor_labels(labels, require_at_least=2)
    return {label: idx for idx, label in enumerate(active)}


def training_label_map(
    labels: Iterable[Any],
    training_mode: str = "motor_multiclass",
    online_target: str = "left",
) -> dict[str, int]:
    """Classes efetivamente usadas para ajustar o modelo.

    target_vs_rest:
        REST_STIM -> 0
        perna-alvo -> 1
        a outra perna permanece no bloco, mas fica fora do ajuste.

    left_right_vs_rest:
        REST_STIM -> 0
        LEFT -> 1
        RIGHT -> 2
        Este é o modo de "duas classes motoras", mantendo REST explícito.

    motor_multiclass:
        comportamento legado: classes motoras presentes, sem REST_STIM.
    """
    mode = normalize_label(training_mode).lower()
    present = {normalize_label(v) for v in labels}

    if mode in {"target_vs_rest", "leg_vs_rest", "single_leg"}:
        target = normalize_target_label(online_target)
        if target == "BOTH_MI_STIM":
            raise ValueError("O protocolo atual single-target aceita apenas left ou right.")
        missing = [lab for lab in (REST_STIM_LABEL, target) if lab not in present]
        if missing:
            raise ValueError(
                "Treino target_vs_rest requer REST_STIM e a perna-alvo. "
                f"Ausentes: {missing}."
            )
        return {REST_STIM_LABEL: 0, target: 1}

    if mode in {"left_right_vs_rest", "two_legs_vs_rest", "dual_leg", "two_legs"}:
        required = (REST_STIM_LABEL, "LEFT_MI_STIM", "RIGHT_MI_STIM")
        missing = [lab for lab in required if lab not in present]
        if missing:
            raise ValueError(
                "Treino left_right_vs_rest requer LEFT_MI_STIM, RIGHT_MI_STIM e REST_STIM. "
                f"Ausentes: {missing}."
            )
        return {REST_STIM_LABEL: 0, "LEFT_MI_STIM": 1, "RIGHT_MI_STIM": 2}

    if mode in {"motor_multiclass", "multiclass", "motor"}:
        return label_map_for(present)

    raise ValueError(
        f"training_mode inválido: {training_mode!r}. "
        "Use target_vs_rest, left_right_vs_rest ou motor_multiclass."
    )


def display_name(label: str) -> str:
    key = normalize_label(label)
    return str(CLASS_SPECS.get(key, {}).get("display", key))


def control_name(label: str) -> str:
    key = normalize_label(label)
    return str(CLASS_SPECS.get(key, {}).get("control", key.lower()))


def decoder_channel(label: str) -> str:
    key = normalize_label(label)
    return str(CLASS_SPECS.get(key, {}).get("decoder_channel", ""))


def marker_code(label: str, code_map: Mapping[int, str] | None = None) -> int | None:
    key = normalize_label(label)
    if code_map:
        for code, mapped in code_map.items():
            if normalize_label(mapped) == key:
                try:
                    return int(code)
                except Exception:
                    pass
    spec = CLASS_SPECS.get(key)
    return int(spec["default_code"]) if spec else None


def class_details(label_map: Mapping[str, int], code_map: Mapping[int, str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, model_class in sorted(label_map.items(), key=lambda kv: int(kv[1])):
        key = normalize_label(label)
        rows.append(
            {
                "model_class": int(model_class),
                "event_label": key,
                "display_label": display_name(key),
                "control": control_name(key),
                "decoder_channel": decoder_channel(key),
                "marker_code": marker_code(key, code_map),
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
    return active_stim_labels(labels)
