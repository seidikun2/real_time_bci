from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Contrato fixo do stream Signal / BCI.
SIGNAL_CHANNELS = (
    "rep1", "rep2",
    "p_left", "p_rest", "p_right",
    "active_left", "active_rest", "active_right",
)

FEEDBACK_CLASS_ORDER = ("LEFT_MI_STIM", "REST_STIM", "RIGHT_MI_STIM")
PROB_CHANNEL_INDEX = {
    "LEFT_MI_STIM": 2,
    "REST_STIM": 3,
    "RIGHT_MI_STIM": 4,
}
ACTIVE_CHANNEL_INDEX = {
    "LEFT_MI_STIM": 5,
    "REST_STIM": 6,
    "RIGHT_MI_STIM": 7,
}

_ALIAS_TO_EVENT = {
    "LEFT": "LEFT_MI_STIM",
    "LEFT_MI": "LEFT_MI_STIM",
    "LEFT_MI_STIM": "LEFT_MI_STIM",
    "LEFT_LEG": "LEFT_MI_STIM",
    "L": "LEFT_MI_STIM",
    "RIGHT": "RIGHT_MI_STIM",
    "RIGHT_MI": "RIGHT_MI_STIM",
    "RIGHT_MI_STIM": "RIGHT_MI_STIM",
    "RIGHT_LEG": "RIGHT_MI_STIM",
    "R": "RIGHT_MI_STIM",
    "REST": "REST_STIM",
    "REST_STIM": "REST_STIM",
    "RESTSTIM": "REST_STIM",
    "NO_GO": "REST_STIM",
    "NOGO": "REST_STIM",
}


def _norm(value: Any) -> str:
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def canonical_feedback_label(value: Any) -> str | None:
    """Normaliza apenas as três classes operacionais atuais."""
    return _ALIAS_TO_EVENT.get(_norm(value))


def _same_model_class(a: Any, b: Any) -> bool:
    try:
        if bool(a == b):
            return True
    except Exception:
        pass
    sa, sb = str(a).strip(), str(b).strip()
    if sa == sb:
        return True
    try:
        return float(sa) == float(sb)
    except Exception:
        return False


def semantic_label_for_model_class(model_class: Any, model_meta: Mapping[str, Any] | None) -> str | None:
    """Resolve uma entrada de clf.classes_ para LEFT/REST/RIGHT.

    Prioridade:
    1) o próprio label em clf.classes_ (LEFT, REST_STIM, RIGHT_MI_STIM etc.);
    2) metadata ``classes`` do modelo;
    3) inversão de ``classes_map``.
    """
    direct = canonical_feedback_label(model_class)
    if direct is not None:
        return direct

    meta = dict(model_meta or {})
    for row in meta.get("classes", []) or []:
        if not isinstance(row, Mapping):
            continue
        if _same_model_class(model_class, row.get("model_class")):
            resolved = canonical_feedback_label(row.get("event_label", row.get("display_label", "")))
            if resolved is not None:
                return resolved

    class_map = meta.get("classes_map", {}) or {}
    if isinstance(class_map, Mapping):
        for event_label, mapped_class in class_map.items():
            if _same_model_class(model_class, mapped_class):
                resolved = canonical_feedback_label(event_label)
                if resolved is not None:
                    return resolved
    return None


def classifier_probabilities(clf, feat: np.ndarray, model_meta: Mapping[str, Any] | None = None):
    """Retorna probabilidades semânticas e flags de classes realmente treinadas.

    As probabilidades vêm diretamente de ``predict_proba``. Classes ausentes no
    modelo ficam em zero e não entram em qualquer renormalização.
    """
    if not hasattr(clf, "predict_proba"):
        raise RuntimeError("O classificador selecionado não possui predict_proba().")

    classes = list(getattr(clf, "classes_", []))
    if not classes:
        raise RuntimeError("O classificador não informa clf.classes_.")

    x = np.asarray(feat, dtype=float).reshape(1, -1)
    values = np.asarray(clf.predict_proba(x)[0], dtype=float)
    if len(values) != len(classes):
        raise RuntimeError("predict_proba e clf.classes_ possuem tamanhos diferentes.")

    probs = {label: 0.0 for label in FEEDBACK_CLASS_ORDER}
    active = {label: 0.0 for label in FEEDBACK_CLASS_ORDER}
    unresolved = []

    for raw_class, probability in zip(classes, values):
        label = semantic_label_for_model_class(raw_class, model_meta)
        if label is None:
            unresolved.append(raw_class)
            continue
        probs[label] += float(probability)
        active[label] = 1.0

    if unresolved:
        raise RuntimeError(
            "Classe(s) do modelo sem semântica LEFT/REST/RIGHT: "
            + ", ".join(repr(v) for v in unresolved)
        )
    if not any(active.values()):
        raise RuntimeError("Nenhuma classe LEFT/REST/RIGHT pôde ser resolvida no modelo.")

    return probs, active


def signal_vector(rep1: float, rep2: float, probs: Mapping[str, float], active: Mapping[str, float]) -> list[float]:
    return [
        float(rep1), float(rep2),
        float(probs.get("LEFT_MI_STIM", 0.0)),
        float(probs.get("REST_STIM", 0.0)),
        float(probs.get("RIGHT_MI_STIM", 0.0)),
        float(active.get("LEFT_MI_STIM", 0.0)),
        float(active.get("REST_STIM", 0.0)),
        float(active.get("RIGHT_MI_STIM", 0.0)),
    ]


def probabilities_from_signal(sample) -> dict[str, float]:
    arr = list(sample)
    if len(arr) < len(SIGNAL_CHANNELS):
        raise ValueError(f"Signal/BCI deve ter {len(SIGNAL_CHANNELS)} canais: {', '.join(SIGNAL_CHANNELS)}")
    return {label: float(arr[idx]) for label, idx in PROB_CHANNEL_INDEX.items()}


def active_flags_from_signal(sample) -> dict[str, float]:
    arr = list(sample)
    if len(arr) < len(SIGNAL_CHANNELS):
        raise ValueError(f"Signal/BCI deve ter {len(SIGNAL_CHANNELS)} canais: {', '.join(SIGNAL_CHANNELS)}")
    return {label: float(arr[idx]) for label, idx in ACTIVE_CHANNEL_INDEX.items()}


def movement_targets(label: str | None) -> tuple[float, float]:
    """Converte estado semântico em alvo das pernas; REST nunca é bilateral."""
    canonical = canonical_feedback_label(label) if label is not None else None
    if canonical == "LEFT_MI_STIM":
        return 1.0, 0.0
    if canonical == "RIGHT_MI_STIM":
        return 0.0, 1.0
    return 0.0, 0.0


def legacy_state_id(label: str | None) -> int:
    """Mantém IDs antigos do GrazMI_Control: 2 fica reservado ao BOTH legado."""
    canonical = canonical_feedback_label(label) if label is not None else None
    if canonical == "LEFT_MI_STIM":
        return 1
    if canonical == "RIGHT_MI_STIM":
        return 3
    return 0
