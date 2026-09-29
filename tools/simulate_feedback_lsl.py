#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Valida e, opcionalmente, publica o novo contrato Signal/BCI + GrazMI_Control.

Uso:
    python tools/simulate_feedback_lsl.py
    python tools/simulate_feedback_lsl.py --lsl --seconds 1.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_core.feedback_contract import (  # noqa: E402
    SIGNAL_CHANNELS, classifier_probabilities, signal_vector,
    movement_targets, legacy_state_id,
)

CONTROL_CHANNELS = (
    "left_leg", "right_leg", "rest", "left", "both", "right",
    "confidence", "density_gate", "state_id",
)


class MockClassifier:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes, dtype=object)
        self._p = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        return np.repeat(self._p[None, :], len(X), axis=0)


def semantic_state(probs, active):
    labels = [lab for lab, flag in active.items() if flag >= 0.5]
    return max(labels, key=lambda lab: probs.get(lab, 0.0)) if labels else "REST_STIM"


def control_vector(state, confidence, density_gate=1.0):
    left_leg, right_leg = movement_targets(state)
    sid = legacy_state_id(state)
    return [
        left_leg, right_leg,
        1.0 if sid == 0 else 0.0,
        1.0 if sid == 1 else 0.0,
        0.0,  # BOTH legado: nunca ativo no protocolo atual
        1.0 if sid == 3 else 0.0,
        float(confidence), float(density_gate), float(sid),
    ]


def make_case(name, classes, p, meta, rep):
    clf = MockClassifier(classes, p)
    probs, active = classifier_probabilities(clf, np.array([0.0, 0.0]), meta)
    sig = signal_vector(rep[0], rep[1], probs, active)
    state = semantic_state(probs, active)
    ctrl = control_vector(state, probs[state])
    return name, probs, active, sig, state, ctrl


def build_cases():
    return [
        make_case(
            "single_target_LEFT",
            classes=[0, 1], p=[0.18, 0.82],
            meta={"classes_map": {"REST_STIM": 0, "LEFT_MI_STIM": 1}},
            rep=(0.20, -0.15),
        ),
        make_case(
            "single_target_RIGHT",
            classes=["REST", "RIGHT_MI_STIM"], p=[0.20, 0.80],
            meta={}, rep=(-0.10, 0.30),
        ),
        make_case(
            "two_legs",
            classes=["LEFT", "REST_STIM", "RIGHT"], p=[0.15, 0.20, 0.65],
            meta={}, rep=(0.35, 0.05),
        ),
        make_case(
            "REST_dominante",
            classes=["LEFT", "REST", "RIGHT"], p=[0.08, 0.84, 0.08],
            meta={}, rep=(0.00, 0.00),
        ),
    ]


def print_cases(cases):
    print("Signal / BCI:")
    print("  " + ", ".join(SIGNAL_CHANNELS))
    for name, probs, active, sig, state, ctrl in cases:
        print(f"\n{name}")
        print("  Signal : [" + ", ".join(f"{v:.2f}" for v in sig) + "]")
        print(f"  state  : {state}")
        print("  Control: [" + ", ".join(f"{v:.2f}" for v in ctrl) + "]")


def validate(cases):
    by_name = {row[0]: row for row in cases}
    left = by_name["single_target_LEFT"][3]
    right = by_name["single_target_RIGHT"][3]
    two = by_name["two_legs"]
    rest = by_name["REST_dominante"]

    assert left[2] > 0 and left[3] > 0 and left[4] == 0
    assert left[5:8] == [1.0, 1.0, 0.0]
    assert right[2] == 0 and right[3] > 0 and right[4] > 0
    assert right[5:8] == [0.0, 1.0, 1.0]
    assert all(v > 0 for v in two[2].values())
    assert two[3][5:8] == [1.0, 1.0, 1.0]

    # Contrato crítico: REST dominante nunca gera movimento bilateral.
    rest_ctrl = rest[5]
    assert rest[4] == "REST_STIM"
    assert rest_ctrl[0] == 0.0 and rest_ctrl[1] == 0.0
    assert rest_ctrl[4] == 0.0
    print("\n[OK] single LEFT, single RIGHT e two_legs validados.")
    print("[OK] p_rest alto -> left_leg=0 e right_leg=0; BOTH legado permanece 0.")


def publish_lsl(cases, seconds):
    try:
        from pylsl import StreamInfo, StreamOutlet
    except Exception as exc:
        raise RuntimeError(f"pylsl não disponível: {exc}") from exc

    sig_info = StreamInfo("Signal", "BCI", len(SIGNAL_CHANNELS), 10.0, "float32", "feedback-contract-sim")
    sig_ch = sig_info.desc().append_child("channels")
    for label in SIGNAL_CHANNELS:
        sig_ch.append_child("channel").append_child_value("label", label)
    sig_out = StreamOutlet(sig_info)

    ctrl_info = StreamInfo("GrazMI_Control", "BCIControl", len(CONTROL_CHANNELS), 10.0, "float32", "feedback-control-sim")
    ctrl_ch = ctrl_info.desc().append_child("channels")
    for label in CONTROL_CHANNELS:
        ctrl_ch.append_child("channel").append_child_value("label", label)
    ctrl_out = StreamOutlet(ctrl_info)

    n = max(1, int(round(float(seconds) * 10.0)))
    print("\nPublicando exemplos LSL...")
    for name, _, _, sig, _, ctrl in cases:
        print(f"  {name}")
        for _ in range(n):
            sig_out.push_sample(sig)
            ctrl_out.push_sample(ctrl)
            time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsl", action="store_true", help="também publica os vetores em LSL")
    ap.add_argument("--seconds", type=float, default=1.0, help="duração de cada caso no modo LSL")
    args = ap.parse_args()

    cases = build_cases()
    print_cases(cases)
    validate(cases)
    if args.lsl:
        publish_lsl(cases, args.seconds)


if __name__ == "__main__":
    main()
