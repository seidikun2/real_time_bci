#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, random, time
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet, local_clock

# ===================== CONFIG =====================

CSV_PATH    = Path("/home/seidi/Documentos/GitHub/real_time_bci/stims_sequence.csv")
STREAM_NAME = "GrazMI_Markers"
STREAM_TYPE = "Markers"
SOURCE_ID   = "GrazMI"

RANDOMIZE_TRIALS = True
START_DELAY_S    = 3.0
END_DELAY_S      = 0.5

CODE_MAP = {
    "BASELINE":      1,
    "ATTENTION":     2,
    "LEFT_MI_STIM":  3,
    "RIGHT_MI_STIM": 4,
    "ATTEMPT":       5,
    "REST":          6,
    "BLOCK_END":     99,
}

DUR = {
    "BASELINE":  3.0,
    "ATTENTION": 2.0,
    "STIM":      1.25,
    "ATTEMPT":   3.75,
    "REST":      2.0,
}

# ===================== FUNÇÕES =====================

def load_trials(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        trials = list(csv.DictReader(f))

    if not trials:
        raise ValueError(f"CSV vazio: {csv_path}")

    if "event_label" not in trials[0]:
        raise ValueError("O CSV precisa ter uma coluna chamada 'event_label'.")

    if RANDOMIZE_TRIALS:
        random.shuffle(trials)

    return trials


def wait_until(t_target: float):
    while time.perf_counter() < t_target:
        time.sleep(0.001)


def send_marker(outlet: StreamOutlet, label: str, t0: float):
    label = label.strip().upper()
    if label not in CODE_MAP:
        raise ValueError(f"Marcador inválido: {label}")

    code  = int(CODE_MAP[label])
    t_lsl = local_clock()
    t_exp = time.perf_counter() - t0

    outlet.push_sample([code], timestamp=t_lsl)
    print(f"{t_exp:8.3f}s | {label:14s} | code={code:3d} | t_lsl={t_lsl:.6f}")


def run_protocol():
    info   = StreamInfo(STREAM_NAME, STREAM_TYPE, 1, 0, "int32", SOURCE_ID)
    outlet = StreamOutlet(info)
    trials = load_trials(CSV_PATH)

    print(f"\nLSL stream criado: {STREAM_NAME}")
    print(f"CSV: {CSV_PATH}")
    print(f"N trials: {len(trials)}")
    print(f"Iniciando em {START_DELAY_S:.1f} segundos...\n")
    time.sleep(START_DELAY_S)

    t0     = time.perf_counter()
    next_t = t0

    send_marker(outlet, "BASELINE", t0)
    next_t += DUR["BASELINE"]
    wait_until(next_t)

    for i, trial in enumerate(trials, start=1):
        event_label = trial["event_label"].strip().upper()

        if event_label not in ("LEFT_MI_STIM", "RIGHT_MI_STIM"):
            raise ValueError(
                "event_label inválido no CSV. Use LEFT_MI_STIM ou RIGHT_MI_STIM. "
                f"Valor encontrado: {event_label}"
            )

        print(f"\nTrial {i}/{len(trials)}")

        send_marker(outlet, "ATTENTION", t0)
        next_t += DUR["ATTENTION"]
        wait_until(next_t)

        send_marker(outlet, event_label, t0)
        next_t += DUR["STIM"]
        wait_until(next_t)

        send_marker(outlet, "ATTEMPT", t0)
        next_t += DUR["ATTEMPT"]
        wait_until(next_t)

        send_marker(outlet, "REST", t0)
        next_t += DUR["REST"]
        wait_until(next_t)

    send_marker(outlet, "BLOCK_END", t0)
    time.sleep(END_DELAY_S)
    print("\nFim do bloco. Marcador BLOCK_END enviado.")


# ===================== MAIN =====================

if __name__ == "__main__":
    run_protocol()
