# main.py
import threading
import time
import numpy as np
from pathlib import Path

from config_models       import load_config, AppConfig
from online_inference    import run_realtime_decoder
from receive_data_log    import run_receive
from decoder_calibration import run_calibration as run_decoder_calibration
from check_data          import run_check_data

from gtec_signal_transmit import run_gtec_transmission as run_transmission


def start_gtec_tx(cfg: AppConfig, mode: str, stop_event: threading.Event, threads: list) -> None:
    print(">> MODO AQUISIÇÃO REAL: g.HIamp -> LSL.")
    print("   O gtec_signal_transmit.py vai publicar o EEG via LSL.")

    tx_thread = threading.Thread(
        target=run_transmission,
        args=(cfg, mode, stop_event),
        daemon=True,
    )

    threads.append(tx_thread)
    tx_thread.start()

    # Dá tempo para o stream LSL do g.HIamp aparecer antes do receiver procurar EEG.
    time.sleep(1.5)


def run_imagery_training(cfg: AppConfig) -> None:
    """
    Etapa 1 - treino de imagética.
    - Inicia transmissão do g.HIamp via LSL.
    - Inicia recepção para logar sinal + marcadores.
    """
    print("\n=== ETAPA 1: Treino de imagética ===")
    print(
        f"Sujeito: {cfg.experiment.subject_id}, "
        f"Sessão: {cfg.experiment.session_id}, "
        f"Tipo: {cfg.experiment.session_type}"
    )

    stop_event = threading.Event()
    threads = []

    start_gtec_tx(cfg, "train", stop_event, threads)

    rx_thread = threading.Thread(
        target=run_receive,
        args=(cfg, "train", stop_event),
        daemon=True,
    )

    threads.append(rx_thread)
    rx_thread.start()

    print("\n[INFO] Processos de treino iniciados.")
    print("       - TX g.HIamp ativo via LSL")
    print("       - RX ativo para registrar EEG + marcadores")
    print("       - PsychoPy deve estar rodando e enviando marcadores via LSL")
    print("       Quando o bloco de treino terminar, pressione ENTER para encerrar a etapa.")

    input("\n>>> Pressione ENTER ao final do protocolo de treino para encerrar TX/RX... ")

    stop_event.set()

    for th in threads:
        th.join(timeout=5.0)

    print("[INFO] Etapa de treino encerrada.\n")


def run_calibration(cfg: AppConfig) -> None:
    print("\n=== ETAPA 2: Calibração de modelo ===")

    res = run_decoder_calibration(cfg)
    acc_mean = float(np.mean(res["accs_cv"])) if res["accs_cv"] else float("nan")

    print(f"[main] Acurácia média CV ~ {acc_mean:.3f}")
    input("Revise as métricas numéricas. Pressione ENTER para continuar...")

    qc = input("Deseja visualizar os dados (check_data)? [s/N]: ").strip().lower()

    if qc in ("s", "sim", "y", "yes"):
        run_check_data(cfg)


def run_realtime(cfg: AppConfig) -> None:
    """
    Etapa 3 - tempo-real.
    - Inicia transmissão do g.HIamp via LSL.
    - Inicia RX para log.
    - Roda decoder no processo principal.
    """
    print("\n=== ETAPA 3: Tempo-real (TX g.HIamp + RX + INFERÊNCIA) ===")

    stop_event = threading.Event()
    threads = []

    start_gtec_tx(cfg, "realtime", stop_event, threads)

    rx_thread = threading.Thread(
        target=run_receive,
        args=(cfg, "realtime", stop_event),
        daemon=True,
    )

    threads.append(rx_thread)
    rx_thread.start()

    print("\n[INFO] Tempo-real iniciado.")
    print("       - TX g.HIamp ativo via LSL")
    print("       - RX ativo para log")
    print("       - Decoder ativo no processo principal")

    try:
        run_realtime_decoder(cfg, mode="realtime")

    except KeyboardInterrupt:
        print("\n[main] Tempo-real interrompido pelo usuário.")

    finally:
        stop_event.set()

        for th in threads:
            th.join(timeout=5.0)

        print("[main] Tempo-real encerrado.")


def main() -> None:
    CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
    cfg: AppConfig = load_config(CONFIG_PATH)
    mode = getattr(cfg.runtime, "start_mode", "full_loop").lower()

    if mode == "realtime_only":
        print("\n[main] start_mode='realtime_only' -> indo direto para tempo-real.\n")
        run_realtime(cfg)
        return

    if mode == "train_once":
        print("\n[main] start_mode='train_once' -> Treino + Calibração uma vez.\n")
        run_imagery_training(cfg)
        run_calibration(cfg)

        ok = input("Ir para tempo-real agora? [s/N]: ").strip().lower()

        if ok in ("s", "sim", "y", "yes"):
            run_realtime(cfg)
        else:
            print("[main] Encerrando sem iniciar tempo-real.")

        return

    print("\n[main] start_mode='full_loop' -> Treino + Calibração em loop até OK, depois tempo-real.\n")

    while True:
        run_imagery_training(cfg)
        run_calibration(cfg)

        ok = input("As métricas estão OK? Prosseguir para tempo-real? [s/N]: ").strip().lower()

        if ok in ("s", "sim", "y", "yes"):
            break

        print("Repetindo Treino + Calibração...\n")

    run_realtime(cfg)


if __name__ == "__main__":
    main()