# main.py
import threading
import numpy as np

from config_models       import load_config, AppConfig
from online_inference    import run_realtime_decoder
from realtime_signal_transmit import run_transmission
from receive_data_log    import run_receive
from decoder_calibration import run_calibration as run_decoder_calibration
from check_data          import run_check_data


def run_imagery_training(cfg: AppConfig) -> None:
    """
    Etapa 1 - treino de imagética.
    - Sempre roda a RECEPÇÃO (run_receive) para logar sinal + marcadores.
    - Se simulate_signal=True, também roda a TRANSMISSÃO SIMULADA (run_transmission).
    Ambos rodam em paralelo e são controlados por um stop_event.
    """
    print("\n=== ETAPA 1: Treino de imagética ===")
    print(
        f"Sujeito: {cfg.experiment.subject_id}, "
        f"Sessão: {cfg.experiment.session_id}, "
        f"Tipo: {cfg.experiment.session_type}"
    )

    stop_event = threading.Event()
    threads    = []

    # TX simulada (opcional)
    if cfg.runtime.simulate_signal:
        print(">> MODO TESTE: sinal EEG SIMULADO via LSL (transmissão + recepção).")
        tx_thread = threading.Thread(
            target=run_transmission,
            args=(cfg, "train", stop_event),
            daemon=True,
        )
        threads.append(tx_thread)
        tx_thread.start()
    else:
        print(">> MODO AQUISIÇÃO REAL: NÃO será iniciada transmissão simulada.")
        print("   Certifique-se de que o software de aquisição está transmitindo o EEG por LSL.")

    # RX sempre
    rx_thread = threading.Thread(
        target=run_receive,
        args=(cfg, "train", stop_event),
        daemon=True,
    )
    threads.append(rx_thread)
    rx_thread.start()

    print("\n[INFO] Processos de treino iniciados.")
    print("       - RX sempre ativo")
    if cfg.runtime.simulate_signal:
        print("       - TX simulado ativo")
    else:
        print("       - TX REAL (hardware) esperado via LSL (software de aquisição)")
    print("       Quando o bloco de treino terminar, pressione ENTER para encerrar a etapa.")

    input("\n>>> Pressione ENTER ao final do protocolo de treino para encerrar TX/RX... ")

    # Sinaliza para as threads saírem do loop
    stop_event.set()

    # Dá um tempo para elas saírem de forma limpa
    for th in threads:
        th.join(timeout=2.0)

    print("[INFO] Etapa de treino encerrada.\n")


def run_calibration(cfg: AppConfig) -> None:
    print("\n=== ETAPA 2: Calibração de modelo ===")

    res       = run_decoder_calibration(cfg)
    acc_mean  = float(np.mean(res["accs_cv"])) if res["accs_cv"] else float("nan")
    print(f"[main] Acurácia média CV ~ {acc_mean:.3f}")
    input("Revise as métricas numéricas. Pressione ENTER para continuar...")

    qc = input("Deseja visualizar os dados (check_data)? [s/N]: ").strip().lower()
    if qc in ("s", "sim", "y", "yes"):
        run_check_data(cfg)


def run_realtime(cfg: AppConfig) -> None:
    """
    Etapa 3 - Tempo-real.
    RX sempre ativo para log; TX pode ser simulado ou real;
    decoder roda no thread principal (permite Ctrl+C).
    """
    print("\n=== ETAPA 3: Tempo-real (TX + RX + INFERÊNCIA) ===")

    # RX sempre ativo (para log)
    rx_thread = threading.Thread(
        target=run_receive,
        args=(cfg,),
        kwargs={"mode": "realtime"},
        daemon=True,
    )
    rx_thread.start()

    # Se simulado, liga TX
    if cfg.runtime.simulate_signal:
        tx_thread = threading.Thread(
            target=run_transmission,
            args=(cfg,),
            kwargs={"mode": "realtime"},
            daemon=True,
        )
        tx_thread.start()
        print(">> MODO TESTE: sinal EEG SIMULADO via LSL (TX + RX + decoder).")
    else:
        print(">> MODO AQUISIÇÃO REAL: RX + decoder (sem TX simulado).")

    # Decoder roda no thread principal
    run_realtime_decoder(cfg, mode="realtime")


def main() -> None:
    cfg: AppConfig = load_config("config.yaml")
    mode           = getattr(cfg.runtime, "start_mode", "full_loop").lower()

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

    # default: full_loop
    print("\n[main] start_mode='full_loop' -> Treino+Calib em loop até OK, depois tempo-real.\n")
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
