from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import yaml


@dataclass
class ExperimentConfig:
    subject_id: str
    session_id: int
    log_root: str
    exp_name: str
    session_type: str


@dataclass
class LSLConfig:
    marker_name: str
    marker_type: str
    signal_name: str
    signal_type: str


# ---- SINAL SIMULADO ----
@dataclass
class SimSignalConfig:
    fs: float
    channels: int
    chunk: int
    noise_std: float


@dataclass
class SimBurstConfig:
    freq: float
    amp: float
    dur: float
    taper_frac: float


@dataclass
class SimProfilesConfig:
    left_mi: Tuple[float, float]
    right_mi: Tuple[float, float]


@dataclass
class CodesConfig:
    left_mi: int
    right_mi: int
    attempt: int
    code_map: Dict[int, str]


@dataclass
class RuntimeConfig:
    simulate_signal: bool
    # controla o fluxo principal:
    #  - "full_loop"     -> Treino+Calib repetem até OK, depois realtime
    #  - "train_once"    -> 1x Treino+Calib, pergunta se vai para realtime
    #  - "realtime_only" -> pula Treino/Calib e vai direto para realtime
    start_mode: str = "full_loop"


# ---- CALIBRAÇÃO / MODELO ----
@dataclass
class ModelConfig:
    fs_hz: float
    bp_order: int
    bp_band: List[float]   # [low, high]

    epoch_s: float
    trial_offset_s: float

    pca_dim: int
    svc_c: float
    rng_seed: int
    cv_splits: int

    select_by: str
    index_base: int
    select_channels: List[int]


# ---- CHECK_DATA ----
@dataclass
class CheckDataConfig:
    select_channels: List[int]
    hp_cutoff_hz: float
    hp_order: int
    classes: List[str]
    tmin: float
    tmax: float
    baseline_s: float
    save_png: bool


# ---- DECODER (tempo real) ----
@dataclass
class DecoderConfig:
    epoch_s: float
    step_s: float
    band_hz: List[float]
    filter_order: int
    outlet_name: str
    lsl_rate_hz: float
    left_label: Any | None
    right_label: Any | None


@dataclass
class AppConfig:
    experiment:   ExperimentConfig
    lsl:          LSLConfig
    sim_signal:   SimSignalConfig
    sim_burst:    SimBurstConfig
    sim_profiles: SimProfilesConfig
    codes:        CodesConfig
    runtime:      RuntimeConfig
    model:        ModelConfig
    check_data:   CheckDataConfig
    decoder:      DecoderConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # ---- blocos básicos ----
    exp = ExperimentConfig(**raw["experiment"])
    lsl = LSLConfig(**raw["lsl"])

    # ---- sinal simulado ----
    sim_sig = SimSignalConfig(**raw["sim_signal"])
    sim_bur = SimBurstConfig(**raw["sim_burst"])
    sim_prof = SimProfilesConfig(
        left_mi=tuple(raw["sim_profiles"]["left_mi"]),
        right_mi=tuple(raw["sim_profiles"]["right_mi"]),
    )

    # ---- códigos ----
    codes_raw = raw["codes"]
    codes = CodesConfig(
        left_mi=codes_raw["left_mi"],
        right_mi=codes_raw["right_mi"],
        attempt=codes_raw["attempt"],
        code_map={int(k): v for k, v in codes_raw["map"].items()},
    )

    # ---- runtime ----
    runtime_raw = raw.get("runtime", {})
    run_cfg = RuntimeConfig(
        simulate_signal=runtime_raw.get("simulate_signal", True),
        start_mode=runtime_raw.get("start_mode", "full_loop"),
    )

    # ---- model / calibração ----
    model_cfg = ModelConfig(**raw["model"])

    # ---- check_data ----
    check_cfg = CheckDataConfig(**raw["check_data"])

    # ---- decoder tempo real ----
    decoder_cfg = DecoderConfig(**raw["decoder"])

    return AppConfig(
        experiment=exp,
        lsl=lsl,
        sim_signal=sim_sig,
        sim_burst=sim_bur,
        sim_profiles=sim_prof,
        codes=codes,
        runtime=run_cfg,
        model=model_cfg,
        check_data=check_cfg,
        decoder=decoder_cfg,
    )
