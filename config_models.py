from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import copy
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
    decoder_name: str = "Signal"
    decoder_type: str = "BCI"


@dataclass
class RuntimeConfig:
    simulate_signal: bool
    start_mode: str = "full_loop"


@dataclass
class ProtocolConfig:
    start_phase: str = "execution"
    block_end_code: int = 99
    block_end_label: str = "BLOCK_END"
    motor_session_type: str = "EM_treino"
    imagery_session_type: str = "IM_treino"
    online_session_type: str = "IM_online"
    min_cv_accuracy: Optional[float] = None
    auto_check_data: bool = True
    online_model_prefix: str = "ask"
    model_session_types: List[str] = field(default_factory=lambda: ["EM_treino", "IM_treino"])
    realtime_plot: bool = True
    realtime_plot_script: str = "plot_decoder_realtime.py"
    realtime_plot_new_console: bool = True


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
    block_end: int
    code_map: Dict[int, str]


@dataclass
class ModelConfig:
    fs_hz: float
    bp_order: int
    bp_band: List[float]
    epoch_s: float
    trial_offset_s: float
    pca_dim: int
    svc_c: float
    rng_seed: int
    cv_splits: int
    select_by: str
    index_base: int
    select_channels: List[Any]


@dataclass
class CheckDataConfig:
    select_channels: List[Any]
    hp_cutoff_hz: float
    hp_order: int
    classes: List[str]
    tmin: float
    tmax: float
    baseline_s: float
    save_png: bool


@dataclass
class DecoderConfig:
    epoch_s: float
    step_s: float
    band_hz: List[float]
    filter_order: int
    outlet_name: str
    outlet_type: str
    lsl_rate_hz: float
    left_label: Any | None
    right_label: Any | None


@dataclass
class AppConfig:
    experiment: ExperimentConfig
    lsl: LSLConfig
    runtime: RuntimeConfig
    protocol: ProtocolConfig
    sim_signal: SimSignalConfig
    sim_burst: SimBurstConfig
    sim_profiles: SimProfilesConfig
    codes: CodesConfig
    model: ModelConfig
    check_data: CheckDataConfig
    decoder: DecoderConfig

    # parâmetros globais de sessão/pipeline
    fs_hz: float
    n_channels: int
    trial_duration_s: float
    trial_offset_s: float
    window_s: float
    step_s: float
    filter_mode: str
    bp_order: int
    bp_band_hz: List[float]
    select_by: str
    index_base: int
    select_channels: List[Any]
    feature_space: str
    classifier: str
    pca_dim: int
    svc_c: float
    cv_splits: int
    rng_seed: int

    _raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)


def _get(raw: Dict[str, Any], key: str, default: Any = None) -> Any:
    return raw.get(key, default)


def _nested(raw: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = raw.get(section, {}) or {}
    return value.get(key, default) if isinstance(value, dict) else default


def _first(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _as_list(value, default=None):
    if value is None:
        return [] if default is None else default
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _protocol(raw: Dict[str, Any], trial_duration_s: float) -> ProtocolConfig:
    p = raw.get("protocol", {}) or {}
    return ProtocolConfig(
        start_phase=p.get("start_phase", "execution"),
        block_end_code=int(p.get("block_end_code", _nested(raw, "codes", "block_end", 99))),
        block_end_label=p.get("block_end_label", "BLOCK_END"),
        motor_session_type=p.get("motor_session_type", "EM_treino"),
        imagery_session_type=p.get("imagery_session_type", "IM_treino"),
        online_session_type=p.get("online_session_type", "IM_online"),
        min_cv_accuracy=p.get("min_cv_accuracy", None),
        auto_check_data=bool(p.get("auto_check_data", True)),
        online_model_prefix=str(p.get("online_model_prefix", "ask")),
        model_session_types=_as_list(p.get("model_session_types", ["EM_treino", "IM_treino"])),
        realtime_plot=bool(p.get("realtime_plot", True)),
        realtime_plot_script=str(p.get("realtime_plot_script", "plot_decoder_realtime.py")),
        realtime_plot_new_console=bool(p.get("realtime_plot_new_console", True)),
    )


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Globais: preferem a raiz do YAML, mas aceitam config antigo agrupado.
    fs_hz = float(_first(_get(raw, "fs_hz"), _nested(raw, "model", "fs_hz"), _nested(raw, "sim_signal", "fs"), default=256.0))
    n_channels = int(_first(_get(raw, "n_channels"), _nested(raw, "sim_signal", "channels"), default=16))
    trial_duration_s = float(_first(_get(raw, "trial_duration_s"), _nested(raw, "protocol", "trial_duration_s"), _nested(raw, "check_data", "tmax"), default=3.75))
    trial_offset_s = float(_first(_get(raw, "trial_offset_s"), _nested(raw, "model", "trial_offset_s"), default=0.0))
    window_s = float(_first(_get(raw, "window_s"), _nested(raw, "model", "epoch_s"), _nested(raw, "decoder", "epoch_s"), default=1.0))
    step_s = float(_first(_get(raw, "step_s"), _nested(raw, "decoder", "step_s"), _nested(raw, "model", "step_s"), default=0.05))
    filter_mode = str(_first(_get(raw, "filter_mode"), _nested(raw, "model", "filter_mode"), default="causal"))
    bp_order = int(_first(_get(raw, "bp_order"), _nested(raw, "model", "bp_order"), _nested(raw, "decoder", "filter_order"), default=3))
    bp_band_hz = list(_first(_get(raw, "bp_band_hz"), _nested(raw, "model", "bp_band"), _nested(raw, "decoder", "band_hz"), default=[5.0, 40.0]))
    select_by = str(_first(_get(raw, "select_by"), _nested(raw, "model", "select_by"), default="index"))
    index_base = int(_first(_get(raw, "index_base"), _nested(raw, "model", "index_base"), default=1))
    select_channels = _as_list(_first(_get(raw, "select_channels"), _nested(raw, "model", "select_channels"), default=[]))
    feature_space = str(_get(raw, "feature_space", "riemann_pca"))
    classifier = str(_get(raw, "classifier", "svm_linear"))
    pca_dim = int(_first(_get(raw, "pca_dim"), _nested(raw, "model", "pca_dim"), default=2))
    svc_c = float(_first(_get(raw, "svc_c"), _nested(raw, "model", "svc_c"), default=1.0))
    cv_splits = int(_first(_get(raw, "cv_splits"), _nested(raw, "model", "cv_splits"), default=5))
    rng_seed = int(_first(_get(raw, "rng_seed"), _nested(raw, "model", "rng_seed"), default=42))

    exp_raw = raw.get("experiment", {}) or {}
    lsl_raw = raw.get("lsl", {}) or {}
    runtime_raw = raw.get("runtime", {}) or {}
    codes_raw = raw.get("codes", {}) or {}
    code_map_raw = codes_raw.get("map", codes_raw.get("code_map", {}))
    code_map = {int(k): str(v) for k, v in code_map_raw.items()}

    exp = ExperimentConfig(
        subject_id=str(exp_raw.get("subject_id", "TT000")),
        session_id=int(exp_raw.get("session_id", 1)),
        log_root=str(exp_raw.get("log_root", "Dados")),
        exp_name=str(exp_raw.get("exp_name", "TEST")),
        session_type=str(exp_raw.get("session_type", "IM_treino")),
    )
    lsl = LSLConfig(
        marker_name=str(lsl_raw.get("marker_name", "GrazMI_Markers")),
        marker_type=str(lsl_raw.get("marker_type", "Markers")),
        signal_name=str(lsl_raw.get("signal_name", "gHIamp_EEG")),
        signal_type=str(lsl_raw.get("signal_type", "EEG")),
        decoder_name=str(lsl_raw.get("decoder_name", raw.get("decoder_outlet_name", "Signal"))),
        decoder_type=str(lsl_raw.get("decoder_type", raw.get("decoder_outlet_type", "BCI"))),
    )
    runtime = RuntimeConfig(
        simulate_signal=bool(runtime_raw.get("simulate_signal", True)),
        start_mode=str(runtime_raw.get("start_mode", "full_loop")),
    )
    protocol = _protocol(raw, trial_duration_s)

    sim_signal = SimSignalConfig(
        fs=fs_hz,
        channels=n_channels,
        chunk=int(_get(raw, "sim_chunk", _nested(raw, "sim_signal", "chunk", 10))),
        noise_std=float(_get(raw, "sim_noise_std", _nested(raw, "sim_signal", "noise_std", 0.8))),
    )
    sim_burst = SimBurstConfig(
        freq=float(_get(raw, "sim_burst_freq_hz", _nested(raw, "sim_burst", "freq", 10.0))),
        amp=float(_get(raw, "sim_burst_amp", _nested(raw, "sim_burst", "amp", 1.5))),
        dur=trial_duration_s,
        taper_frac=float(_get(raw, "sim_taper_frac", _nested(raw, "sim_burst", "taper_frac", 0.2))),
    )
    sim_profiles = SimProfilesConfig(
        left_mi=tuple(_get(raw, "sim_profile_left_mi", _nested(raw, "sim_profiles", "left_mi", [1.0, 0.3]))),
        right_mi=tuple(_get(raw, "sim_profile_right_mi", _nested(raw, "sim_profiles", "right_mi", [0.3, 1.0]))),
    )
    codes = CodesConfig(
        left_mi=int(codes_raw.get("left_mi", 3)),
        right_mi=int(codes_raw.get("right_mi", 4)),
        attempt=int(codes_raw.get("attempt", 5)),
        block_end=int(codes_raw.get("block_end", 99)),
        code_map=code_map,
    )
    model = ModelConfig(
        fs_hz=fs_hz,
        bp_order=bp_order,
        bp_band=bp_band_hz,
        epoch_s=window_s,
        trial_offset_s=trial_offset_s,
        pca_dim=pca_dim,
        svc_c=svc_c,
        rng_seed=rng_seed,
        cv_splits=cv_splits,
        select_by=select_by,
        index_base=index_base,
        select_channels=select_channels,
    )
    check_data = CheckDataConfig(
        select_channels=select_channels,
        hp_cutoff_hz=float(_nested(raw, "check_data", "hp_cutoff_hz", 0.5)),
        hp_order=bp_order,
        classes=list(_nested(raw, "check_data", "classes", ["LEFT_MI_STIM", "RIGHT_MI_STIM"])),
        tmin=float(_get(raw, "check_tmin_s", _nested(raw, "check_data", "tmin", -0.5))),
        tmax=trial_duration_s,
        baseline_s=float(_nested(raw, "check_data", "baseline_s", 0.0)),
        save_png=bool(_get(raw, "check_save_png", _nested(raw, "check_data", "save_png", True))),
    )
    decoder = DecoderConfig(
        epoch_s=window_s,
        step_s=step_s,
        band_hz=bp_band_hz,
        filter_order=bp_order,
        outlet_name=str(_get(raw, "decoder_outlet_name", _nested(raw, "decoder", "outlet_name", lsl.decoder_name))),
        outlet_type=str(_get(raw, "decoder_outlet_type", _nested(raw, "decoder", "outlet_type", lsl.decoder_type))),
        lsl_rate_hz=float(_get(raw, "decoder_lsl_rate_hz", _nested(raw, "decoder", "lsl_rate_hz", 64.0))),
        left_label=_get(raw, "left_label", _nested(raw, "decoder", "left_label", None)),
        right_label=_get(raw, "right_label", _nested(raw, "decoder", "right_label", None)),
    )

    # Enriquecido para módulos antigos que ainda consultam raw["protocol"] / raw["model"].
    enriched = copy.deepcopy(raw)
    enriched.setdefault("protocol", {})
    enriched["protocol"]["trial_duration_s"] = trial_duration_s
    enriched.setdefault("model", {})
    enriched["model"].update({
        "fs_hz": fs_hz,
        "bp_order": bp_order,
        "bp_band": bp_band_hz,
        "epoch_s": window_s,
        "step_s": step_s,
        "trial_offset_s": trial_offset_s,
        "pca_dim": pca_dim,
        "svc_c": svc_c,
        "rng_seed": rng_seed,
        "cv_splits": cv_splits,
        "select_by": select_by,
        "index_base": index_base,
        "select_channels": select_channels,
    })
    enriched.setdefault("decoder", {})
    enriched["decoder"].update({
        "epoch_s": window_s,
        "step_s": step_s,
        "band_hz": bp_band_hz,
        "filter_order": bp_order,
        "outlet_name": decoder.outlet_name,
        "lsl_rate_hz": decoder.lsl_rate_hz,
        "left_label": decoder.left_label,
        "right_label": decoder.right_label,
    })

    return AppConfig(
        experiment=exp,
        lsl=lsl,
        runtime=runtime,
        protocol=protocol,
        sim_signal=sim_signal,
        sim_burst=sim_burst,
        sim_profiles=sim_profiles,
        codes=codes,
        model=model,
        check_data=check_data,
        decoder=decoder,
        fs_hz=fs_hz,
        n_channels=n_channels,
        trial_duration_s=trial_duration_s,
        trial_offset_s=trial_offset_s,
        window_s=window_s,
        step_s=step_s,
        filter_mode=filter_mode,
        bp_order=bp_order,
        bp_band_hz=bp_band_hz,
        select_by=select_by,
        index_base=index_base,
        select_channels=select_channels,
        feature_space=feature_space,
        classifier=classifier,
        pca_dim=pca_dim,
        svc_c=svc_c,
        cv_splits=cv_splits,
        rng_seed=rng_seed,
        _raw_config=enriched,
    )
