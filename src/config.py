"""Configuration management and validation for the drowsiness warning system.

Loads default YAML configuration, validates parameter domains and data types,
and merges optional runtime calibration overrides.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera acquisition configuration."""

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    disconnect_timeout_sec: float = 3.0
    max_consecutive_dropped_frames: int = 5


@dataclass
class LandmarksConfig:
    """Facial landmark detector configuration."""

    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class DrowsinessConfig:
    """Drowsiness estimation thresholds and durations."""

    ear_threshold: float = 0.21
    mar_threshold: float = 0.60
    yawn_min_duration_sec: float = 1.0
    perclos_window_sec: float = 60.0
    perclos_threshold: float = 0.35
    blink_max_duration_sec: float = 0.40
    microsleep_min_duration_sec: float = 1.0
    drowsiness_debounce_sec: float = 1.5
    recovery_debounce_sec: float = 0.5


@dataclass
class DistractionConfig:
    """Distraction detection thresholds and boundaries."""

    yaw_threshold_deg: float = 30.0
    pitch_min_deg: float = -20.0
    pitch_max_deg: float = 20.0
    distraction_debounce_sec: float = 2.0
    no_face_debounce_sec: float = 3.0


@dataclass
class ModelConfig:
    """Deep learning model and fusion configuration."""

    cnn_enabled: bool = True
    model_path: str = "models/mobilenetv2_eye_state.pt"
    model_metadata_path: str = "models/mobilenetv2_eye_state.json"
    input_size: list[int] = field(default_factory=lambda: [128, 128])
    cnn_inference_interval: int = 2
    ear_weight: float = 0.50
    cnn_weight: float = 0.50
    fusion_threshold: float = 0.50


@dataclass
class AlarmConfig:
    """Auditory alarm engine configuration."""

    audio_backend: str = "auto"
    alarm_cooldown_sec: float = 5.0
    volume_cap: float = 0.70
    max_continuous_alarm_sec: float = 10.0
    warning_frequency_hz: int = 440
    drowsy_frequency_hz: int = 800
    distracted_frequency_hz: int = 600
    fault_frequency_hz: int = 1000


@dataclass
class CalibrationConfig:
    """Baseline calibration configuration."""

    calibration_duration_sec: float = 10.0
    ear_multiplier: float = 0.75
    calibration_output_path: str = "config/calibration.json"
    min_valid_baseline_ear: float = 0.20
    max_valid_baseline_ear: float = 0.45


@dataclass
class LoggingConfig:
    """Logging and privacy configuration."""

    log_dir: str = "logs"
    save_frames: bool = False
    debug_frames_dir: str = "saved_frames"
    show_display: bool = True


@dataclass
class AppConfig:
    """Root configuration holding all sub-configurations."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarks: LandmarksConfig = field(default_factory=LandmarksConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    distraction: DistractionConfig = field(default_factory=DistractionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def validate_config(cfg: AppConfig) -> None:
    """Validates parameter ranges, types, and logical consistency.

    Args:
        cfg: An AppConfig instance to validate.

    Raises:
        ValueError: If any parameter value falls outside its acceptable domain.
        TypeError: If any parameter is of an unexpected type.
    """
    # Camera validation
    if cfg.camera.camera_index < 0:
        raise ValueError(f"camera_index must be >= 0, got {cfg.camera.camera_index}")
    if cfg.camera.frame_width < 160 or cfg.camera.frame_height < 120:
        raise ValueError(
            f"frame dimensions too small ({cfg.camera.frame_width}x{cfg.camera.frame_height})"
        )
    if cfg.camera.target_fps < 1:
        raise ValueError(f"target_fps must be >= 1, got {cfg.camera.target_fps}")

    # Landmarks validation
    if not (0.0 < cfg.landmarks.min_detection_confidence <= 1.0):
        raise ValueError("min_detection_confidence must be in (0.0, 1.0]")
    if not (0.0 < cfg.landmarks.min_tracking_confidence <= 1.0):
        raise ValueError("min_tracking_confidence must be in (0.0, 1.0]")

    # Drowsiness validation
    if not (0.05 <= cfg.drowsiness.ear_threshold <= 0.50):
        raise ValueError(
            f"ear_threshold must be in [0.05, 0.50], got {cfg.drowsiness.ear_threshold}"
        )
    if not (0.20 <= cfg.drowsiness.mar_threshold <= 1.50):
        raise ValueError(
            f"mar_threshold must be in [0.20, 1.50], got {cfg.drowsiness.mar_threshold}"
        )
    if not (0.05 <= cfg.drowsiness.perclos_threshold <= 1.0):
        raise ValueError(
            f"perclos_threshold must be in [0.05, 1.0], got {cfg.drowsiness.perclos_threshold}"
        )
    if cfg.drowsiness.perclos_window_sec < 5.0:
        raise ValueError(
            f"perclos_window_sec must be >= 5.0, got {cfg.drowsiness.perclos_window_sec}"
        )
    if cfg.drowsiness.blink_max_duration_sec >= cfg.drowsiness.microsleep_min_duration_sec:
        raise ValueError(
            f"blink_max_duration_sec ({cfg.drowsiness.blink_max_duration_sec}) must be "
            f"less than microsleep_min_duration_sec ({cfg.drowsiness.microsleep_min_duration_sec})"
        )
    if cfg.drowsiness.drowsiness_debounce_sec <= 0.0:
        raise ValueError("drowsiness_debounce_sec must be positive")
    if cfg.drowsiness.recovery_debounce_sec <= 0.0:
        raise ValueError("recovery_debounce_sec must be positive")

    # Distraction validation
    if not (5.0 <= cfg.distraction.yaw_threshold_deg <= 85.0):
        raise ValueError(
            f"yaw_threshold_deg must be in [5.0, 85.0], got {cfg.distraction.yaw_threshold_deg}"
        )
    if cfg.distraction.pitch_min_deg >= cfg.distraction.pitch_max_deg:
        raise ValueError(
            f"pitch_min_deg ({cfg.distraction.pitch_min_deg}) must be < "
            f"pitch_max_deg ({cfg.distraction.pitch_max_deg})"
        )
    if cfg.distraction.distraction_debounce_sec <= 0.0:
        raise ValueError("distraction_debounce_sec must be positive")
    if cfg.distraction.no_face_debounce_sec <= 0.0:
        raise ValueError("no_face_debounce_sec must be positive")

    # Model validation
    if not (0.0 <= cfg.model.ear_weight <= 1.0):
        raise ValueError(f"ear_weight must be in [0.0, 1.0], got {cfg.model.ear_weight}")
    if not (0.0 <= cfg.model.cnn_weight <= 1.0):
        raise ValueError(f"cnn_weight must be in [0.0, 1.0], got {cfg.model.cnn_weight}")
    if cfg.model.cnn_inference_interval < 1:
        raise ValueError("cnn_inference_interval must be >= 1")

    # Alarm validation
    if cfg.alarm.audio_backend not in ("auto", "pygame", "winsound", "mock"):
        raise ValueError(f"Invalid audio_backend: {cfg.alarm.audio_backend}")
    if not (0.0 < cfg.alarm.volume_cap <= 1.0):
        raise ValueError(f"volume_cap must be in (0.0, 1.0], got {cfg.alarm.volume_cap}")
    if cfg.alarm.alarm_cooldown_sec < 0.0:
        raise ValueError("alarm_cooldown_sec must be non-negative")
    if cfg.alarm.max_continuous_alarm_sec <= 0.0:
        raise ValueError("max_continuous_alarm_sec must be positive")


def _dict_to_dataclass(cls: Any, data: dict[str, Any]) -> Any:
    """Helper to instantiate a dataclass from a dictionary ignoring extra keys."""
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)


def load_config(
    config_path: str | Path = "config/default.yaml",
    calibration_path: str | Path | None = "config/calibration.json",
) -> AppConfig:
    """Loads configuration from YAML and merges runtime calibration if present.

    Args:
        config_path: Path to the primary YAML configuration file.
        calibration_path: Optional path to dynamic calibration overrides JSON.

    Returns:
        A validated AppConfig instance.

    Raises:
        FileNotFoundError: If the primary YAML config file does not exist.
        ValueError: If configuration values fail schema or domain validation.
    """
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    app_cfg = AppConfig(
        camera=_dict_to_dataclass(CameraConfig, raw_dict.get("camera", {})),
        landmarks=_dict_to_dataclass(LandmarksConfig, raw_dict.get("landmarks", {})),
        drowsiness=_dict_to_dataclass(DrowsinessConfig, raw_dict.get("drowsiness", {})),
        distraction=_dict_to_dataclass(DistractionConfig, raw_dict.get("distraction", {})),
        model=_dict_to_dataclass(ModelConfig, raw_dict.get("model", {})),
        alarm=_dict_to_dataclass(AlarmConfig, raw_dict.get("alarm", {})),
        calibration=_dict_to_dataclass(CalibrationConfig, raw_dict.get("calibration", {})),
        logging=_dict_to_dataclass(LoggingConfig, raw_dict.get("logging", {})),
    )

    # Merge calibration file if present
    if calibration_path:
        calib_file = Path(calibration_path)
        if calib_file.is_file():
            try:
                with open(calib_file, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                if "calibrated_ear_threshold" in calib_data:
                    calibrated_ear = float(calib_data["calibrated_ear_threshold"])
                    logger.info("Applying calibrated EAR threshold: %.4f", calibrated_ear)
                    app_cfg.drowsiness.ear_threshold = calibrated_ear
            except (json.JSONDecodeError, ValueError, OSError) as e:
                logger.warning("Failed to load calibration file %s: %s", calib_file, e)

    validate_config(app_cfg)
    return app_cfg
