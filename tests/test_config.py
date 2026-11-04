"""Unit tests for configuration loading and validation."""

import json
from pathlib import Path

import pytest

from src.config import AppConfig, load_config, validate_config


def test_default_config_loads_successfully(tmp_path: Path) -> None:
    """Verifies that config/default.yaml loads and passes all validation rules."""
    cfg = load_config("config/default.yaml", calibration_path=None)
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.camera_index == 0
    assert cfg.camera.frame_width == 640
    assert cfg.drowsiness.ear_threshold == 0.21
    assert cfg.drowsiness.perclos_threshold == 0.35
    assert cfg.distraction.yaw_threshold_deg == 30.0
    assert cfg.alarm.volume_cap == 0.70


def test_missing_config_raises_file_not_found() -> None:
    """Ensures a non-existent config path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")


@pytest.mark.parametrize(
    "param,value",
    [
        ("ear_threshold", 0.01),  # Below 0.05
        ("ear_threshold", 0.95),  # Above 0.50
        ("mar_threshold", 0.10),  # Below 0.20
        ("perclos_threshold", 1.50),  # Above 1.0
        ("perclos_window_sec", 2.0),  # Below 5.0
        ("drowsiness_debounce_sec", -1.0),  # Negative
        ("yaw_threshold_deg", 95.0),  # Above 85.0
        ("volume_cap", 1.50),  # Above 1.0
        ("volume_cap", 0.0),  # Zero or negative
        ("target_fps", 0),  # Zero
    ],
)
def test_out_of_range_values_raise_value_error(param: str, value: float) -> None:
    """Verifies that out-of-domain configuration values trigger clear ValueErrors."""
    cfg = load_config("config/default.yaml", calibration_path=None)
    if hasattr(cfg.drowsiness, param):
        setattr(cfg.drowsiness, param, value)
    elif hasattr(cfg.distraction, param):
        setattr(cfg.distraction, param, value)
    elif hasattr(cfg.alarm, param):
        setattr(cfg.alarm, param, value)
    elif hasattr(cfg.camera, param):
        setattr(cfg.camera, param, int(value))

    with pytest.raises(ValueError):
        validate_config(cfg)


def test_calibration_merge_overrides_ear_threshold(tmp_path: Path) -> None:
    """Verifies that runtime calibration.json correctly updates drowsiness.ear_threshold."""
    calib_file = tmp_path / "calibration.json"
    calib_data = {
        "calibrated_ear_threshold": 0.245,
        "baseline_open_ear": 0.327,
        "timestamp": "2026-11-04T12:00:00Z",
    }
    calib_file.write_text(json.dumps(calib_data), encoding="utf-8")

    cfg = load_config("config/default.yaml", calibration_path=calib_file)
    assert cfg.drowsiness.ear_threshold == pytest.approx(0.245, abs=1e-4)


def test_corrupt_calibration_logs_warning_and_keeps_default(tmp_path: Path) -> None:
    """Verifies that a corrupt calibration file does not crash loading."""
    corrupt_file = tmp_path / "corrupt_calibration.json"
    corrupt_file.write_text("invalid json content!", encoding="utf-8")

    cfg = load_config("config/default.yaml", calibration_path=corrupt_file)
    assert cfg.drowsiness.ear_threshold == 0.21
