"""Unit tests for baseline open-eye EAR calibration engine."""

import json
from pathlib import Path

import pytest

from src.calibration import EarCalibrator


def test_calibration_normal_sequence(tmp_path: Path) -> None:
    """Verifies that a normal 10-second calibration calculates 0.75 * median open EAR."""
    calib_json = tmp_path / "calibration.json"
    calibrator = EarCalibrator(
        duration_sec=10.0,
        ear_multiplier=0.75,
        output_path=str(calib_json),
    )

    t = 100.0
    # Feed 10.5 seconds of data (start at 100.0, end at 110.5):
    # 20 samples of blinks (EAR = 0.10)
    # 85 samples of open eyes (EAR = 0.32)
    for _ in range(20):
        calibrator.add_sample(ear=0.10, timestamp=t)
        t += 0.1

    for _ in range(85):
        calibrator.add_sample(ear=0.32, timestamp=t)
        t += 0.1

    assert calibrator.is_complete is True
    baseline, thresh = calibrator.finish_calibration()

    # The 20 blink samples (lowest 20%) should be discarded.
    # The remaining 80 samples are all 0.32.
    # Median is 0.32. Calibrated threshold = 0.32 * 0.75 = 0.24.
    assert baseline == pytest.approx(0.32, abs=1e-4)
    assert thresh == pytest.approx(0.24, abs=1e-4)

    # Check JSON output
    assert calib_json.is_file()
    data = json.loads(calib_json.read_text(encoding="utf-8"))
    assert data["baseline_open_ear"] == 0.32
    assert data["calibrated_ear_threshold"] == 0.24


def test_calibration_out_of_bounds_fallback(tmp_path: Path) -> None:
    """Verifies that an unphysiologically low baseline triggers safe fallback."""
    calib_json = tmp_path / "calibration.json"
    calibrator = EarCalibrator(
        duration_sec=5.0,
        ear_multiplier=0.75,
        output_path=str(calib_json),
        min_valid_baseline_ear=0.20,
        default_fallback_threshold=0.21,
    )

    t = 100.0
    # Driver kept eyes closed the whole time (EAR = 0.12 < 0.20 min valid)
    for _ in range(60):
        t += 0.1
        calibrator.add_sample(ear=0.12, timestamp=t)

    _baseline, thresh = calibrator.finish_calibration()
    # Should safely fall back to default threshold 0.21
    assert thresh == 0.21
