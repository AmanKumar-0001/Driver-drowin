"""Unit tests for geometric facial feature extraction (EAR, MAR, Blinks)."""

import numpy as np
import pytest

from src.features import (
    MOUTH_BOTTOM_INDICES,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
    MOUTH_TOP_INDICES,
    calculate_average_ear,
    calculate_mar,
    classify_closure_duration,
    compute_aspect_ratio_6pt,
)


def test_aspect_ratio_exact_geometry() -> None:
    """Verifies that an idealized synthetic rectangle yields exact theoretical aspect ratio."""
    # Rectangle of width 20, height 4:
    # p1=(-10, 0), p4=(10, 0) -> width = 20
    # p2=(-5, 2), p6=(-5, -2) -> v1 = 4
    # p3=(5, 2), p5=(5, -2) -> v2 = 4
    # Ratio = (4 + 4) / (2 * 20) = 8 / 40 = 0.20
    p1 = (-10.0, 0.0)
    p2 = (-5.0, 2.0)
    p3 = (5.0, 2.0)
    p4 = (10.0, 0.0)
    p5 = (5.0, -2.0)
    p6 = (-5.0, -2.0)

    ratio = compute_aspect_ratio_6pt(p1, p2, p3, p4, p5, p6)
    assert ratio == pytest.approx(0.20, abs=1e-6)


def test_aspect_ratio_zero_width_epsilon_protection() -> None:
    """Ensures degenerate co-located corner points do not cause ZeroDivisionError."""
    p_zero = (0.0, 0.0)
    ratio = compute_aspect_ratio_6pt(p_zero, p_zero, p_zero, p_zero, p_zero, p_zero)
    assert ratio == 0.0
    assert not np.isnan(ratio)


def test_calculate_average_ear_synthetic_mesh() -> None:
    """Tests EAR calculation across a synthetic 478-point landmark array."""
    landmarks = np.zeros((478, 3), dtype=np.float64)

    # Populate left eye as open (aspect ratio ~ 0.30)
    # p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
    landmarks[33] = [-10.0, 0.0, 0.0]
    landmarks[160] = [-5.0, 3.0, 0.0]
    landmarks[158] = [5.0, 3.0, 0.0]
    landmarks[133] = [10.0, 0.0, 0.0]
    landmarks[153] = [5.0, -3.0, 0.0]
    landmarks[144] = [-5.0, -3.0, 0.0]

    # Populate right eye as closed (aspect ratio ~ 0.10)
    # p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
    landmarks[362] = [-10.0, 0.0, 0.0]
    landmarks[385] = [-5.0, 1.0, 0.0]
    landmarks[387] = [5.0, 1.0, 0.0]
    landmarks[263] = [10.0, 0.0, 0.0]
    landmarks[373] = [5.0, -1.0, 0.0]
    landmarks[380] = [-5.0, -1.0, 0.0]

    left_ear, right_ear, avg_ear = calculate_average_ear(landmarks)
    assert left_ear == pytest.approx(0.30, abs=1e-5)
    assert right_ear == pytest.approx(0.10, abs=1e-5)
    assert avg_ear == pytest.approx(0.20, abs=1e-5)


def test_calculate_mar_closed_vs_yawning() -> None:
    """Verifies MAR values for simulated neutral mouth vs. wide open yawn."""
    landmarks = np.zeros((478, 3), dtype=np.float64)

    # Mouth corners width = 40 (left=-20, right=20)
    landmarks[MOUTH_LEFT_CORNER] = [-20.0, 0.0, 0.0]
    landmarks[MOUTH_RIGHT_CORNER] = [20.0, 0.0, 0.0]

    # Neutral closed mouth: vertical gap = 4
    for t_idx, b_idx, x in zip(MOUTH_TOP_INDICES, MOUTH_BOTTOM_INDICES, [-10.0, 0.0, 10.0]):
        landmarks[t_idx] = [x, 2.0, 0.0]
        landmarks[b_idx] = [x, -2.0, 0.0]

    mar_closed = calculate_mar(landmarks)
    # Expected: (4 + 4 + 4) / (2 * 40) = 12 / 80 = 0.15
    assert mar_closed == pytest.approx(0.15, abs=1e-5)

    # Yawning mouth: vertical gap = 24
    for t_idx, b_idx, x in zip(MOUTH_TOP_INDICES, MOUTH_BOTTOM_INDICES, [-10.0, 0.0, 10.0]):
        landmarks[t_idx] = [x, 12.0, 0.0]
        landmarks[b_idx] = [x, -12.0, 0.0]

    mar_yawn = calculate_mar(landmarks)
    # Expected: (24 + 24 + 24) / (2 * 40) = 72 / 80 = 0.90
    assert mar_yawn == pytest.approx(0.90, abs=1e-5)
    assert mar_yawn > 0.60  # Satisfies yawn threshold


@pytest.mark.parametrize(
    "duration,expected",
    [
        (0.00, "OPEN"),
        (-0.5, "OPEN"),
        (0.15, "NORMAL_BLINK"),
        (0.35, "NORMAL_BLINK"),
        (0.60, "PROLONGED_CLOSURE"),
        (1.00, "MICROSLEEP"),
        (2.50, "MICROSLEEP"),
    ],
)
def test_classify_closure_duration(duration: float, expected: str) -> None:
    """Verifies physiological categorization of closure duration."""
    result = classify_closure_duration(duration, blink_max_sec=0.40, microsleep_min_sec=1.00)
    assert result == expected
