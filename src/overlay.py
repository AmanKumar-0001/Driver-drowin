"""Heads-Up Display (HUD) overlay and visual telemetry rendering.

Renders color-coded driver states, biometric telemetry gauges, debounce progress bars,
active camera indicators, and mandatory persistent safety disclaimers.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.features import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)
from src.fusion import SystemState

# Color definitions in BGR format
COLOR_GREEN = (40, 200, 40)
COLOR_YELLOW = (0, 215, 255)
COLOR_ORANGE = (0, 140, 255)
COLOR_RED = (40, 40, 230)
COLOR_MAGENTA = (200, 40, 200)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (20, 20, 20)
COLOR_GRAY = (120, 120, 120)

STATE_COLORS = {
    SystemState.ALERT: COLOR_GREEN,
    SystemState.WARNING: COLOR_YELLOW,
    SystemState.DROWSY: COLOR_RED,
    SystemState.DISTRACTED: COLOR_ORANGE,
    SystemState.NO_FACE: COLOR_MAGENTA,
}

DISCLAIMER_TEXT = (
    "RESEARCH PROTOTYPE. NOT A CERTIFIED SAFETY DEVICE. DO NOT RELY ON FOR DRIVING SAFETY."
)


def draw_hud(
    frame: np.ndarray,
    state: SystemState,
    ear: float,
    ear_threshold: float,
    mar: float,
    mar_threshold: float,
    perclos: float,
    perclos_threshold: float,
    yaw: float,
    pitch: float,
    fps: float,
    elapsed_sec: float,
    debounce_progress: float,
    hazard_reason: str,
    cnn_closed_prob: float | None = None,
    is_muted: bool = False,
    is_calibrating: bool = False,
    calib_progress: float = 0.0,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """Renders comprehensive HUD overlay atop the raw video frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Mandatory Safety Disclaimer Banner at bottom
    cv2.rectangle(canvas, (0, h - 26), (w, h), COLOR_BLACK, -1)
    cv2.putText(
        canvas,
        DISCLAIMER_TEXT,
        (10, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    # 2. Camera Active Indicator (Top-Right)
    cam_text = "CAM ACTIVE"
    cv2.circle(canvas, (w - 110, 20), 6, COLOR_RED, -1)
    cv2.putText(
        canvas,
        cam_text,
        (w - 95, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    if is_muted:
        cv2.putText(
            canvas,
            "[MUTED]",
            (w - 200, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            COLOR_ORANGE,
            2,
            cv2.LINE_AA,
        )

    # 3. Calibration Screen Override
    if is_calibrating:
        cv2.rectangle(canvas, (w // 8, h // 4), (7 * w // 8, 3 * h // 4), COLOR_BLACK, -1)
        cv2.rectangle(canvas, (w // 8, h // 4), (7 * w // 8, 3 * h // 4), COLOR_YELLOW, 2)
        cv2.putText(
            canvas,
            "BASELINE CALIBRATION IN PROGRESS",
            (w // 8 + 20, h // 4 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_YELLOW,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Look forward with eyes naturally open...",
            (w // 8 + 20, h // 4 + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )
        # Calibration Progress Bar
        bar_x = w // 8 + 20
        bar_y = h // 4 + 110
        bar_w = 6 * w // 8 - 40
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), COLOR_GRAY, 1)
        fill_w = int(bar_w * calib_progress)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + 20), COLOR_YELLOW, -1)
        return canvas

    # 4. Color-Coded State Badge (Top-Left)
    state_color = STATE_COLORS.get(state, COLOR_WHITE)
    cv2.rectangle(canvas, (10, 10), (220, 50), state_color, -1)
    cv2.putText(
        canvas,
        state.value,
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        COLOR_BLACK if state == SystemState.WARNING else COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )

    # Debounce Progress Bar underneath state badge
    if debounce_progress > 0.0 and state == SystemState.ALERT:
        bar_w = int(210 * debounce_progress)
        cv2.rectangle(canvas, (10, 52), (10 + bar_w, 58), COLOR_ORANGE, -1)

    # 5. Telemetry Metrics Panel (Left column)
    panel_y = 80
    metrics = [
        f"EAR: {ear:.3f} (th={ear_threshold:.2f})",
        f"MAR: {mar:.3f} (th={mar_threshold:.2f})",
        f"PERCLOS: {perclos * 100:.1f}% (th={perclos_threshold * 100:.0f}%)",
        f"Head Pose: Y={yaw:.1f} P={pitch:.1f}",
        f"FPS: {fps:.1f} | T: {elapsed_sec:.0f}s",
    ]
    if cnn_closed_prob is not None:
        metrics.insert(3, f"CNN Closed P: {cnn_closed_prob:.2f}")

    for idx, line in enumerate(metrics):
        cv2.putText(
            canvas,
            line,
            (12, panel_y + (idx * 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    # Hazard explanation banner when not nominal
    if state != SystemState.ALERT:
        cv2.rectangle(
            canvas,
            (10, panel_y + len(metrics) * 20 + 5),
            (w - 10, panel_y + len(metrics) * 20 + 32),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            canvas,
            f"ALERT: {hazard_reason}",
            (15, panel_y + len(metrics) * 20 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            state_color,
            1,
            cv2.LINE_AA,
        )

    # 6. Optional Eye Contour Dots
    if landmarks is not None:
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            pt = landmarks[idx]
            px = int(pt[0] * w) if pt[0] <= 1.0 else int(pt[0])
            py = int(pt[1] * h) if pt[1] <= 1.0 else int(pt[1])
            cv2.circle(canvas, (px, py), 1, COLOR_YELLOW, -1)

    return canvas
