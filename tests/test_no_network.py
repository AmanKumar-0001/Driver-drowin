"""Verification test enforcing zero network socket activity during inference.

Patches standard socket network calls to guarantee strict air-gapped local execution
and ensure complete privacy for driver biometric telemetry.
"""

import socket
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.alarm import AlarmManager, MockAudioBackend
from src.config import load_config
from src.features import calculate_average_ear, calculate_mar
from src.fusion import DrowsinessFSM, SystemState
from src.logger import SessionLogger
from src.overlay import draw_hud
from src.perclos import PerclosTracker


class NetworkAccessAttemptedError(RuntimeError):
    """Raised when an unauthorized outbound network connection is attempted."""


def blocked_socket_connect(*args, **kwargs):
    raise NetworkAccessAttemptedError(
        f"Network socket connection blocked during inference: args={args}"
    )


def test_strict_zero_network_during_inference(tmp_path: Path) -> None:
    """Asserts that the entire inference and warning pipeline executes without network I/O."""
    # Monkey-patch socket connections to fail unconditionally
    with (
        patch.object(socket.socket, "connect", side_effect=blocked_socket_connect),
        patch.object(socket, "create_connection", side_effect=blocked_socket_connect),
    ):

        # 1. Load config
        cfg = load_config("config/default.yaml", calibration_path=None)
        assert cfg is not None

        # 2. Compute features on synthetic landmarks
        landmarks = np.zeros((478, 3), dtype=np.float64)
        _left_ear, _right_ear, avg_ear = calculate_average_ear(landmarks)
        mar = calculate_mar(landmarks)

        # 3. Update PERCLOS tracker
        perclos_tracker = PerclosTracker(window_sec=10.0, min_history_sec=1.0)
        perclos_score = perclos_tracker.update(is_closed=False, timestamp=100.0)

        # 4. Update FSM
        fsm = DrowsinessFSM()
        decision = fsm.update(
            face_detected=True,
            ear=avg_ear,
            perclos=perclos_score,
            mar=mar,
            yaw=0.0,
            pitch=0.0,
            timestamp=100.0,
        )
        assert decision.state == SystemState.ALERT

        # 5. Audio alarm update with mock backend
        alarm_mgr = AlarmManager(backend=MockAudioBackend())
        alarm_mgr.update(decision.state, timestamp=100.0)

        # 6. Session logging
        logger = SessionLogger(log_dir=tmp_path / "logs")
        logger.log_frame(
            elapsed_sec=1.0,
            state=decision.state.value,
            face_detected=True,
            ear=avg_ear,
            mar=mar,
            perclos=perclos_score,
            yaw=0.0,
            pitch=0.0,
            eye_closed_fused=False,
            fps=30.0,
            hazard_reason=decision.active_hazard_reason,
        )
        logger.close()

        # 7. Render HUD overlay
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered_frame = draw_hud(
            frame=synthetic_frame,
            state=decision.state,
            ear=avg_ear,
            ear_threshold=0.21,
            mar=mar,
            mar_threshold=0.60,
            perclos=perclos_score,
            perclos_threshold=0.35,
            yaw=0.0,
            pitch=0.0,
            fps=30.0,
            elapsed_sec=1.0,
            debounce_progress=0.0,
            hazard_reason=decision.active_hazard_reason,
            landmarks=landmarks,
        )
        assert rendered_frame.shape == (480, 640, 3)
