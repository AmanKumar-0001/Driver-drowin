"""Structured CSV session logging for state transitions and driver telemetry.

Records timestamped numeric metrics and discrete events without storing images,
guaranteeing driver privacy. Provides aggregate performance metrics upon session close.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CSV_HEADERS = [
    "timestamp_iso",
    "elapsed_sec",
    "state",
    "face_detected",
    "ear",
    "mar",
    "perclos",
    "yaw_deg",
    "pitch_deg",
    "cnn_closed_prob",
    "eye_closed_fused",
    "model_ear_disagreement",
    "alarm_dispatched",
    "fps",
    "hazard_reason",
]


class SessionLogger:
    """Manages CSV metric recording and session lifecycle auditing."""

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{session_id}.csv"

        self._fh = open(self.log_file, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.writer(self._fh)
        self._writer.writerow(CSV_HEADERS)
        self._fh.flush()

        self._start_time = datetime.now(timezone.utc)
        self._record_count = 0
        self._state_counts: dict[str, int] = {}
        self._alarm_count = 0
        self._fps_samples: list[float] = []

        logger.info("Initialized session audit log at %s", self.log_file)

    def log_frame(
        self,
        elapsed_sec: float,
        state: str,
        face_detected: bool,
        ear: float,
        mar: float,
        perclos: float,
        yaw: float,
        pitch: float,
        eye_closed_fused: bool,
        fps: float,
        hazard_reason: str,
        cnn_closed_prob: float | None = None,
        model_ear_disagreement: bool = False,
        alarm_dispatched: bool = False,
    ) -> None:
        """Appends a single telemetry record to the active session log."""
        iso_ts = datetime.now(timezone.utc).isoformat()
        row = [
            iso_ts,
            f"{elapsed_sec:.3f}",
            state,
            str(face_detected),
            f"{ear:.4f}",
            f"{mar:.4f}",
            f"{perclos:.4f}",
            f"{yaw:.2f}",
            f"{pitch:.2f}",
            f"{cnn_closed_prob:.4f}" if cnn_closed_prob is not None else "",
            str(eye_closed_fused),
            str(model_ear_disagreement),
            str(alarm_dispatched),
            f"{fps:.1f}",
            hazard_reason.replace(",", ";"),
        ]
        self._writer.writerow(row)
        self._record_count += 1
        self._state_counts[state] = self._state_counts.get(state, 0) + 1
        if alarm_dispatched:
            self._alarm_count += 1
        if fps > 0.0:
            self._fps_samples.append(fps)

        if self._record_count % 30 == 0:
            self._fh.flush()

    def get_summary(self) -> dict[str, Any]:
        """Calculates aggregate metrics for the concluded session."""
        avg_fps = sum(self._fps_samples) / len(self._fps_samples) if self._fps_samples else 0.0
        return {
            "total_frames": self._record_count,
            "duration_sec": (datetime.now(timezone.utc) - self._start_time).total_seconds(),
            "average_fps": round(avg_fps, 2),
            "alarm_triggers": self._alarm_count,
            "state_distribution": self._state_counts,
            "log_file": str(self.log_file),
        }

    def close(self) -> None:
        """Flushes buffers and safely closes the underlying CSV file handle."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
            logger.info("Session log finalized with %d records.", self._record_count)
