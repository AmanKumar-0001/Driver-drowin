"""Driver baseline EAR calibration engine.

Acquires open-eye EAR samples over a 10-second startup period to individualize
the operational drowsiness threshold (threshold = baseline * multiplier).
Filters involuntary blinks and validates physiological sanity boundaries.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class EarCalibrator:
    """Manages baseline open-eye EAR calibration and persistence."""

    def __init__(
        self,
        duration_sec: float = 10.0,
        ear_multiplier: float = 0.75,
        output_path: str = "config/calibration.json",
        min_valid_baseline_ear: float = 0.20,
        max_valid_baseline_ear: float = 0.45,
        default_fallback_threshold: float = 0.21,
    ) -> None:
        self.duration_sec = float(duration_sec)
        self.ear_multiplier = float(ear_multiplier)
        self.output_path = Path(output_path)
        self.min_valid_baseline_ear = float(min_valid_baseline_ear)
        self.max_valid_baseline_ear = float(max_valid_baseline_ear)
        self.default_fallback_threshold = float(default_fallback_threshold)

        self._start_time: float | None = None
        self._samples: list[tuple[float, float]] = []  # (timestamp, ear)
        self._is_completed: bool = False
        self._baseline_ear: float | None = None
        self._calibrated_threshold: float | None = None

    @property
    def is_complete(self) -> bool:
        """True if the calibration sequence has concluded."""
        return self._is_completed

    @property
    def progress(self) -> float:
        """Returns calibration progress in [0.0, 1.0]."""
        if self._is_completed:
            return 1.0
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return min(max(elapsed / self.duration_sec, 0.0), 1.0)

    def add_sample(self, ear: float, timestamp: float | None = None) -> bool:
        """Adds an EAR observation to the calibration buffer.

        Args:
            ear: Measured average Eye Aspect Ratio.
            timestamp: Optional epoch timestamp.

        Returns:
            True if calibration completed on this sample, False otherwise.
        """
        curr_t = float(time.time() if timestamp is None else timestamp)

        if self._start_time is None:
            self._start_time = curr_t

        if not self._is_completed:
            self._samples.append((curr_t, float(ear)))

            if (curr_t - self._start_time) >= self.duration_sec:
                self.finish_calibration()
                return True

        return False

    def finish_calibration(self) -> tuple[float, float]:
        """Finalizes calibration, calculates thresholds, and writes calibration.json.

        Returns:
            A tuple of (baseline_open_ear, calibrated_ear_threshold).
        """
        if not self._samples:
            logger.warning("No calibration samples recorded; falling back to default.")
            self._baseline_ear = self.default_fallback_threshold / self.ear_multiplier
            self._calibrated_threshold = self.default_fallback_threshold
            self._is_completed = True
            return self._baseline_ear, self._calibrated_threshold

        # Extract EAR values and sort
        ear_values = sorted([s[1] for s in self._samples])

        # Filter out the lowest 20% of samples (involuntary blinks)
        cutoff_index = int(len(ear_values) * 0.20)
        filtered_ears = ear_values[cutoff_index:] if cutoff_index < len(ear_values) else ear_values

        # Compute median of remaining open-eye samples
        median_idx = len(filtered_ears) // 2
        computed_baseline = filtered_ears[median_idx]

        # Physiological sanity validation
        if not (self.min_valid_baseline_ear <= computed_baseline <= self.max_valid_baseline_ear):
            logger.warning(
                "Computed baseline EAR (%.4f) outside physiological sanity bounds [%.2f, %.2f]. "
                "Falling back to default threshold %.2f",
                computed_baseline,
                self.min_valid_baseline_ear,
                self.max_valid_baseline_ear,
                self.default_fallback_threshold,
            )
            computed_baseline = self.default_fallback_threshold / self.ear_multiplier
            calibrated_thresh = self.default_fallback_threshold
        else:
            calibrated_thresh = computed_baseline * self.ear_multiplier

        self._baseline_ear = float(computed_baseline)
        self._calibrated_threshold = float(calibrated_thresh)
        self._is_completed = True

        self.save_calibration_file()
        return self._baseline_ear, self._calibrated_threshold

    def save_calibration_file(self) -> None:
        """Saves calibration results to JSON."""
        data = {
            "baseline_open_ear": round(float(self._baseline_ear or 0.0), 4),
            "calibrated_ear_threshold": round(float(self._calibrated_threshold or 0.0), 4),
            "ear_multiplier": self.ear_multiplier,
            "sample_count": len(self._samples),
            "duration_sec": self.duration_sec,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved calibration profile to %s", self.output_path)
