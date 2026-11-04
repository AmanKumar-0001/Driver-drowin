"""Time-aware sliding window PERCLOS (Percentage of Eye Closure) tracking.

Maintains a circular buffer of timestamped eye state observations and computes
the percentage of time the eyes were closed over a configurable window (default 60s).
Handles variable frame rates and dropped frames with mathematical precision.
"""

from __future__ import annotations

import time
from collections import deque


class PerclosTracker:
    """Computes time-weighted PERCLOS over a continuous sliding temporal window."""

    def __init__(self, window_sec: float = 60.0, min_history_sec: float = 2.0) -> None:
        """Initializes the PERCLOS tracker.

        Args:
            window_sec: Length of sliding historical window in seconds (default 60s).
            min_history_sec: Minimum elapsed duration required before emitting non-zero values.
        """
        if window_sec <= 0.0:
            raise ValueError(f"window_sec must be positive, got {window_sec}")
        self.window_sec: float = float(window_sec)
        self.min_history_sec: float = float(min_history_sec)
        # Internal buffer holds tuples: (timestamp, is_closed)
        self._history: deque[tuple[float, bool]] = deque()

    def update(self, is_closed: bool, timestamp: float | None = None) -> float:
        """Appends a new observation and returns the updated PERCLOS score.

        Args:
            is_closed: True if eyes are classified as closed, False otherwise.
            timestamp: Epoch timestamp in seconds. Defaults to time.time().

        Returns:
            Calculated PERCLOS ratio in [0.0, 1.0].
        """
        curr_time = float(time.time() if timestamp is None else timestamp)

        # Enforce monotonic timestamps
        if self._history and curr_time < self._history[-1][0]:
            curr_time = self._history[-1][0]

        self._history.append((curr_time, bool(is_closed)))

        # Evict observations older than (curr_time - window_sec)
        cutoff_time = curr_time - self.window_sec
        while len(self._history) > 1 and self._history[0][0] < cutoff_time:
            # Check if second item is also before cutoff before popping to keep boundary sample
            if len(self._history) > 2 and self._history[1][0] <= cutoff_time:
                self._history.popleft()
            else:
                break

        return self.get_perclos()

    def get_perclos(self) -> float:
        """Calculates the current PERCLOS ratio across the active buffer.

        Returns:
            Proportion of eye closure time [0.0, 1.0].
        """
        if len(self._history) < 2:
            return 0.0

        total_elapsed = self._history[-1][0] - self._history[0][0]
        if total_elapsed < self.min_history_sec or total_elapsed <= 0.0:
            return 0.0

        closed_duration = 0.0
        # Integrate time step-by-step between consecutive observation points
        for i in range(1, len(self._history)):
            prev_t, prev_closed = self._history[i - 1]
            curr_t, _ = self._history[i]
            dt = curr_t - prev_t
            if dt > 0.0 and prev_closed:
                closed_duration += dt

        return min(max(closed_duration / total_elapsed, 0.0), 1.0)

    def reset(self) -> None:
        """Clears all historical observations."""
        self._history.clear()

    @property
    def sample_count(self) -> int:
        """Returns the number of observations currently in the buffer."""
        return len(self._history)

    @property
    def active_duration(self) -> float:
        """Returns the time span currently covered by the buffer in seconds."""
        if len(self._history) < 2:
            return 0.0
        return self._history[-1][0] - self._history[0][0]
