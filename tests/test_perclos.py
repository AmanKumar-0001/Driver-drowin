"""Unit tests for sliding window PERCLOS tracking and variable frame rates."""

import pytest

from src.perclos import PerclosTracker


def test_perclos_empty_and_short_history() -> None:
    """Verifies that an empty or insufficiently populated buffer returns 0.0."""
    tracker = PerclosTracker(window_sec=60.0, min_history_sec=2.0)
    assert tracker.get_perclos() == 0.0
    assert tracker.update(is_closed=True, timestamp=100.0) == 0.0
    # Elapsed is only 1.0s, which is < min_history_sec (2.0s)
    assert tracker.update(is_closed=True, timestamp=101.0) == 0.0


def test_perclos_all_closed_continuous() -> None:
    """Verifies that continuous eye closure over 10s yields PERCLOS = 1.0."""
    tracker = PerclosTracker(window_sec=60.0, min_history_sec=2.0)
    t = 1000.0
    # Feed 10 seconds of closed observations at 10 Hz
    for _ in range(100):
        t += 0.1
        score = tracker.update(is_closed=True, timestamp=t)

    assert score == pytest.approx(1.0, abs=1e-4)


def test_perclos_fifty_percent_duty_cycle() -> None:
    """Verifies that alternating 1s open / 1s closed yields PERCLOS ~ 0.50."""
    tracker = PerclosTracker(window_sec=60.0, min_history_sec=2.0)
    t = 1000.0
    for _cycle in range(20):
        # 1.0s closed (10 steps of 0.1s)
        for _ in range(10):
            t += 0.1
            tracker.update(is_closed=True, timestamp=t)
        # 1.0s open (10 steps of 0.1s)
        for _ in range(10):
            t += 0.1
            tracker.update(is_closed=False, timestamp=t)

    score = tracker.get_perclos()
    assert score == pytest.approx(0.50, abs=0.05)


def test_perclos_window_eviction() -> None:
    """Verifies that observations older than window_sec are completely evicted."""
    window = 10.0
    tracker = PerclosTracker(window_sec=window, min_history_sec=2.0)
    t = 0.0

    # First 10 seconds: completely closed
    for _ in range(100):
        t += 0.1
        tracker.update(is_closed=True, timestamp=t)
    assert tracker.get_perclos() == pytest.approx(1.0, abs=1e-3)

    # Next 15 seconds: completely open
    for _ in range(150):
        t += 0.1
        tracker.update(is_closed=False, timestamp=t)

    # The 10s closed window should be completely evicted from the active 10s window
    assert tracker.get_perclos() == pytest.approx(0.0, abs=1e-3)


def test_perclos_variable_framerate_integration() -> None:
    """Verifies that variable delta-t (e.g. frame drops) is integrated correctly."""
    tracker = PerclosTracker(window_sec=60.0, min_history_sec=1.0)
    # Event 1: Closed for 3.0 seconds (from t=0.0 to t=3.0)
    tracker.update(is_closed=True, timestamp=0.0)
    tracker.update(is_closed=False, timestamp=3.0)

    # Event 2: Open for 1.0 second (from t=3.0 to t=4.0)
    tracker.update(is_closed=False, timestamp=4.0)

    # Total elapsed: 4.0s. Closed duration: 3.0s.
    # Expected PERCLOS = 3.0 / 4.0 = 0.75
    assert tracker.get_perclos() == pytest.approx(0.75, abs=1e-3)


def test_perclos_reset() -> None:
    """Verifies that reset clears all internal history."""
    tracker = PerclosTracker(window_sec=60.0, min_history_sec=2.0)
    tracker.update(is_closed=True, timestamp=1.0)
    tracker.update(is_closed=True, timestamp=5.0)
    assert tracker.sample_count == 2
    tracker.reset()
    assert tracker.sample_count == 0
    assert tracker.get_perclos() == 0.0
