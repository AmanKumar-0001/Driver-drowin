"""Auditory alarm engine with cooldowns, volume limits, and non-blocking backends.

Provides distinct auditory signals for WARNING, DROWSY, DISTRACTED, and NO_FACE states.
Safeguards hearing through an enforced volume cap and maximum continuous duration limit.
"""

from __future__ import annotations

import io
import logging
import math
import struct
import sys
import threading
import time
import wave
from abc import ABC, abstractmethod

from src.fusion import SystemState

logger = logging.getLogger(__name__)


def generate_sine_wave_wav(frequency_hz: int, duration_sec: float, volume: float = 0.70) -> bytes:
    """Generates an in-memory 16-bit mono PCM WAV audio buffer.

    Args:
        frequency_hz: Audio pitch in Hertz.
        duration_sec: Tone duration in seconds.
        volume: Amplitude scalar in [0.0, 1.0].

    Returns:
        Raw bytes representing a valid RIFF/WAV file.
    """
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)
    amplitude = int(32767.0 * max(min(volume, 1.0), 0.0))

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(total_samples):
            t = float(i) / sample_rate
            val = int(amplitude * math.sin(2.0 * math.pi * frequency_hz * t))
            frames.extend(struct.pack("<h", val))

        wf.writeframes(frames)

    return wav_buf.getvalue()


class AudioBackend(ABC):
    """Abstract interface for audio output engines."""

    @abstractmethod
    def play_tone(self, frequency_hz: int, duration_sec: float, volume: float) -> None:
        """Plays an audible tone non-blockingly."""

    @abstractmethod
    def stop(self) -> None:
        """Stops any currently playing audio."""

    @abstractmethod
    def close(self) -> None:
        """Releases underlying audio hardware resources."""


class MockAudioBackend(AudioBackend):
    """Mock audio backend for deterministic headless testing."""

    def __init__(self) -> None:
        self.play_history: list[tuple[float, int, float, float]] = []
        self.stopped_count: int = 0
        self.is_closed: bool = False

    def play_tone(self, frequency_hz: int, duration_sec: float, volume: float) -> None:
        self.play_history.append((time.time(), frequency_hz, duration_sec, volume))

    def stop(self) -> None:
        self.stopped_count += 1

    def close(self) -> None:
        self.is_closed = True


class WinsoundAudioBackend(AudioBackend):
    """Windows native audio backend using winsound in a background thread."""

    def __init__(self) -> None:
        import winsound  # type: ignore

        self._winsound = winsound
        self._thread: threading.Thread | None = None

    def play_tone(self, frequency_hz: int, duration_sec: float, volume: float) -> None:
        def _worker() -> None:
            try:
                # winsound.Beep takes duration in milliseconds
                self._winsound.Beep(int(frequency_hz), int(duration_sec * 1000))
            except (RuntimeError, OSError) as e:
                logger.debug("winsound playback error: %s", e)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class PygameAudioBackend(AudioBackend):
    """Cross-platform audio backend using pygame.mixer and synthesized WAV buffers."""

    def __init__(self) -> None:
        import pygame

        self._pygame = pygame
        if not pygame.mixer.get_init():
            # Initialize with standard CD quality parameters
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        self._current_sound: pygame.mixer.Sound | None = None

    def play_tone(self, frequency_hz: int, duration_sec: float, volume: float) -> None:
        try:
            wav_bytes = generate_sine_wave_wav(frequency_hz, duration_sec, volume)
            sound = self._pygame.mixer.Sound(io.BytesIO(wav_bytes))
            sound.set_volume(max(min(volume, 1.0), 0.0))
            sound.play()
            self._current_sound = sound
        except (RuntimeError, OSError) as e:
            logger.debug("pygame audio play error: %s", e)

    def stop(self) -> None:
        try:
            if self._pygame.mixer.get_init():
                self._pygame.mixer.stop()
        except (RuntimeError, OSError) as e:
            logger.debug("pygame stop error: %s", e)

    def close(self) -> None:
        try:
            if self._pygame.mixer.get_init():
                self._pygame.mixer.quit()
        except (RuntimeError, OSError) as e:
            logger.debug("pygame quit error: %s", e)


class AlarmManager:
    """Manages audio alarm dispatching, cooldowns, and acoustic safety bounds."""

    def __init__(
        self,
        backend: AudioBackend | None = None,
        backend_name: str = "auto",
        alarm_cooldown_sec: float = 5.0,
        volume_cap: float = 0.70,
        max_continuous_alarm_sec: float = 10.0,
        warning_frequency_hz: int = 440,
        drowsy_frequency_hz: int = 800,
        distracted_frequency_hz: int = 600,
        fault_frequency_hz: int = 1000,
    ) -> None:
        self.alarm_cooldown_sec = max(alarm_cooldown_sec, 0.0)
        self.volume_cap = max(min(volume_cap, 1.0), 0.0)
        self.max_continuous_alarm_sec = max(max_continuous_alarm_sec, 1.0)
        self.warning_frequency_hz = warning_frequency_hz
        self.drowsy_frequency_hz = drowsy_frequency_hz
        self.distracted_frequency_hz = distracted_frequency_hz
        self.fault_frequency_hz = fault_frequency_hz

        self.muted: bool = False
        self._last_play_time: float | None = None
        self._continuous_start_time: float | None = None
        self._safety_pause_until: float | None = None

        if backend is not None:
            self.backend = backend
        else:
            self.backend = self._init_backend(backend_name)

    @staticmethod
    def _init_backend(backend_name: str) -> AudioBackend:
        """Instantiates the preferred or available audio driver."""
        if backend_name == "mock":
            return MockAudioBackend()

        if backend_name in ("auto", "pygame"):
            try:
                return PygameAudioBackend()
            except Exception:
                if backend_name == "pygame":
                    raise

        if backend_name in ("auto", "winsound") and sys.platform.startswith("win"):
            try:
                return WinsoundAudioBackend()
            except (RuntimeError, OSError) as e:
                logger.debug("winsound initialization failed: %s", e)

        return MockAudioBackend()

    def toggle_mute(self) -> bool:
        """Toggles audio mute state. Returns the new muted boolean."""
        self.muted = not self.muted
        if self.muted:
            self.backend.stop()
        return self.muted

    def update(self, state: SystemState, timestamp: float | None = None) -> bool:
        """Evaluates system state and triggers appropriate alarm tone if permitted.

        Args:
            state: Current SystemState from the FSM.
            timestamp: Epoch/monotonic time in seconds.

        Returns:
            True if an alarm tone was dispatched, False otherwise.
        """
        curr_t = float(time.time() if timestamp is None else timestamp)

        # In nominal ALERT state, stop alarm and reset continuous counter
        if state == SystemState.ALERT:
            self._continuous_start_time = None
            self._safety_pause_until = None
            self.backend.stop()
            return False

        if self.muted:
            return False

        # Check safety pause (continuous alarm limit reached)
        if self._safety_pause_until is not None:
            if curr_t < self._safety_pause_until:
                return False  # Currently in safety silence pause
            self._safety_pause_until = None
            self._continuous_start_time = curr_t

        # Track continuous alarm duration
        if self._continuous_start_time is None:
            self._continuous_start_time = curr_t
        elif (curr_t - self._continuous_start_time) >= self.max_continuous_alarm_sec:
            # Enforce 5.0 second safety silence pause
            self._safety_pause_until = curr_t + 5.0
            self._continuous_start_time = None
            self.backend.stop()
            return False

        # Verify cooldown
        if self._last_play_time is not None:
            time_since_last = curr_t - self._last_play_time
            if time_since_last < self.alarm_cooldown_sec:
                return False

        # Dispatch tone corresponding to state
        freq, dur = self._get_tone_params(state)
        self.backend.play_tone(frequency_hz=freq, duration_sec=dur, volume=self.volume_cap)
        self._last_play_time = curr_t
        return True

    def _get_tone_params(self, state: SystemState) -> tuple[int, float]:
        """Returns (frequency_hz, duration_sec) for a given hazard state."""
        if state == SystemState.WARNING:
            return self.warning_frequency_hz, 0.40
        if state == SystemState.DROWSY:
            return self.drowsy_frequency_hz, 0.80
        if state == SystemState.DISTRACTED:
            return self.distracted_frequency_hz, 0.60
        # NO_FACE or FAULT
        return self.fault_frequency_hz, 1.00

    def close(self) -> None:
        """Shuts down audio subsystem."""
        self.backend.stop()
        self.backend.close()
