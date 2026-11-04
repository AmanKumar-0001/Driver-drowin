"""Pure Finite State Machine (FSM) for temporal sensor and decision fusion.

Fuses rule-based vision metrics (EAR, MAR, PERCLOS, head pose) and deep learning
inferences with debouncing, priority resolution, and hysteresis.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass


class SystemState(str, enum.Enum):
    """Discrete operational states of the warning system."""

    ALERT = "ALERT"
    WARNING = "WARNING"
    DROWSY = "DROWSY"
    DISTRACTED = "DISTRACTED"
    NO_FACE = "NO_FACE"


@dataclass
class FSMDecision:
    """Output encapsulation of the FSM evaluation cycle."""

    state: SystemState
    previous_state: SystemState
    state_changed: bool
    eye_closed_fused: bool
    fused_closed_prob: float
    model_ear_disagreement: bool
    active_hazard_reason: str
    debounce_progress: float  # [0.0, 1.0] progress toward triggering current candidate


class DrowsinessFSM:
    """Pure Finite State Machine managing driver state transitions and priority."""

    def __init__(
        self,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.60,
        yawn_min_duration_sec: float = 1.0,
        perclos_threshold: float = 0.35,
        yaw_threshold_deg: float = 30.0,
        pitch_min_deg: float = -20.0,
        pitch_max_deg: float = 20.0,
        drowsiness_debounce_sec: float = 1.5,
        distraction_debounce_sec: float = 2.0,
        no_face_debounce_sec: float = 3.0,
        recovery_debounce_sec: float = 0.5,
        ear_weight: float = 0.50,
        cnn_weight: float = 0.50,
        fusion_threshold: float = 0.50,
    ) -> None:
        """Initializes the FSM with operational thresholds and debouncing timers."""
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.yawn_min_duration_sec = yawn_min_duration_sec
        self.perclos_threshold = perclos_threshold
        self.yaw_threshold_deg = yaw_threshold_deg
        self.pitch_min_deg = pitch_min_deg
        self.pitch_max_deg = pitch_max_deg
        self.drowsiness_debounce_sec = drowsiness_debounce_sec
        self.distraction_debounce_sec = distraction_debounce_sec
        self.no_face_debounce_sec = no_face_debounce_sec
        self.recovery_debounce_sec = recovery_debounce_sec
        self.ear_weight = ear_weight
        self.cnn_weight = cnn_weight
        self.fusion_threshold = fusion_threshold

        # Current state
        self._current_state = SystemState.ALERT

        # Candidate timer trackers: stores starting timestamp or None
        self._no_face_start: float | None = None
        self._drowsy_start: float | None = None
        self._distracted_start: float | None = None
        self._yawn_start: float | None = None
        self._recovery_start: float | None = None

    @property
    def current_state(self) -> SystemState:
        """Returns the active system state."""
        return self._current_state

    def reset(self) -> None:
        """Resets the state machine to default nominal state."""
        self._current_state = SystemState.ALERT
        self._no_face_start = None
        self._drowsy_start = None
        self._distracted_start = None
        self._yawn_start = None
        self._recovery_start = None

    def update(
        self,
        face_detected: bool,
        ear: float,
        perclos: float,
        mar: float,
        yaw: float,
        pitch: float,
        cnn_closed_prob: float | None = None,
        timestamp: float | None = None,
    ) -> FSMDecision:
        """Evaluates inputs and updates the state machine.

        Args:
            face_detected: Whether face mesh landmarks were successfully tracked.
            ear: Current average Eye Aspect Ratio.
            perclos: Current PERCLOS score in [0.0, 1.0].
            mar: Current Mouth Aspect Ratio.
            yaw: Current head yaw angle in degrees.
            pitch: Current head pitch angle in degrees.
            cnn_closed_prob: Optional probability of eyes closed from MobileNetV2.
            timestamp: Monotonic or epoch timestamp in seconds.

        Returns:
            FSMDecision summarizing state, transitions, and telemetry.
        """
        curr_t = float(time.time() if timestamp is None else timestamp)
        prev_state = self._current_state

        # 1. Eye State Fusion
        ear_closed_vote = 1.0 if ear < self.ear_threshold else 0.0
        disagreement = False
        if cnn_closed_prob is not None:
            cnn_closed_vote = float(np_clip(cnn_closed_prob, 0.0, 1.0))
            fused_prob = (self.ear_weight * ear_closed_vote) + (self.cnn_weight * cnn_closed_vote)
            eye_closed = fused_prob >= self.fusion_threshold
            # Disagreement flag: EAR and CNN hold opposing binary conclusions
            if (ear_closed_vote >= 0.5) != (cnn_closed_vote >= 0.5):
                disagreement = True
        else:
            fused_prob = ear_closed_vote
            eye_closed = ear_closed_vote >= 0.5

        # 2. Raw Candidate Hazard Assessment
        cand_no_face = not face_detected
        cand_drowsy = eye_closed or (perclos > self.perclos_threshold)
        cand_distracted = (abs(yaw) > self.yaw_threshold_deg) or (
            pitch < self.pitch_min_deg or pitch > self.pitch_max_deg
        )
        cand_yawn = mar > self.mar_threshold

        # 3. Update Debounce Timers
        self._no_face_start = self._update_timer(cand_no_face, self._no_face_start, curr_t)
        self._drowsy_start = self._update_timer(cand_drowsy, self._drowsy_start, curr_t)
        self._distracted_start = self._update_timer(cand_distracted, self._distracted_start, curr_t)
        self._yawn_start = self._update_timer(cand_yawn, self._yawn_start, curr_t)

        # 4. Check Debounce Durations
        dur_no_face = (curr_t - self._no_face_start) if self._no_face_start is not None else 0.0
        dur_drowsy = (curr_t - self._drowsy_start) if self._drowsy_start is not None else 0.0
        dur_distracted = (
            (curr_t - self._distracted_start) if self._distracted_start is not None else 0.0
        )
        dur_yawn = (curr_t - self._yawn_start) if self._yawn_start is not None else 0.0

        sat_no_face = dur_no_face >= self.no_face_debounce_sec
        sat_drowsy = dur_drowsy >= self.drowsiness_debounce_sec
        sat_distracted = dur_distracted >= self.distraction_debounce_sec
        sat_yawn = dur_yawn >= self.yawn_min_duration_sec

        # 5. Resolve State via Strict Priority Order:
        # NO_FACE > DROWSY > DISTRACTED > WARNING > ALERT
        target_state = None
        hazard_reason = "Nominal Attention"
        debounce_prog = 0.0

        if sat_no_face:
            target_state = SystemState.NO_FACE
            hazard_reason = f"Face undetected for {dur_no_face:.1f}s"
            debounce_prog = 1.0
        elif sat_drowsy:
            target_state = SystemState.DROWSY
            hazard_reason = (
                f"Prolonged closure ({dur_drowsy:.1f}s)"
                if eye_closed
                else f"PERCLOS ({perclos:.2f}) > {self.perclos_threshold}"
            )
            debounce_prog = 1.0
        elif sat_distracted:
            target_state = SystemState.DISTRACTED
            hazard_reason = (
                f"Head turned (yaw={yaw:.1f}°, pitch={pitch:.1f}°) for {dur_distracted:.1f}s"
            )
            debounce_prog = 1.0
        elif sat_yawn:
            target_state = SystemState.WARNING
            hazard_reason = f"Sustained yawning (MAR={mar:.2f}) for {dur_yawn:.1f}s"
            debounce_prog = 1.0
        else:
            # No high-priority hazard has satisfied debounce; compute highest partial debounce progress
            debounce_prog = max(
                (
                    (dur_no_face / self.no_face_debounce_sec)
                    if self._no_face_start is not None
                    else 0.0
                ),
                (
                    (dur_drowsy / self.drowsiness_debounce_sec)
                    if self._drowsy_start is not None
                    else 0.0
                ),
                (
                    (dur_distracted / self.distraction_debounce_sec)
                    if self._distracted_start is not None
                    else 0.0
                ),
                ((dur_yawn / self.yawn_min_duration_sec) if self._yawn_start is not None else 0.0),
            )

        # 6. Apply Transitions with Recovery Debouncing
        if target_state is not None:
            # Immediate escalation to higher-priority or newly triggered hazard
            self._current_state = target_state
            self._recovery_start = None
        else:
            # No hazardous condition satisfied; handle recovery back toward ALERT
            if self._current_state != SystemState.ALERT:
                if self._recovery_start is None:
                    self._recovery_start = curr_t

                recovery_dur = curr_t - self._recovery_start
                if recovery_dur >= self.recovery_debounce_sec:
                    self._current_state = SystemState.ALERT
                    self._recovery_start = None
                    hazard_reason = "Recovered to Alert"
                else:
                    hazard_reason = (
                        f"Recovering ({recovery_dur:.1f}s / {self.recovery_debounce_sec:.1f}s)"
                    )
            else:
                self._recovery_start = None

        return FSMDecision(
            state=self._current_state,
            previous_state=prev_state,
            state_changed=(self._current_state != prev_state),
            eye_closed_fused=eye_closed,
            fused_closed_prob=fused_prob,
            model_ear_disagreement=disagreement,
            active_hazard_reason=hazard_reason,
            debounce_progress=min(debounce_prog, 1.0),
        )

    @staticmethod
    def _update_timer(is_active: bool, start_time: float | None, curr_time: float) -> float | None:
        """Helper to start, sustain, or reset a timestamp tracker."""
        if is_active:
            return curr_time if start_time is None else start_time
        return None


def np_clip(val: float, low: float, high: float) -> float:
    """Clamps a numeric scalar between low and high bounds."""
    return max(min(val, high), low)
