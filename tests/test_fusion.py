"""Unit tests for debounced finite state machine decision fusion."""

from src.fusion import DrowsinessFSM, SystemState


def test_initial_state_is_alert() -> None:
    """Verifies that the state machine initializes cleanly in ALERT."""
    fsm = DrowsinessFSM()
    assert fsm.current_state == SystemState.ALERT


def test_normal_blink_does_not_trigger_drowsy() -> None:
    """Simulates a 0.3s physiological blink; verifies state remains ALERT."""
    fsm = DrowsinessFSM(drowsiness_debounce_sec=1.5)
    t = 100.0

    # 0.3 seconds of closed eyes at 30 FPS (~9 frames)
    for _ in range(9):
        t += 0.033
        dec = fsm.update(
            face_detected=True,
            ear=0.15,  # Closed
            perclos=0.10,
            mar=0.20,
            yaw=0.0,
            pitch=0.0,
            timestamp=t,
        )
        assert dec.state == SystemState.ALERT

    # Eyes open again
    t += 0.033
    dec = fsm.update(
        face_detected=True,
        ear=0.30,  # Open
        perclos=0.10,
        mar=0.20,
        yaw=0.0,
        pitch=0.0,
        timestamp=t,
    )
    assert dec.state == SystemState.ALERT


def test_sustained_closure_triggers_drowsy() -> None:
    """Simulates 1.6s of continuous closed eyes; verifies transition to DROWSY at 1.5s."""
    fsm = DrowsinessFSM(drowsiness_debounce_sec=1.5)
    t = 100.0

    # Start closure at t=100.0; feed up to 1.4s of closure -> Still ALERT
    for i in range(15):
        dec = fsm.update(
            face_detected=True,
            ear=0.15,
            perclos=0.10,
            mar=0.20,
            yaw=0.0,
            pitch=0.0,
            timestamp=t + i * 0.10,
        )
        assert dec.state == SystemState.ALERT

    # Cross 1.5s boundary at 1.6s -> Transitions to DROWSY
    dec = fsm.update(
        face_detected=True,
        ear=0.15,
        perclos=0.10,
        mar=0.20,
        yaw=0.0,
        pitch=0.0,
        timestamp=t + 1.60,
    )
    assert dec.state == SystemState.DROWSY
    assert dec.state_changed is True


def test_sustained_head_yaw_triggers_distracted() -> None:
    """Simulates 2.1s of sustained head turn; verifies transition to DISTRACTED at 2.0s."""
    fsm = DrowsinessFSM(distraction_debounce_sec=2.0)
    t = 100.0

    # Start head turn at t=100.0; feed up to 1.9s with yaw = 35 deg -> Still ALERT
    for i in range(20):
        dec = fsm.update(
            face_detected=True,
            ear=0.30,
            perclos=0.10,
            mar=0.20,
            yaw=35.0,
            pitch=0.0,
            timestamp=t + i * 0.10,
        )
        assert dec.state == SystemState.ALERT

    # Cross 2.0s at 2.1s -> DISTRACTED
    dec = fsm.update(
        face_detected=True,
        ear=0.30,
        perclos=0.10,
        mar=0.20,
        yaw=35.0,
        pitch=0.0,
        timestamp=t + 2.10,
    )
    assert dec.state == SystemState.DISTRACTED


def test_sustained_yawn_triggers_warning() -> None:
    """Simulates 1.1s of sustained yawning; verifies transition to WARNING."""
    fsm = DrowsinessFSM(yawn_min_duration_sec=1.0)
    t = 100.0

    # Start yawn at t=100.0; feed up to 0.9s -> Still ALERT
    for i in range(10):
        dec = fsm.update(
            face_detected=True,
            ear=0.30,
            perclos=0.10,
            mar=0.75,
            yaw=0.0,
            pitch=0.0,
            timestamp=t + i * 0.10,
        )
        assert dec.state == SystemState.ALERT

    # Cross 1.0s at 1.1s -> WARNING
    dec = fsm.update(
        face_detected=True,
        ear=0.30,
        perclos=0.10,
        mar=0.75,
        yaw=0.0,
        pitch=0.0,
        timestamp=t + 1.10,
    )
    assert dec.state == SystemState.WARNING


def test_priority_order_drowsy_over_distracted() -> None:
    """Verifies that if both drowsiness and distraction criteria are met, DROWSY wins."""
    fsm = DrowsinessFSM(drowsiness_debounce_sec=1.5, distraction_debounce_sec=2.0)
    t = 100.0

    # Both conditions active for 2.5 seconds
    for _ in range(25):
        t += 0.10
        dec = fsm.update(
            face_detected=True,
            ear=0.12,  # Closed
            perclos=0.10,
            mar=0.20,
            yaw=40.0,  # Distracted
            pitch=0.0,
            timestamp=t,
        )

    assert dec.state == SystemState.DROWSY


def test_priority_order_no_face_over_drowsy() -> None:
    """Verifies that NO_FACE (after 3.0s) takes strict priority over DROWSY."""
    fsm = DrowsinessFSM(drowsiness_debounce_sec=1.5, no_face_debounce_sec=3.0)
    t = 100.0

    # Missing face for 3.5 seconds
    for _ in range(35):
        t += 0.10
        dec = fsm.update(
            face_detected=False,  # No face
            ear=0.0,
            perclos=0.80,
            mar=0.0,
            yaw=0.0,
            pitch=0.0,
            timestamp=t,
        )

    assert dec.state == SystemState.NO_FACE


def test_recovery_debounce_to_alert() -> None:
    """Verifies that recovery from DROWSY back to ALERT requires 0.5s of nominal attention."""
    fsm = DrowsinessFSM(drowsiness_debounce_sec=1.0, recovery_debounce_sec=0.5)
    t = 100.0

    # Trigger DROWSY (1.2s closure)
    for i in range(13):
        dec = fsm.update(
            face_detected=True,
            ear=0.10,
            perclos=0.10,
            mar=0.20,
            yaw=0.0,
            pitch=0.0,
            timestamp=t + i * 0.10,
        )
    assert dec.state == SystemState.DROWSY

    # Normal conditions for 0.3s (< recovery_debounce of 0.5s) -> Remains DROWSY
    rec_start = t + 1.20
    for i in range(4):
        dec = fsm.update(
            face_detected=True,
            ear=0.30,
            perclos=0.10,
            mar=0.20,
            yaw=0.0,
            pitch=0.0,
            timestamp=rec_start + i * 0.10,
        )
        assert dec.state == SystemState.DROWSY

    # Cross 0.5s recovery duration at 0.6s -> Recovers to ALERT
    dec = fsm.update(
        face_detected=True,
        ear=0.30,
        perclos=0.10,
        mar=0.20,
        yaw=0.0,
        pitch=0.0,
        timestamp=rec_start + 0.60,
    )
    assert dec.state == SystemState.ALERT


def test_cnn_ear_disagreement_flag() -> None:
    """Verifies that disagreement is correctly flagged when EAR says open but CNN says closed."""
    fsm = DrowsinessFSM(ear_threshold=0.21)
    dec = fsm.update(
        face_detected=True,
        ear=0.30,  # EAR says OPEN (0.30 >= 0.21)
        perclos=0.0,
        mar=0.20,
        yaw=0.0,
        pitch=0.0,
        cnn_closed_prob=0.85,  # CNN says CLOSED (0.85 >= 0.50)
        timestamp=100.0,
    )
    assert dec.model_ear_disagreement is True
