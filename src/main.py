"""Primary application entry point and real-time execution orchestrator.

Coordinates camera capture, MediaPipe landmark tracking, geometric feature computation,
MobileNetV2 eye-state verification, debounced FSM fusion, audio alarms, HUD display, and logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from src.alarm import AlarmManager
from src.calibration import EarCalibrator
from src.camera import CameraReader, SyntheticCameraReader
from src.config import AppConfig, load_config
from src.features import calculate_average_ear, calculate_mar
from src.fusion import DrowsinessFSM, SystemState
from src.head_pose import estimate_head_pose
from src.inference import EyeStatePredictor, crop_eye_region
from src.landmarks import FaceLandmarkDetector
from src.logger import SessionLogger
from src.overlay import draw_hud

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONSENT_AND_SAFETY_NOTICE = """
================================================================================
DRIVER DROWSINESS AND DISTRACTION WARNING SYSTEM
FINAL-YEAR RESEARCH PROTOTYPE — AIML / COMPUTER VISION

SAFETY DISCLAIMER:
- This system is a laboratory research prototype and NOT a certified automotive safety device.
- NEVER rely on this software for vehicle driving safety.
- The software operates strictly in advisory mode and NEVER sends commands to any vehicle,
  steering, braking, or throttle system.

PRIVACY & ETHICS NOTICE:
- All processing is conducted locally in volatile RAM. No image frames or facial
  representations are transmitted over any network interface.
- Raw video is never stored to disk by default.
- By proceeding, you confirm that you are the monitored subject or have obtained
  explicit informed consent from the subject.
================================================================================
"""


def parse_arguments() -> argparse.Namespace:
    """Parses command-line configuration arguments."""
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness and Distraction Warning System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="config/calibration.json",
        help="Path to calibration.json",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true", help="Skip startup EAR calibration"
    )
    parser.add_argument("--no-display", action="store_true", help="Run headless without GUI window")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Opt-in: save anonymized eye crops for debug",
    )
    parser.add_argument(
        "--synthetic-camera",
        action="store_true",
        help="Use synthetic video feed for testing",
    )
    parser.add_argument(
        "--rule-based-only", action="store_true", help="Disable CNN inference entirely"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Exit after processing N frames (for benchmarking/CI)",
    )
    return parser.parse_args()


def run_system(args: argparse.Namespace) -> None:
    """Runs the main real-time perception and warning loop."""
    print(CONSENT_AND_SAFETY_NOTICE)

    # 1. Load and validate configuration
    cfg: AppConfig = load_config(args.config, calibration_path=args.calibration)
    if args.no_display:
        cfg.logging.show_display = False
    if args.save_frames:
        cfg.logging.save_frames = True
    if args.rule_based_only:
        cfg.model.cnn_enabled = False

    # 2. Instantiate system components
    camera = (
        SyntheticCameraReader(
            cfg.camera.frame_width, cfg.camera.frame_height, cfg.camera.target_fps
        )
        if args.synthetic_camera
        else CameraReader(
            cfg.camera.camera_index,
            cfg.camera.frame_width,
            cfg.camera.frame_height,
            cfg.camera.target_fps,
            cfg.camera.max_consecutive_dropped_frames,
        )
    )

    detector = FaceLandmarkDetector(
        max_num_faces=cfg.landmarks.max_num_faces,
        refine_landmarks=cfg.landmarks.refine_landmarks,
        min_detection_confidence=cfg.landmarks.min_detection_confidence,
        min_tracking_confidence=cfg.landmarks.min_tracking_confidence,
    )

    from src.perclos import PerclosTracker

    perclos_tracker = PerclosTracker(
        window_sec=cfg.drowsiness.perclos_window_sec,
        min_history_sec=2.0,
    )

    fsm = DrowsinessFSM(
        ear_threshold=cfg.drowsiness.ear_threshold,
        mar_threshold=cfg.drowsiness.mar_threshold,
        yawn_min_duration_sec=cfg.drowsiness.yawn_min_duration_sec,
        perclos_threshold=cfg.drowsiness.perclos_threshold,
        yaw_threshold_deg=cfg.distraction.yaw_threshold_deg,
        pitch_min_deg=cfg.distraction.pitch_min_deg,
        pitch_max_deg=cfg.distraction.pitch_max_deg,
        drowsiness_debounce_sec=cfg.drowsiness.drowsiness_debounce_sec,
        distraction_debounce_sec=cfg.distraction.distraction_debounce_sec,
        no_face_debounce_sec=cfg.distraction.no_face_debounce_sec,
        recovery_debounce_sec=cfg.drowsiness.recovery_debounce_sec,
        ear_weight=cfg.model.ear_weight,
        cnn_weight=cfg.model.cnn_weight,
        fusion_threshold=cfg.model.fusion_threshold,
    )

    alarm_mgr = AlarmManager(
        backend_name=cfg.alarm.audio_backend,
        alarm_cooldown_sec=cfg.alarm.alarm_cooldown_sec,
        volume_cap=cfg.alarm.volume_cap,
        max_continuous_alarm_sec=cfg.alarm.max_continuous_alarm_sec,
        warning_frequency_hz=cfg.alarm.warning_frequency_hz,
        drowsy_frequency_hz=cfg.alarm.drowsy_frequency_hz,
        distracted_frequency_hz=cfg.alarm.distracted_frequency_hz,
        fault_frequency_hz=cfg.alarm.fault_frequency_hz,
    )

    session_logger = SessionLogger(log_dir=cfg.logging.log_dir)

    cnn_predictor = (
        EyeStatePredictor(model_path=cfg.model.model_path) if cfg.model.cnn_enabled else None
    )

    # 3. Open Video Source
    if not camera.open():
        logger.critical("Could not open camera device. Triggering audible fail-safe alarm.")
        alarm_mgr.update(SystemState.NO_FACE)
        time.sleep(1.0)
        alarm_mgr.close()
        session_logger.close()
        sys.exit(1)

    # 4. Calibration Management
    calibrator: EarCalibrator | None = None
    is_calibrating = False
    if not args.skip_calibration and not Path(args.calibration).is_file():
        logger.info("Initiating 10-second open-eye baseline calibration...")
        calibrator = EarCalibrator(
            duration_sec=cfg.calibration.calibration_duration_sec,
            ear_multiplier=cfg.calibration.ear_multiplier,
            output_path=args.calibration,
            min_valid_baseline_ear=cfg.calibration.min_valid_baseline_ear,
            max_valid_baseline_ear=cfg.calibration.max_valid_baseline_ear,
        )
        is_calibrating = True

    # 5. Main Execution Loop
    session_start_t = time.time()
    last_loop_t = session_start_t
    cached_cnn_prob: float | None = None
    frame_counter = 0

    fps_rolling = 30.0

    logger.info("Entering real-time monitoring loop. Press 'q' to quit, 'm' to mute.")

    try:
        while True:
            loop_start_t = time.time()
            elapsed_sec = loop_start_t - session_start_t
            frame_counter += 1

            if args.max_frames is not None and frame_counter >= args.max_frames:
                logger.info("Reached maximum frames limit (%d). Exiting loop.", args.max_frames)
                break

            # Frame Acquisition
            success, frame = camera.read()
            if not success or frame is None:
                # Tolerated dropped frame
                logger.debug("Dropped frame encountered at t=%.2f", elapsed_sec)
                continue

            # Landmark Detection
            landmarks = detector.process(frame)
            face_detected = landmarks is not None

            # Telemetry computation defaults
            ear = 0.0
            mar = 0.0
            perclos = 0.0
            yaw, pitch, _roll = 0.0, 0.0, 0.0

            if face_detected and landmarks is not None:
                _, _, ear = calculate_average_ear(landmarks)
                mar = calculate_mar(landmarks)
                yaw, pitch, _roll = estimate_head_pose(landmarks, frame.shape)

                # Active calibration cycle
                if is_calibrating and calibrator is not None:
                    calib_done = calibrator.add_sample(ear, loop_start_t)
                    if calib_done:
                        _, calibrated_th = calibrator.finish_calibration()
                        fsm.ear_threshold = calibrated_th
                        is_calibrating = False
                        logger.info(
                            "Calibration concluded! Threshold set to %.4f",
                            calibrated_th,
                        )
                else:
                    # Update PERCLOS
                    is_closed_geom = ear < fsm.ear_threshold
                    perclos = perclos_tracker.update(is_closed_geom, loop_start_t)

                    # MobileNetV2 Eye-State Inference (every N frames)
                    if (
                        cnn_predictor
                        and cnn_predictor.is_available
                        and (frame_counter % cfg.model.cnn_inference_interval == 0)
                    ):
                        cached_cnn_prob = cnn_predictor.predict(frame, landmarks)

            # Evaluate Debounced Finite State Machine
            decision = fsm.update(
                face_detected=face_detected,
                ear=ear,
                perclos=perclos,
                mar=mar,
                yaw=yaw,
                pitch=pitch,
                cnn_closed_prob=cached_cnn_prob,
                timestamp=loop_start_t,
            )

            # Update Auditory Alarm
            alarm_dispatched = alarm_mgr.update(decision.state, timestamp=loop_start_t)

            # Measure Loop FPS
            dt = time.time() - last_loop_t
            last_loop_t = time.time()
            instant_fps = 1.0 / max(dt, 1e-4)
            fps_rolling = 0.90 * fps_rolling + 0.10 * instant_fps

            # Session Logging
            session_logger.log_frame(
                elapsed_sec=elapsed_sec,
                state=decision.state.value,
                face_detected=face_detected,
                ear=ear,
                mar=mar,
                perclos=perclos,
                yaw=yaw,
                pitch=pitch,
                eye_closed_fused=decision.eye_closed_fused,
                fps=fps_rolling,
                hazard_reason=decision.active_hazard_reason,
                cnn_closed_prob=cached_cnn_prob,
                model_ear_disagreement=decision.model_ear_disagreement,
                alarm_dispatched=alarm_dispatched,
            )

            # Privacy-safe opt-in frame debug saving
            if (
                cfg.logging.save_frames
                and face_detected
                and landmarks is not None
                and frame_counter % 30 == 0
            ):
                eye_crop = crop_eye_region(frame, landmarks, [33, 160, 158, 133, 153, 144])
                if eye_crop is not None:
                    debug_dir = Path(cfg.logging.debug_frames_dir)
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(debug_dir / f"crop_{int(loop_start_t * 1000)}.png"),
                        cv2.cvtColor(eye_crop, cv2.COLOR_RGB2BGR),
                    )

            # Render Overlay and Handle Window Events
            if cfg.logging.show_display:
                calib_p = calibrator.progress if (is_calibrating and calibrator) else 0.0
                hud_frame = draw_hud(
                    frame=frame,
                    state=decision.state,
                    ear=ear,
                    ear_threshold=fsm.ear_threshold,
                    mar=mar,
                    mar_threshold=fsm.mar_threshold,
                    perclos=perclos,
                    perclos_threshold=fsm.perclos_threshold,
                    yaw=yaw,
                    pitch=pitch,
                    fps=fps_rolling,
                    elapsed_sec=elapsed_sec,
                    debounce_progress=decision.debounce_progress,
                    hazard_reason=decision.active_hazard_reason,
                    cnn_closed_prob=cached_cnn_prob,
                    is_muted=alarm_mgr.muted,
                    is_calibrating=is_calibrating,
                    calib_progress=calib_p,
                    landmarks=landmarks,
                )
                cv2.imshow("Driver Drowsiness & Distraction Warning System", hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    logger.info("Quit command received.")
                    break
                elif key == ord("m"):
                    is_muted = alarm_mgr.toggle_mute()
                    logger.info("Audio mute toggled: %s", is_muted)
                elif key == ord("c"):
                    logger.info("Manual calibration triggered.")
                    calibrator = EarCalibrator(
                        duration_sec=cfg.calibration.calibration_duration_sec,
                        ear_multiplier=cfg.calibration.ear_multiplier,
                        output_path=args.calibration,
                    )
                    is_calibrating = True

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt (Ctrl+C) caught. Initiating clean exit.")
    except Exception:
        logger.exception("Unexpected exception in primary execution loop")
    finally:
        # Clean Resource Teardown
        logger.info("Shutting down system subsystems...")
        camera.release()
        detector.close()
        alarm_mgr.close()
        cv2.destroyAllWindows()
        session_logger.close()
        summary = session_logger.get_summary()
        logger.info("Session Concluded: %s", summary)


def main() -> None:
    args = parse_arguments()
    run_system(args)


if __name__ == "__main__":
    main()
