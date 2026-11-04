# System Architecture and Data Flow

## 1. Layered Decomposition

```
+─────────────────────────────────────────────────────────────+
|                    PRESENTATION & I/O                       |
|   camera.py  •  overlay.py  •  alarm.py  •  logger.py       |
+─────────────────────────────────────────────────────────────+
                              ▲
                              │
+─────────────────────────────────────────────────────────────+
|                    DECISION & FUSION                        |
|   fusion.py (Debounced FSM)  •  calibration.py              |
+─────────────────────────────────────────────────────────────+
                              ▲
                              │
+─────────────────────────────────────────────────────────────+
|                    DOMAIN LOGIC & VISION                    |
|   features.py (EAR, MAR)  •  perclos.py  •  head_pose.py    |
+─────────────────────────────────────────────────────────────+
                              ▲
                              │
+─────────────────────────────────────────────────────────────+
|                    MACHINE LEARNING                         |
|   cnn_model.py (MobileNetV2)  •  inference.py               |
+─────────────────────────────────────────────────────────────+
                              ▲
                              │
+─────────────────────────────────────────────────────────────+
|                    CONFIGURATION ENGINE                     |
|   config.py  •  default.yaml                                |
+─────────────────────────────────────────────────────────────+
```

---

## 2. Per-Frame Synchronous Data Flow Pipeline

```
[Raw Webcam Frame (640x480 BGR)]
       │
       ▼
[MediaPipe Face Mesh (refine_landmarks=True)]
       │
       ├── (No Face > 3.0s) ──────────────► [NO_FACE State] ──► [Continuous Alarm]
       │
       ▼ (478 3D Landmarks)
[Geometric Feature Extraction]
       ├── calculate_average_ear() ──► EAR Value
       ├── calculate_mar() ──────────► MAR Value
       └── estimate_head_pose() ────► Yaw, Pitch, Roll
       │
       ▼
[MobileNetV2 Eye-State Inference] (Every N=2 frames)
       └── crop_eye_region() ──► [128x128] ──► P(Closed)
       │
       ▼
[PERCLOS Tracker] (60.0s time-aware buffer)
       └── Integrates eye closure ratio over sliding window
       │
       ▼
[Debounced FSM (DrowsinessFSM)]
       ├── Evaluates candidate conditions and debounce persistence
       └── Resolves priority: NO_FACE > DROWSY > DISTRACTED > WARNING > ALERT
       │
       ▼
[System Dispatch]
       ├── AlarmManager: Updates tone, checks cooldown & continuous limit
       ├── SessionLogger: Appends CSV telemetry row
       └── Overlay Engine: Draws color-coded HUD, gauges, & persistent disclaimer
```

---

## 3. Threading Architecture
* **Main Thread:** Executes video capture, landmark extraction, geometric feature calculations, CNN inference, FSM state updates, and HUD rendering synchronously at $\approx 25\text{ FPS}$.
* **Audio Dispatch Daemon:** Audio waveform playback runs asynchronously on background worker threads managed by `pygame.mixer` or `winsound` (`SND_ASYNC`), ensuring sound buffer operations never cause frame drops in the video processing loop.

