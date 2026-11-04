# Real-Time Driver Drowsiness and Distraction Warning System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Security Vetted](https://img.shields.io/badge/Security-pip--audit%20Passed-brightgreen.svg)]()
[![Air-Gapped Local Inference](https://img.shields.io/badge/Privacy-100%25%20Air--Gapped-success.svg)]()

A high-performance research prototype combining rule-based computer vision (EAR, MAR, PERCLOS, `solvePnP` head pose) with deep learning (MobileNetV2 transfer learning), fused through a debounced finite state machine with multi-tier auditory alarms. Developed as a final-year B.Tech / AIML capstone project.

---

> [!CAUTION]
> ### MANDATORY SAFETY DISCLAIMER & ETHICAL USAGE
> * **Research Prototype Only:** This system is an academic research prototype and is **NOT** a certified automotive safety device. Never rely on this software for vehicle driving safety.
> * **Zero Vehicle Control:** The software operates strictly as an advisory monitoring interface. It **never** interfaces with, connects to, or sends actuation commands to any vehicle, CAN bus, steering, braking, or throttle system.
> * **Informed Consent:** By running this software, the user confirms they are the monitored individual or have obtained explicit informed consent from the subject.
> * **Privacy & Air-Gap:** All processing occurs locally in volatile RAM. No image frames, facial landmarks, or biometric tokens are transmitted over the network or saved to disk by default.

---

## 1. System Overview and Key Features

* **Multi-Modal Sensing:**
  * **Facial Landmarks:** 468 base 3D points + 10 refined iris contour landmarks via Google MediaPipe Face Mesh.
  * **Eye Aspect Ratio (EAR):** Standard 6-point Euclidean aspect ratio calculation per eye.
  * **Mouth Aspect Ratio (MAR):** Upper/lower lip geometric tracking for yawn detection ($\text{MAR} > 0.60$ for $\ge 1.0\text{ s}$).
  * **PERCLOS:** Time-aware continuous sliding window ($60.0\text{ s}$) computing percentage of eye closure.
  * **Blink vs. Microsleep:** Involuntary blinks ($< 0.40\text{ s}$) are filtered; prolonged closures ($\ge 1.0\text{ s}$) flag microsleep.
  * **Head Pose Estimation:** 3D Euler angles (yaw, pitch, roll) via `cv2.solvePnP` against an anthropometric generic face model.
  * **Deep Learning Verification:** Lightweight MobileNetV2 ($128 \times 128$) for eye-state classification, fused via weighted probability ensemble.
* **Debounced Finite State Machine:**
  * States: `ALERT`, `WARNING`, `DROWSY`, `DISTRACTED`, `NO_FACE`.
  * Deterministic priority: $\text{NO\_FACE} > \text{DROWSY} > \text{DISTRACTED} > \text{WARNING} > \text{ALERT}$.
  * Configurable debounce windows ($1.5\text{ s}$ drowsiness, $2.0\text{ s}$ distraction) eliminate false alarms from natural blinks and shoulder-checking.
* **Auditory Alarms & Safeguards:**
  * Multi-frequency acoustic tones (soft chime for WARNING, urgent buzzer for DROWSY/DISTRACTED).
  * Software-enforced volume cap ($0.70$) and continuous alarm duration limit ($10.0\text{ s}$) with a $5.0\text{ s}$ safety pause.

---

## 2. Project Architecture

```
drowsiness_system/
├── config/
│   ├── default.yaml          # All operational thresholds, durations, and device parameters
│   └── calibration.json      # Dynamic baseline EAR calibration overrides (generated)
├── data/                     # Strictly git-ignored datasets
│   └── splits.json           # Subject-wise partition registry
├── docs/
│   ├── datasets.md           # Dataset licenses, academic terms, and download guidelines
│   ├── architecture.md       # Layered decomposition, data flow, and threading model
│   └── safety.md             # Safety protocols, air-gap guarantees, and ethics notices
├── logs/                     # Git-ignored session metric logs (.csv)
├── models/
│   ├── mobilenetv2_eye_state.pt    # Serialized PyTorch state_dict (weights_only=True)
│   └── mobilenetv2_eye_state.json  # Architecture, checksum, and class metadata
├── report/
│   ├── figures/              # Generated ROC curves, confusion matrices, and ablation charts
│   └── results.md            # Benchmark tables, ablation findings, and bias limitations
├── src/
│   ├── __init__.py
│   ├── config.py             # YAML loader with strict domain and boundary validation
│   ├── camera.py             # Fault-tolerant OpenCV capture with dropped-frame handling
│   ├── landmarks.py          # MediaPipe Face Mesh wrapper (refine_landmarks=True)
│   ├── features.py           # Pure mathematical functions: EAR, MAR, blink classifier
│   ├── perclos.py            # Time-aware 60s sliding window circular buffer
│   ├── head_pose.py          # Perspective-n-Point 3D head pose estimator
│   ├── cnn_model.py          # MobileNetV2 architecture with custom classifier head
│   ├── inference.py          # Normalized eye bounding-box cropping and batched inference
│   ├── fusion.py             # Pure debounced FSM with priority resolution
│   ├── alarm.py              # Audio engine with cooldowns and volume capping
│   ├── calibration.py        # 10-second open-eye baseline acquisition routine
│   ├── logger.py             # Sanitized CSV session logging
│   ├── overlay.py            # HUD graphics with telemetry and persistent disclaimer
│   └── main.py               # Application entry point and runtime orchestrator
├── scripts/
│   ├── prepare_data.py       # Subject-wise dataset partitioning and synthetic generator
│   ├── train.py              # Transfer learning pipeline with checkpointing
│   └── evaluate.py           # Evaluation suite, ROC-AUC, confusion matrices, ablations
├── tests/
│   ├── test_features.py      # Geometric feature verification on synthetic meshes
│   ├── test_perclos.py       # Sliding window expiration and variable delta-t tests
│   ├── test_fusion.py        # FSM state transition, debouncing, and priority tests
│   ├── test_alarm.py         # Audio cooldown, duration, and volume cap tests
│   ├── test_config.py        # Schema boundary and out-of-range validation tests
│   ├── test_head_pose.py     # 3D pose estimation and distraction boundary tests
│   ├── test_calibration.py   # Baseline EAR calibration and outlier rejection tests
│   └── test_no_network.py    # Strict socket monkey-patch air-gap verification
├── .gitignore
├── requirements.txt          # Security-vetted pinned dependencies
└── README.md
```

---

## 3. Installation and Setup

### 3.1 Prerequisites
* Python 3.10 or 3.11
* Web camera (or use `--synthetic-camera` for headless testing)
* Audio output device (laptop speakers or headphones)

### 3.2 Setup (Windows)
```powershell
# 1. Clone the repository
git clone https://github.com/example/drowsiness_system.git
cd drowsiness_system

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate environment
.venv\Scripts\Activate.ps1

# 4. Install security-vetted dependencies
pip install -r requirements.txt

# 5. Verify security posture
pip-audit -r requirements.txt
```

### 3.3 Setup (Linux / macOS)
```bash
# 1. Clone repository
git clone https://github.com/example/drowsiness_system.git
cd drowsiness_system

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify security posture
pip-audit -r requirements.txt
```

---

## 4. Quick Start and Operational Guide

### 4.1 Running the Rule-Based Demo (15-Minute Out-of-the-Box Quickstart)
Launch the application immediately using pure computer vision (EAR + MAR + PERCLOS + Head Pose) without downloading deep learning weights:
```bash
python -m src.main --rule-based-only
```

### 4.2 Full Hybrid Pipeline Execution (Rule-Based + MobileNetV2)
```bash
python -m src.main
```

### 4.3 Runtime Controls
* **`q` or `ESC`:** Exit the application cleanly.
* **`m`:** Toggle audio mute.
* **`c`:** Trigger manual 10-second open-eye baseline recalibration.

### 4.4 Headless Benchmarking Mode
To run on a remote server or in CI/CD without opening an X11/GUI display:
```bash
python -m src.main --no-display --synthetic-camera
```

---

## 5. Model Training and Scientific Evaluation

### 5.1 Dataset Preparation (Subject-Wise Partitioning)
Generate deterministic subject-wise partitions (`70% train`, `15% val`, `15% test`):
```bash
python scripts/prepare_data.py --synthetic
```

### 5.2 Model Training (Transfer Learning)
Train the MobileNetV2 classifier head (Phase 1) followed by fine-tuning top residual blocks (Phase 2):
```bash
python scripts/train.py --epochs 4 --batch-size 16 --output-model models/mobilenetv2_eye_state.pt
```

### 5.3 Comparative Evaluation and Ablation Suite
Run the full comparative benchmark across the three variants and ablations, saving figures to `report/figures/`:
```bash
python scripts/evaluate.py
```

---

## 6. Test Suite and Verification

Run the complete test suite including unit tests and the zero-network air-gap test:
```bash
pytest tests/ -v
```

---

## 7. Security and Vulnerability Review

All dependencies are pinned to exact versions in `requirements.txt` and audited against the Python Packaging Advisory Database using `pip-audit`.

* **PyTorch Weights Safety:** All checkpoint deserialization strictly enforces `torch.load(..., weights_only=True)`. Standard pickle deserialization is prohibited.
* **Air-Gap Verification:** `tests/test_no_network.py` patches standard socket calls and verifies zero outbound socket activity during inference.
* **Audit Command:**
  ```bash
  pip-audit -r requirements.txt
  ```

---

## 8. Troubleshooting

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| `Could not open camera device` | Camera index mismatch or OS permission denied | Specify `--config` with a different `camera_index` (e.g., `1` for external USB camera), or test with `--synthetic-camera`. Ensure webcam privacy permissions are enabled in Windows Settings. |
| No audio during alarm | Missing audio device or system muted | Press `m` in GUI window to unmute. On Linux, ensure `alsa-utils` or `pulseaudio` is active. |
| MediaPipe landmark failure | Unsupported Python version or GPU driver conflict | Use Python 3.10.11 or 3.11.x. Confirm `mediapipe==0.10.14`. |
| Low FPS ($< 15\text{ FPS}$) | CPU thermal throttling or unoptimized inference interval | In `config/default.yaml`, increase `cnn_inference_interval` to `3` or `4` to run CNN inference less frequently on CPU while preserving rule-based EAR tracking on every frame. |

