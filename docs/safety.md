# Safety, Privacy, and Ethics Specification

## 1. Safety of Use

### 1.1 Persistent On-Screen Disclaimer
Every rendered frame contains an un-dismissible, high-contrast disclaimer across the lower border:
```
RESEARCH PROTOTYPE. NOT A CERTIFIED SAFETY DEVICE. DO NOT RELY ON THIS SYSTEM FOR DRIVING SAFETY.
```

### 1.2 Non-Vehicular Actuation Mandate
> [!CAUTION]
> **Zero Vehicle Control Clause:**
> This software is an advisory research prototype designed solely for driver monitoring in academic settings. The system **never** interfaces with, connects to, or sends commands to any vehicle subsystem, CAN bus, OBD-II diagnostic port, steering actuator, braking system, or engine throttle. It must never be utilized as an active safety intervention mechanism.

### 1.3 Fail-Safe Architecture
If any critical subsystem encounters an unrecoverable fault—including webcam bus disconnection, repeated frame corruption, or landmark model initialization failure—the system enforces a fail-safe protocol:
1. Immediately dispatches an audible emergency fault tone ($1000\text{ Hz}$).
2. Renders a prominent visual fault badge (`CAMERA_FAULT` / `NO_FACE`).
3. Emits a critical diagnostic trace to standard error.
4. **Never fails silently.**

### 1.4 Acoustic Safeguards
* **Enforced Volume Cap:** Sound output is bounded to a maximum amplitude scalar of $0.70$ ($70\%$) in software to protect driver hearing.
* **Continuous Sounding Limit:** Continuous alarm buzzer duration is capped at $10.0\text{ s}$. If a hazard persists beyond $10.0\text{ s}$, the system enters an automatic $5.0\text{ s}$ acoustic pause before re-evaluating, avoiding sensory overload.
* **Alarm Cooldown:** Minimum $5.0\text{ s}$ silence interval enforced between repeating warnings for the same condition.

---

## 2. Privacy Guarantees

### 2.1 Complete Air-Gapped Local Execution
All computational workflows—video frame ingestion, facial landmark extraction, feature computation, deep learning inference, and FSM evaluation—run exclusively inside local volatile memory (RAM). Zero telemetry packets, biometric tokens, facial embeddings, or frame pixels are transmitted across any network interface. Enforced via automated unit testing (`tests/test_no_network.py`).

### 2.2 Ephemeral Video Processing
Raw video frames captured from the webcam are processed in-memory and immediately dereferenced for garbage collection. Raw camera video is **never written to disk** by default.

### 2.3 Opt-In Anonymized Debugging (`--save-frames`)
Saving frames to disk is strictly opt-in via the `--save-frames` CLI flag. When enabled, the system saves only tightly cropped, normalized eye patches ($128 \times 128$) for model auditing. Bounding boxes are restricted to the ocular region, eliminating all identifying facial morphology.

### 2.4 Sanitized Telemetry Logs
Session CSV audit logs record exclusively timestamps, discrete operational states, and scalar floating-point metrics (EAR, MAR, PERCLOS, angles, FPS). Zero identity, facial, or biometric credentials are stored.

### 2.5 Headless Operation
Researchers operating in headless environments (e.g. servers or benchmarking harnesses) can supply `--no-display` to suppress all graphical window rendering.

---

## 3. Ethics and Consent

### 3.1 Informed Consent Protocol
Upon startup, the CLI outputs an explicit consent and safety banner. By executing the system, the user confirms they are the monitored individual or have obtained explicit informed consent from the subject.

### 3.2 Absence of Biometric Identification
The pipeline is designed exclusively for state estimation (attentive vs. drowsy, forward-gaze vs. distracted). It does **not** perform facial recognition, does not match against identity databases, and does not retain facial embeddings.

### 3.3 Bias and Limitations Analysis
* **Skin Tone (Fitzpatrick Phototypes I–VI):** MediaPipe landmark detection confidence can degrade under low-lux conditions for darker skin tones. Mitigated by individual baseline EAR calibration and adaptive lighting recommendations.
* **Eyeglasses & Reflections:** Specular glints from prescription lenses can interfere with pupil iris tracking. The hybrid system relies on head pose and yawn indicators when ocular confidence is diminished.
* **Face Coverings & Masks:** Surgical masks occlude mouth landmarks (MAR). Documented that the system safely degrades to EAR and head-pose monitoring under partial lower-face occlusion.

