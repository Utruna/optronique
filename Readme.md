# 🔭 Visée Optronique — Autonomous Target Acquisition Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLO](https://img.shields.io/badge/AI-YOLOv10-orange)
![Platform](https://img.shields.io/badge/Platform-Linux%20Pop!_OS-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**Visée Optronique** is an experimental real-time computer vision framework designed to demonstrate sub-10 ms
end-to-end latency for object detection and closed-loop pointer control on Linux.

Built natively for the **Linux (X11)** display stack, the project pushes the boundaries of the
detection–action loop by coupling neural inference (YOLOv10 / ONNX) with **kernel-level input
injection** (`uinput`), bypassing every user-space abstraction layer that would otherwise introduce
latency.

> **⚠️ Ethical & Legal Disclaimer:** This project is released **strictly for educational and research
> purposes**. It serves as a technical demonstration of modern object-detection models, real-time
> systems programming, and biomechanical motion-smoothing algorithms (Bézier G2 curvature continuity).
> The author does not endorse or condone the use of this software in any context where it would violate
> the terms of service of a third-party platform, alter the experience of other users, or breach
> applicable law. All benchmarks and demonstrations were conducted in isolated, single-player or
> controlled environments.

---

## ⚡ Key Technical Features

* **Neural Vision Engine**: **YOLOv10** optimised for GPU inference — native PyTorch and ONNX export
  paths delivering **200+ FPS** on modern NVIDIA hardware.
* **High-Frequency Capture Pipeline**: Direct framebuffer access via **X11/Xlib** on a calibrated
  *Region of Interest* (ROI), achieving sub-millisecond capture latency without a compositing
  penalty.
* **Biomechanical Smoothing Algorithms**:
  * Adaptive G2 Curvature Continuity (quadratic Bézier with momentum carry-over).
  * Non-linear Ease-Out acceleration (`√t` time-warping function).
  * High-frequency micro-stepping to saturate mouse polling rates (up to 1000 Hz).
* **Kernel-Level Event Injection**: Hardware-emulated pointer events via the Linux `uinput` kernel
  module, eliminating user-space overhead and software-detectable artefacts.

## 🛠️ System Architecture

The system operates as a **closed feedback loop**:

1. **Acquisition** — Raw framebuffer capture via Xorg/X11 (`XGetImage`).
2. **Inference** — Convolutional Neural Network (YOLOv10) object detection on the ROI.
3. **Decision** — Euclidean nearest-target selection and dynamic head-offset compensation.
4. **Action** — Delta computation → G2 Bézier micro-step generation → `uinput` kernel injection.

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌─────────────┐
│  X11 Capture │ →  │  YOLOv10     │ →  │ DecisionEngine │ →  │   uinput    │
│  (XGetImage) │    │  Inference   │    │  + Kalman      │    │  REL_X/Y    │
└─────────────┘    └──────────────┘    └────────────────┘    └─────────────┘
       ↑___________________________ Closed feedback loop _____________________|
```

## 📋 Requirements

* **OS**: Linux (tested on Pop!_OS 22.04 LTS).
* **Display server**: **X11 / Xorg** (Wayland is not supported for high-performance framebuffer
  capture).
* **GPU**: NVIDIA recommended (CUDA 11.8+ supported).
* **Dependencies**: Python 3.10+, `python-xlib`, `uinput`, `ultralytics`, `opencv-python`, `numpy`,
  `onnxruntime-gpu`, `onnxslim`.

## 📸 Technical Preview

| Multi-Target Detection | Trajectory Computation (Vector) |
|:---:|:---:|
| ![YOLOv10 Detection](./Img_readme/v10_capture_8.jpg) | ![Bézier sighting vector](./Img_readme/sim_20260202141617_1.jpg) |
| *Real-time segmentation demo on a live 416×416 ROI.* | *Non-linear Bézier trajectory for biomechanical pointer smoothing.* |

## 📺 Demo

<div align="center">
  <p><em>Debug mode visualisation — Bézier G2 trajectory and dynamic Head Offset.</em></p>
  <img src="Img_readme/Gif_fonctionnement.gif" alt="Autonomous Sighting System Demo" width="100%">
</div>

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Utruna/optronique.git
   cd optronique
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **YOLO weights & ONNX optimisation (recommended for 200+ FPS)**
   - The model is loaded from `yolov10n.pt` by default (project root).
   - **For maximum throughput**, export to ONNX with graph optimisation:
     ```bash
     yolo export model=yolov10n.pt format=onnx imgsz=416 half=True device=0
     onnxslim yolov10n.onnx yolov10n_opt.onnx
     ```
   - When using ONNX, update the `model_path` argument in `VisionSystem.py` to point to
     `yolov10n_opt.onnx`.

4. **Run**
   - Interactive launch (requires `sudo` if `/dev/uinput` is not accessible to the current user):
     ```bash
     sudo .venv/bin/python main.py
     ```

## 🎛️ Runtime Controls & Calibration

| Key | Action |
|---|---|
| **HOME** | Toggle target acquisition on / off |
| **PAGE DOWN** | Toggle OpenCV debug overlay |
| **F9** | Enter / exit crosshair offset adjustment mode |
| **F10** | Persist current offsets to `config_hardware.py` |
| **END** | Exit the application |

**Calibration utilities:**
- `calibrate_roi.py` — Interactively position the ROI window and save `GAME_WINDOW_X` / `GAME_WINDOW_Y`.
- `test_capture.py` — Capture a single frame and write `capture_test.jpg` for visual verification.
- `test_yolo_direct2.py` — Run YOLOv10 inference on a captured image for offline benchmarking.

## ⚙️ Configuration

All hardware tuning parameters live in `config_hardware.py`:
- Inference thresholds: `YOLO_CONFIDENCE`, `YOLO_MAX_DETECTIONS`
- Motion smoothing: `BEZIER_STEPS`, `SMOOTH_FACTOR`, `SENS_MULTIPLIER`, `MICRO_MOVEMENT_DELAY`
- Persistent sighting offsets: `AIM_OFFSET_X`, `AIM_OFFSET_Y` (read at startup, adjustable via F9/F10)

## 🧪 Tests

Unit tests live in `test_unitaire/`. Run with:
```bash
pytest test_unitaire/
# or
python -m unittest discover test_unitaire
```

## Technical Notes

- `VisionSystem.py` runs in **synchronous mode** (capture + inference in the main loop) to guarantee
  every frame is processed by YOLOv10 at runtime without inter-thread race conditions.
- If you encounter an unpickling error on model load, the loader forces `weights_only=False` to ensure
  compatibility with all PyTorch/Ultralytics version combinations.
