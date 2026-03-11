"""
config_hardware — Hardware-Tuned Runtime Parameters for the Target Acquisition Pipeline.

This module centralises every tunable constant that depends on the host hardware profile.
Separating these values from the application logic allows the same codebase to be redeployed
on different hardware configurations by editing a single file.

System under test
-----------------
* CPU  : AMD Ryzen 9 9600X (6 cores / 12 threads @ up to 5.4 GHz boost)
* RAM  : 32 GB DDR5
* GPU  : NVIDIA RTX 5070 (12 GB GDDR7 VRAM, Ada Lovelace architecture)

Parameter categories
--------------------
1. **GPU / Inference** — Controls the YOLOv10 inference engine (TensorRT, precision, thresholds).
2. **ROI Geometry** — Defines the capture window position on the physical display.
3. **RAM Buffers** — Queue depths for the asynchronous capture/detection pipeline.
4. **CPU Threading** — Thread counts for OpenCV and capture workers.
5. **Motion Smoothing** — Bézier G2 micro-step budget and Ease-Out tuning knobs.
6. **Advanced CUDA** — Low-level cuDNN and matmul flags for maximum throughput.
"""

# =============================================================================
# GPU / INFERENCE (RTX 5070)
# =============================================================================

# Enable TensorRT for 2–3× inference speedup (requires tensorrt Python package)
USE_TENSORRT = True

# TensorRT VRAM workspace allocation in GB (out of 12 GB available)
TENSORRT_WORKSPACE = 4

# YOLOv10 confidence threshold — lower values detect more candidates at the cost
# of false-positive rate.  0.25 is a conservative baseline for benchmarking.
YOLO_CONFIDENCE = 0.25

# Maximum bounding-box detections retained per frame (limits NMS post-processing cost)
YOLO_MAX_DETECTIONS = 10

# Vertical pixel mask: detections whose top edge is above this row are discarded.
# Set to 0 to disable (no upper boundary filter).
MIN_Y_THRESHOLD = 0

# =============================================================================
# ROI GEOMETRY (Multi-monitor setup)
# =============================================================================

# Top-left corner of the capture ROI in root-window pixel coordinates.
# Calibrated with calibrate_roi.py for the specific monitor layout.
GAME_WINDOW_X = 1074
GAME_WINDOW_Y = 1420

# Sub-pixel crosshair offset to correct residual sighting bias (pixels in ROI space).
# Positive values shift right / down; negative values shift left / up.
AIM_OFFSET_X = -2
AIM_OFFSET_Y = 1

# =============================================================================
# RAM BUFFERS (32 GB available)
# =============================================================================

# Inter-thread queue depths for the async capture → inference pipeline.
# Larger values reduce frame drops at the cost of added latency (queue fill time).
FRAME_QUEUE_SIZE = 3
DETECTION_QUEUE_SIZE = 2

# =============================================================================
# CPU THREADING (Ryzen 9 9600X — 12 logical cores)
# =============================================================================

# Worker thread counts for capture and OpenCV processing stages
NUM_CAPTURE_THREADS = 2
NUM_OPENCV_THREADS = 4

# =============================================================================
# MOTION SMOOTHING — Bézier G2 Biomechanical Algorithm
# =============================================================================
# The movement controller in main.py subdivides each target delta into
# BEZIER_STEPS micro-steps using a quadratic Bézier curve with G2 curvature
# continuity.  The parametric position at step i is:
#
#   B(t) = 2(1−t)·t·P1 + t²·P2,   t = i / BEZIER_STEPS  (linear time parameter)
#
# where P1 is the inertia carry-over control point (previous velocity × 0.4)
# and P2 is the target displacement vector.
# BezierMover.py provides an alternative implementation that applies a √t
# Ease-Out time-warp; see that module's docstring for the mathematical details.

# Number of micro-steps per detection frame (higher = smoother, more uinput calls)
BEZIER_STEPS = 16

# Kalman predictor look-ahead (frames) when the detector loses a tracked target
KALMAN_PREDICTION_FRAMES = 5

# Adaptive smoothing base factor — higher = slower approach, lower = more reactive
SMOOTH_FACTOR = 1.6

# Sensitivity multiplier — scales the raw pixel delta before Bézier interpolation
SENS_MULTIPLIER = 2.2

# Inter-micro-step sleep interval (seconds).  Reducing this saturates the mouse
# polling bus; 50 µs is appropriate for 1000 Hz polling-rate devices.
MICRO_MOVEMENT_DELAY = 0.00005

# =============================================================================
# ADVANCED CUDA FLAGS
# =============================================================================

# Enable Mixed Precision (FP16) inference on Tensor Core hardware
USE_MIXED_PRECISION = True

# cuDNN auto-tuner: finds the fastest convolution algorithm for the fixed input size
CUDA_BENCHMARK = True

# Allow TF32 on matmul and cuDNN ops (Ampere/Ada only) — near-FP32 accuracy at FP16 speed
CUDA_TF32 = True

# Number of warm-up inference passes to compile CUDA kernels before live operation
WARMUP_ITERATIONS = 10

print("""
╔════════════════════════════════════════════════════════════╗
║  HARDWARE-OPTIMISED CONFIGURATION LOADED                   ║
║  • RTX 5070 (12 GB) : TensorRT + Mixed Precision (FP16)   ║
║  • 32 GB RAM        : Enlarged async buffers (3×)         ║
║  • 9600X (12T)      : Multi-threaded capture pipeline     ║
║                                                            ║
║  Expected throughput gain vs. baseline: 2–4×              ║
╚════════════════════════════════════════════════════════════╝
""")
