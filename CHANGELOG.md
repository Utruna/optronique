# Changelog

## [2.1.0] - 2026-02-06: Stabilisation & Calibration Utilities

### Fixes & Reliability
- VisionSystem: switched to **synchronous mode** (X11 capture + YOLOv10 inference in the main
  loop) to eliminate dropped frames during live operation.
- Fixed YOLO model loading: `torch.load` forced with `weights_only=False` to ensure compatibility
  across all PyTorch/Ultralytics version combinations.
- Added `calibrate_roi.py` utility to interactively position the ROI and persist `GAME_WINDOW_X`
  / `GAME_WINDOW_Y` to `config_hardware.py`.
- Added runtime crosshair-offset adjustment mode (F9 to activate, F10 to save offsets).

### Improvements
- Added `test_capture.py` and `test_yolo_direct2.py` for rapid standalone diagnostics (single-frame
  capture and offline inference benchmarking).
- Reorganised: unit tests moved to `test_unitaire/`.
- Motion tuning: adjusted `BEZIER_STEPS`, `SMOOTH_FACTOR`, `SENS_MULTIPLIER`, and `MICRO_MOVEMENT_DELAY`
  for improved reactivity/smoothness balance.

### Notes
- Keeping `VisionSystem.py` separate from the main loop simplifies GPU/X11 debugging.  Utility
  modules may be merged in a future consolidation pass.

---

## [2.0.0] - 2026-02-05: Biomechanical Smoothing Algorithm ("Human-Flow")

This major release replaces the linear interpolation engine with a full biomechanical motion
framework based on G2 curvature-continuous Bézier curves, producing pointer trajectories that
are indistinguishable from human fine-motor movement by heuristic analysis.

<div align="center">
  <img src="Img_readme/Gif_fonctionnement.gif" alt="Bézier G2 Demo" width="100%">
  <p><em>Debug mode: Bézier G2 trajectory with dynamic Head Offset compensation.</em></p>
</div>

### Major Features

#### G2 Curvature Continuity (Quadratic Bézier)
- Implemented quadratic Bézier curves for all pointer motion trajectories.
- **Dynamic inertia**: the cursor preserves the momentum vector of the previous motion segment,
  eliminating abrupt directional changes (robotic angular transitions) in favour of smooth,
  naturally curved paths.
- G2 continuity ensures that not only position (G0) and tangent direction (G1) are continuous
  across segment boundaries, but also curvature (G2) — removing jerk artefacts detectable by
  heuristic motion analysers.

##### Why G2 curvature continuity? (Biomimetics)

Most interpolation schemes use linear or first-order (G1) smoothing, which softens the path
but retains sudden changes in acceleration that are invisible to the eye yet detectable by
statistical motion classifiers.

This implementation uses **G2 (curvature continuity)** Bézier curves:

* **G0 (Position)**: the trajectory is spatially continuous (no teleportation).
* **G1 (Tangent)**: direction changes smoothly without angular discontinuities.
* **G2 (Curvature)**: acceleration itself is smoothed — no "jerk" at the start or end of each
  segment.

**Result**: the cursor exhibits simulated inertia that closely mirrors the fine-motor dynamics of
the human hand and wrist.

#### Non-Linear Ease-Out Acceleration
- Replaced the linear time parameter with a square-root warp (t → √t).
- **Behaviour**: fast initial snap toward the target followed by a micro-deceleration for precise
  final adjustment — replicating human fine-motor control.

#### High-Frequency Micro-Stepping
- Each motion instruction is decomposed into **16 interpolated micro-steps** (configurable via
  `BEZIER_STEPS`).
- Saturates the mouse polling rate (up to 1000 Hz) for absolute fluidity on high-refresh-rate
  displays (144 Hz+).

### Technical Changes
- **New class**: `BezierGenerator` manages G2-continuous quadratic curve generation.
- **Micro-Stepping 4×**: motion resolution increased to 4 micro-steps per detection frame,
  maximising smoothness on high-frequency displays.
- **Reactivity**: `SMOOTH_FACTOR = 2.0` — aggressive balance between stability and acquisition
  speed.
- **Sensitivity**: `SENS_MULTIPLIER = 1.65` — ensures full target coverage despite Bézier
  path damping.
- **Head Offset**: `HEAD_OFFSET_PCT = 0.42` — calibrated to the neck/chin region of the
  bounding box for optimal vertical compensation.
- **Sub-pixel accumulation**: fractional pixel remainders are accumulated across frames for
  mathematically precise long-range tracking.
