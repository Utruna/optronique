import cv2
import keyboard
import time
import math
import threading
import numpy as np
import re
from pathlib import Path

from VisionSystem import VisionSystem
from DecisionEngine import DecisionEngine
from InputHandler import InputHandler
from KalmanFilterMouse import KalmanFilterMouse
import config_hardware as cfg  # Hardware tuning parameters

# --- G2 Curvature-Continuous Bézier curve generator ---
class BezierGenerator:
    def __init__(self):
        # Carry-over velocity from the previous curve for G2 continuity
        self.prev_vx = 0
        self.prev_vy = 0

    def generate_curve(self, target_dx, target_dy, steps):
        """Decompose a displacement into micro-steps via a quadratic Bézier curve.

        The control point P1 is placed in the direction of the previous movement
        vector scaled by an inertia factor (0.4), producing smooth G2 curvature
        continuity between successive motion segments.

        Parameters
        ----------
        target_dx : float
            Target displacement along the X axis (pixels).
        target_dy : float
            Target displacement along the Y axis (pixels).
        steps : int
            Number of micro-steps to generate.

        Returns
        -------
        list of tuple[float, float]
            Ordered list of (step_x, step_y) incremental displacements whose sum
            equals (target_dx, target_dy).
        """
        # Control point in the direction of the previous velocity (inertia = 0.4)
        p1_x = self.prev_vx * 0.4
        p1_y = self.prev_vy * 0.4

        micro_movements = []
        last_x, last_y = 0, 0

        for i in range(1, steps + 1):
            t = i / steps  # t ∈ (0, 1]

            # Quadratic Bézier: B(t) = 2(1−t)·t·P1 + t²·P2  (P0 at origin)
            bx = 2 * (1 - t) * t * p1_x + (t**2) * target_dx
            by = 2 * (1 - t) * t * p1_y + (t**2) * target_dy

            # Incremental step = current position minus last accumulated position
            step_x = bx - last_x
            step_y = by - last_y

            micro_movements.append((step_x, step_y))
            last_x, last_y = bx, by

        # Update carry-over velocity for the next curve segment (G2 continuity)
        self.prev_vx = target_dx
        self.prev_vy = target_dy

        return micro_movements

# --- Main acquisition loop ---
def main_live():
    # --- Baseline motion parameters ---
    SMOOTH_FACTOR = cfg.SMOOTH_FACTOR
    SENS_MULTIPLIER = cfg.SENS_MULTIPLIER
    TRIGGER_DISTANCE = 12
    COOLDOWN_TIR = 0.15
    HEAD_OFFSET_PCT = 0.32

    BEZIER_STEPS = cfg.BEZIER_STEPS
    MICRO_DELAY = cfg.MICRO_MOVEMENT_DELAY

    # 1. System initialisation
    vision = VisionSystem(
        fov_size=416,
        use_tensorrt=cfg.USE_TENSORRT,
        game_window_x=cfg.GAME_WINDOW_X,
        game_window_y=cfg.GAME_WINDOW_Y,
        conf=cfg.YOLO_CONFIDENCE,
        min_y=cfg.MIN_Y_THRESHOLD
    )

    engine = DecisionEngine()
    mouse = InputHandler()
    bezier = BezierGenerator()
    kalman = KalmanFilterMouse()

    remainder_x = 0
    remainder_y = 0

    aim_offset_x = cfg.AIM_OFFSET_X
    aim_offset_y = cfg.AIM_OFFSET_Y
    offset_step = 1
    offset_mode = False
    config_path = Path(__file__).resolve().parent / "config_hardware.py"

    tracking_active = False
    show_debug = False
    last_trigger = 0
    last_valid_target = None

    center = vision.fov_size // 2
    virtual_center = (center, center)

    print("🚀 AUTONOMOUS SIGHTING SYSTEM — X11/uinput Pipeline (RTX 5070 optimised)")
    print(f"⚙️ Config: Smooth={SMOOTH_FACTOR} | Steps={BEZIER_STEPS} | Sens={SENS_MULTIPLIER}")
    print(f"🔥 TensorRT: {'ENABLED' if cfg.USE_TENSORRT else 'DISABLED'}")
    print("[HOME] ON/OFF | [END] Quit | [PAGE DOWN] Debug")
    print("[F9] Offset mode | [F10] Save offsets")

    try:
        while True:
            # Hotkey handling
            if keyboard.is_pressed('end'): break

            if keyboard.is_pressed('home'):
                tracking_active = not tracking_active
                # Reset inertia carry-over to prevent a visual jump on re-enable
                bezier.prev_vx, bezier.prev_vy = 0, 0
                print(f"📡 Tracking: {'ACTIVE' if tracking_active else 'PAUSED'}")
                time.sleep(0.3)

            if keyboard.is_pressed('page down'):
                show_debug = not show_debug
                if not show_debug: cv2.destroyAllWindows()
                print(f"📺 Debug: {'ON' if show_debug else 'OFF'}")
                time.sleep(0.3)

            if keyboard.is_pressed('f9'):
                offset_mode = not offset_mode
                print(f"🎯 Offset mode: {'ON' if offset_mode else 'OFF'} | X={aim_offset_x} Y={aim_offset_y}")
                time.sleep(0.3)

            if offset_mode:
                if keyboard.is_pressed('left'):
                    aim_offset_x -= offset_step
                    print(f"🎯 Offset X={aim_offset_x} Y={aim_offset_y}")
                    time.sleep(0.05)
                elif keyboard.is_pressed('right'):
                    aim_offset_x += offset_step
                    print(f"🎯 Offset X={aim_offset_x} Y={aim_offset_y}")
                    time.sleep(0.05)
                elif keyboard.is_pressed('up'):
                    aim_offset_y -= offset_step
                    print(f"🎯 Offset X={aim_offset_x} Y={aim_offset_y}")
                    time.sleep(0.05)
                elif keyboard.is_pressed('down'):
                    aim_offset_y += offset_step
                    print(f"🎯 Offset X={aim_offset_x} Y={aim_offset_y}")
                    time.sleep(0.05)

                if keyboard.is_pressed('f10'):
                    try:
                        content = config_path.read_text(encoding='utf-8')
                        content = re.sub(r"AIM_OFFSET_X\s*=\s*-?\d+", f"AIM_OFFSET_X = {aim_offset_x}", content)
                        content = re.sub(r"AIM_OFFSET_Y\s*=\s*-?\d+", f"AIM_OFFSET_Y = {aim_offset_y}", content)
                        config_path.write_text(content, encoding='utf-8')
                        print("✅ Offsets saved to config_hardware.py")
                    except Exception as e:
                        print(f"⚠️ Failed to save offsets: {e}")
                    time.sleep(0.3)

            # 2. Capture + detection
            detections, frame = vision.detect_targets()

            if tracking_active:
                target = None

                if detections:
                    target = engine.choose_target(detections, virtual_center)
                    if target:
                        # Kalman update (retained for future predictor re-integration)
                        kalman.update(np.array([target['x'], target['y']]))
                        last_valid_target = target
                # Kalman prediction when YOLO loses the target is intentionally disabled
                # (caused erratic cursor rotation in earlier versions — re-enable when tuned)

                if target:
                    # --- Head-offset target computation ---
                    dx = (target['x'] + aim_offset_x) - center
                    offset_hauteur = target['h'] * HEAD_OFFSET_PCT
                    dy = (target['y'] + aim_offset_y - offset_hauteur) - center
                    dist = math.hypot(dx, dy)

                    # --- Adaptive Biomechanical Smoothing ---

                    # 1. Distance-proportional damping
                    #    ratio = 0 when target is very close, 1 when ≥ 100 px away
                    distance_ratio = min(dist / 100, 1.0)
                    # Less damping at close range for fast snap acquisition
                    adaptive_smooth = SMOOTH_FACTOR + (1.5 * (1 - distance_ratio))

                    target_move_x = (dx * SENS_MULTIPLIER) / adaptive_smooth
                    target_move_y = (dy * SENS_MULTIPLIER) / adaptive_smooth

                    # 2. G2 Bézier micro-step generation
                    curve_steps = bezier.generate_curve(target_move_x, target_move_y, steps=BEZIER_STEPS)

                    # 3. Execute micro-steps with sub-pixel remainder accumulation
                    for step_x, step_y in curve_steps:

                        # Accumulate fractional pixel remainders for long-term precision
                        total_x = step_x + remainder_x
                        total_y = step_y + remainder_y

                        move_x_int = int(total_x)
                        move_y_int = int(total_y)

                        remainder_x = total_x - move_x_int
                        remainder_y = total_y - move_y_int

                        if move_x_int != 0 or move_y_int != 0:
                            mouse.move_mouse(move_x_int, move_y_int)

                        # Inter-step pause to match mouse polling rate
                        time.sleep(MICRO_DELAY)

                    # --- Trigger logic ---
                    if dist < TRIGGER_DISTANCE:
                        now = time.time()
                        if now - last_trigger > COOLDOWN_TIR:
                            mouse.click()
                            last_trigger = now

                    # Debug overlay
                    if show_debug and frame is not None:
                        head_x, head_y = int(target['x']), int(target['y'] - offset_hauteur)
                        adj_head_x = head_x + aim_offset_x
                        adj_head_y = head_y + aim_offset_y
                        cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), 2)
                        cv2.circle(frame, (adj_head_x, adj_head_y), 5, (0, 255, 255), 2)
                        cv2.line(frame, virtual_center, (adj_head_x, adj_head_y), (0, 255, 255), 1)
                        cv2.putText(
                            frame,
                            f"Offset X={aim_offset_x} Y={aim_offset_y}",
                            (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1
                        )

            # 3. Debug window rendering
            if show_debug and frame is not None:
                cv2.drawMarker(frame, virtual_center, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                offset_center = (center + aim_offset_x, center + aim_offset_y)
                cv2.drawMarker(frame, offset_center, (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
                cv2.putText(
                    frame,
                    f"Offset X={aim_offset_x} Y={aim_offset_y}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
                cv2.imshow("Vision Debug", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    show_debug = False

    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        if hasattr(vision, 'stop'): vision.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_live()
