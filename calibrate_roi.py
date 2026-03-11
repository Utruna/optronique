#!/usr/bin/env python3
"""ROI Calibration utility: live preview window with keyboard-driven adjustment."""

import re
import time
from pathlib import Path

import cv2
import numpy as np
import Xlib.display
import Xlib.X

CONFIG_PATH = Path(__file__).resolve().parent / "config_hardware.py"


def read_int(prompt, default):
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
        return int(raw) if raw else default
    except Exception:
        return default


def read_config_values():
    if not CONFIG_PATH.exists():
        return None, None
    content = CONFIG_PATH.read_text(encoding="utf-8")
    mx = re.search(r"GAME_WINDOW_X\s*=\s*(\d+)", content)
    my = re.search(r"GAME_WINDOW_Y\s*=\s*(\d+)", content)
    x = int(mx.group(1)) if mx else None
    y = int(my.group(1)) if my else None
    return x, y


def save_config_values(left, top):
    if not CONFIG_PATH.exists():
        print("❌ config_hardware.py not found.")
        return False
    content = CONFIG_PATH.read_text(encoding="utf-8")
    content, n1 = re.subn(r"GAME_WINDOW_X\s*=\s*\d+", f"GAME_WINDOW_X = {left}", content)
    content, n2 = re.subn(r"GAME_WINDOW_Y\s*=\s*\d+", f"GAME_WINDOW_Y = {top}", content)
    if n1 == 0 or n2 == 0:
        print("⚠️ Keys not found in config. Add them manually.")
        return False
    CONFIG_PATH.write_text(content, encoding="utf-8")
    print("✅ config_hardware.py updated.")
    return True


def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


def main():
    print("=== ROI Calibration (live preview) ===")
    fov = read_int("FOV size (px)", 416)

    display = Xlib.display.Display()
    root = display.screen().root
    geo = root.get_geometry()
    screen_w, screen_h = geo.width, geo.height

    cfg_x, cfg_y = read_config_values()
    if cfg_x is None or cfg_y is None:
        cfg_x = (screen_w // 2) - (fov // 2)
        cfg_y = (screen_h // 2) - (fov // 2)

    left, top = cfg_x, cfg_y
    step = 5

    print("\nControls:")
    print("  Arrow keys / WASD : move ROI")
    print("  +/- : change step size (current: 5 px)")
    print("  O : save to config_hardware.py")
    print("  Q/Esc : quit")

    while True:
        left = clamp(left, 0, screen_w - fov)
        top = clamp(top, 0, screen_h - fov)

        raw_img = root.get_image(left, top, fov, fov, Xlib.X.ZPixmap, 0xFFFFFFFF)
        frame = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(fov, fov, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Crosshair at ROI centre
        cx, cy = fov // 2, fov // 2
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)

        info = f"X={left} Y={top} step={step}"
        cv2.rectangle(frame, (5, 5), (220, 30), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("ROI Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key in (81, ord("a")):  # left
            left -= step
        elif key in (83, ord("d")):  # right
            left += step
        elif key in (82, ord("w")):  # up
            top -= step
        elif key in (84, ord("s")):  # down
            top += step
        elif key in (ord("+"), ord("=")):
            step = min(step + 1, 50)
        elif key in (ord("-"), ord("_")):
            step = max(step - 1, 1)
        elif key in (ord("p"),):
            print(f"X={left} Y={top}")
        elif key in (ord("o"),):
            save_config_values(left, top)

        # Brief sleep to avoid saturating the event loop
        time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
