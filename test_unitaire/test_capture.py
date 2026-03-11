#!/usr/bin/env python3
"""Diagnostic utility: Display what the ROI capture sees."""

import Xlib.display
import Xlib.X
import numpy as np
import cv2

display = Xlib.display.Display()
screen = display.screen(0)
target_drawable = screen.root

# Use the same coordinates as config_hardware.py
x, y = 1074, 1420
fov_size = 416

print(f"Capturing: X={x}, Y={y}, Size={fov_size}x{fov_size}")

raw_img = target_drawable.get_image(x, y, fov_size, fov_size, Xlib.X.ZPixmap, 0xffffffff)
frame = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(fov_size, fov_size, 4)
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# Save for manual inspection
cv2.imwrite('/home/colin/Documents/Projet/aimbot_yolo/capture_test.jpg', frame)
print("✅ Saved: capture_test.jpg")

# Frame statistics
print(f"Min: {frame.min()} | Max: {frame.max()} | Mean: {frame.mean():.1f}")
print(f"Shape: {frame.shape}")
