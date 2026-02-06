#!/usr/bin/env python3
"""Diagnostic: Affiche ce que capture la ROI"""

import Xlib.display
import Xlib.X
import numpy as np
import cv2

display = Xlib.display.Display()
screen = display.screen(0)
target_drawable = screen.root

# Utiliser les mêmes coordonnées
x, y = 1074, 1420
fov_size = 416

print(f"Capturant: X={x}, Y={y}, Size={fov_size}x{fov_size}")

raw_img = target_drawable.get_image(x, y, fov_size, fov_size, Xlib.X.ZPixmap, 0xffffffff)
frame = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(fov_size, fov_size, 4)
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# Sauvegarder pour inspection
cv2.imwrite('/home/colin/Documents/Projet/aimbot_yolo/capture_test.jpg', frame)
print("✅ Sauvegardé: capture_test.jpg")

# Afficher des stats
print(f"Min: {frame.min()} | Max: {frame.max()} | Moyenne: {frame.mean():.1f}")
print(f"Shape: {frame.shape}")
