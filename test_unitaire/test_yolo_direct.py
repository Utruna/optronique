#!/usr/bin/env python3
"""Debug: Teste YOLO directement sur une frame de test"""

import numpy as np
from ultralytics import YOLO
import torch
import cv2

# Charger l'image de test
img = cv2.imread('/home/colin/Documents/Projet/aimbot_yolo/capture_test.jpg')
if img is None:
    print("❌ capture_test.jpg not found!")
    exit(1)

print(f"Image shape: {img.shape}")

# Charger YOLO
model = YOLO('yolov10n.pt')
if torch.cuda.is_available():
    model.to('cuda')
    print("✅ YOLO sur GPU")

# Test direct
print("🧪 Test YOLO sur capture_test.jpg...")
results = model.predict(
    img,
    conf=0.25,
    classes=[0],
    device=0 if torch.cuda.is_available() else 'cpu',
    half=True if torch.cuda.is_available() else False,
    verbose=False
)

# Parse
detections = []
if len(results) > 0 and len(results[0].boxes) > 0:
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            'x': int((x1 + x2) / 2),
            'y': int((y1 + y2) / 2),
            'conf': float(box.conf)
        })
    print(f"✅ {len(detections)} détections trouvées!")
    for d in detections:
        print(f"   → Position: ({d['x']}, {d['y']}) | Confiance: {d['conf']:.2f}")
else:
    print("❌ Aucune détection (même à conf=0.25)")
