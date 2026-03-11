#!/usr/bin/env python3
"""Debug utility: Test YOLOv10 inference directly on a captured test frame (v2)."""

import numpy as np
import torch
import functools
import cv2

# Apply torch.load patch BEFORE importing ultralytics
torch.load = functools.partial(torch.load, weights_only=False)

from ultralytics import YOLO

# Load test image
img = cv2.imread('/home/colin/Documents/Projet/aimbot_yolo/capture_test.jpg')
if img is None:
    print("❌ capture_test.jpg not found!")
    exit(1)

print(f"Image shape: {img.shape}")

# Load YOLOv10
model = YOLO('yolov10n.pt')
if torch.cuda.is_available():
    model.to('cuda')
    print("✅ YOLOv10 on GPU")

# Run inference
print("🧪 Running YOLOv10 inference on capture_test.jpg...")
results = model.predict(
    img,
    conf=0.25,
    classes=[0],
    device=0 if torch.cuda.is_available() else 'cpu',
    half=True if torch.cuda.is_available() else False,
    verbose=False
)

# Parse detections
detections = []
if len(results) > 0 and len(results[0].boxes) > 0:
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            'x': int((x1 + x2) / 2),
            'y': int((y1 + y2) / 2),
            'conf': float(box.conf)
        })
    print(f"✅ {len(detections)} detection(s) found!")
    for d in detections:
        print(f"   → Position: ({d['x']}, {d['y']}) | Confidence: {d['conf']:.2f}")
else:
    print("❌ No detections (even at conf=0.25)")
