"""
YOLOv10 TensorRT Optimisation Script for the RTX 5070.
Exports the base model to a TensorRT engine for maximum inference throughput.
Expected speedup over PyTorch baseline: 2–3×.
"""

import torch
import functools
from ultralytics import YOLO
import time

# Fix PyTorch 2.6+ weights loading
torch.load = functools.partial(torch.load, weights_only=False)

def optimize_model_for_rtx5070():
    """Export YOLOv10 to a TensorRT FP16 engine optimised for the RTX 5070 (12 GB VRAM).

    Returns
    -------
    str
        Path to the generated TensorRT engine file.
    """
    print("🔧 YOLO TensorRT OPTIMISATION — RTX 5070 (12 GB VRAM)")
    print("="*60)

    # Load the base PyTorch model
    model = YOLO("yolov10n.pt")

    # Export to TensorRT (NVIDIA-native engine format)
    print("\n1️⃣ Exporting to TensorRT engine...")
    model.export(
        format='engine',  # TensorRT engine
        imgsz=416,
        half=True,        # FP16 for RTX Tensor Cores
        device=0,
        workspace=4,      # 4 GB VRAM workspace budget
        simplify=True,
        batch=1
    )

    print("✅ Optimised model written: yolov10n.engine")

    # Throughput comparison
    print("\n2️⃣ Throughput benchmark...")
    model_original = YOLO("yolov10n.pt")
    model_optimized = YOLO("yolov10n.engine")

    import numpy as np
    dummy_frame = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)

    # Baseline (PyTorch)
    times_original = []
    for i in range(20):
        start = time.perf_counter()
        model_original.predict(dummy_frame, verbose=False, device=0, half=True)
        elapsed = time.perf_counter() - start
        if i >= 5:  # Skip warm-up passes
            times_original.append(elapsed)

    # TensorRT engine
    times_optimized = []
    for i in range(20):
        start = time.perf_counter()
        model_optimized.predict(dummy_frame, verbose=False)
        elapsed = time.perf_counter() - start
        if i >= 5:  # Skip warm-up passes
            times_optimized.append(elapsed)

    avg_original = np.mean(times_original) * 1000
    avg_optimized = np.mean(times_optimized) * 1000
    speedup = avg_original / avg_optimized

    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    print(f"Baseline PyTorch (.pt):     {avg_original:.2f} ms ({1000/avg_original:.1f} FPS)")
    print(f"TensorRT engine (.engine):  {avg_optimized:.2f} ms ({1000/avg_optimized:.1f} FPS)")
    print(f"🚀 Speedup: {speedup:.2f}× faster")
    print("="*60)

    return "yolov10n.engine"

if __name__ == "__main__":
    optimized_path = optimize_model_for_rtx5070()
    print(f"\n✅ To use the optimised engine: VisionSystem(model_path='{optimized_path}')")
