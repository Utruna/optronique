#!/usr/bin/env python3
"""
YOLOv10 Benchmark: PyTorch (.pt) vs ONNX (.onnx)
Compares real inference performance on the RTX 5070.
"""

import torch
import functools
from ultralytics import YOLO
import numpy as np
import time

torch.load = functools.partial(torch.load, weights_only=False)

def benchmark_model(model_path, num_iterations=50):
    """Benchmark a YOLOv10 model and return average latency and FPS.

    Parameters
    ----------
    model_path : str
        Path to the model file (.pt or .onnx).
    num_iterations : int
        Number of inference passes to average over.

    Returns
    -------
    tuple[float, float]
        (avg_latency_ms, fps)
    """
    print(f"\n🔍 Benchmarking: {model_path}")
    print("-" * 60)

    model = YOLO(model_path)
    if torch.cuda.is_available() and model_path.endswith('.pt'):
        model.to('cuda')

    # Create a synthetic input frame
    dummy_frame = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)

    # GPU warm-up (essential for accurate CUDA timing)
    print("🔥 Warming up GPU...")
    for _ in range(5):
        model.predict(dummy_frame, verbose=False, device=0, half=True, imgsz=416)

    # Timed benchmark
    print("⏱️  Running benchmark...")
    times = []

    for i in range(num_iterations):
        start = time.perf_counter()
        results = model.predict(
            dummy_frame,
            conf=0.30,
            classes=[0],
            imgsz=416,
            device=0,
            half=True,
            verbose=False,
            augment=False,
            agnostic_nms=True
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_iterations} iterations...")

    # Statistics
    times_ms = [t * 1000 for t in times]
    avg_time = np.mean(times_ms)
    min_time = np.min(times_ms)
    max_time = np.max(times_ms)
    std_time = np.std(times_ms)
    fps = 1000 / avg_time

    print("\n📊 Results:")
    print(f"  Average latency: {avg_time:.2f} ms ({fps:.1f} FPS)")
    print(f"  Minimum:         {min_time:.2f} ms")
    print(f"  Maximum:         {max_time:.2f} ms")
    print(f"  Std deviation:   {std_time:.2f} ms")

    return avg_time, fps


def main():
    print("=" * 60)
    print("🚀 YOLOv10 INFERENCE BENCHMARK — RTX 5070")
    print("=" * 60)

    # GPU information
    if torch.cuda.is_available():
        print(f"\n💻 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # PyTorch benchmark
    pt_time, pt_fps = benchmark_model("yolov10n.pt", num_iterations=50)

    # ONNX benchmark
    onnx_time, onnx_fps = benchmark_model("yolov10n.onnx", num_iterations=50)

    # Comparison table
    print("\n" + "=" * 60)
    print("📈 COMPARISON")
    print("=" * 60)
    print(f"\n{'Format':<15} {'Latency (ms)':<15} {'FPS':<15} {'Speedup'}")
    print("-" * 60)
    print(f"{'PyTorch (.pt)':<15} {pt_time:>10.2f} ms  {pt_fps:>10.1f} FPS  {1.0:.2f}×")
    print(f"{'ONNX (.onnx)':<15} {onnx_time:>10.2f} ms  {onnx_fps:>10.1f} FPS  {pt_time/onnx_time:.2f}×")

    speedup = pt_time / onnx_time
    if speedup > 1.5:
        print(f"\n🎉 ONNX is {speedup:.2f}× faster — substantial throughput gain!")
    elif speedup > 1.2:
        print(f"\n✅ ONNX is {speedup:.2f}× faster — meaningful gain.")
    else:
        print(f"\n⚠️  ONNX is only {speedup:.2f}× faster — marginal gain on this hardware.")

    print("\n💡 To use ONNX automatically:")
    print("   The system will detect yolov10n.onnx if present in the project root.")
    print("=" * 60)


if __name__ == "__main__":
    main()
