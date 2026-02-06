#!/usr/bin/env python3
"""
Benchmark YOLO: PyTorch (.pt) vs ONNX (.onnx)
Compare les performances réelles sur votre RTX 5070
"""

import torch
import functools
from ultralytics import YOLO
import numpy as np
import time

torch.load = functools.partial(torch.load, weights_only=False)

def benchmark_model(model_path, num_iterations=50):
    """Benchmark un modèle YOLO"""
    print(f"\n🔍 Benchmark: {model_path}")
    print("-" * 60)
    
    model = YOLO(model_path)
    if torch.cuda.is_available() and model_path.endswith('.pt'):
        model.to('cuda')
    
    # Créer une image dummy
    dummy_frame = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    
    # Warmup (important pour GPU)
    print("🔥 Warmup GPU...")
    for _ in range(5):
        model.predict(dummy_frame, verbose=False, device=0, half=True, imgsz=416)
    
    # Benchmark réel
    print("⏱️  Benchmark en cours...")
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
    
    # Statistiques
    times_ms = [t * 1000 for t in times]
    avg_time = np.mean(times_ms)
    min_time = np.min(times_ms)
    max_time = np.max(times_ms)
    std_time = np.std(times_ms)
    fps = 1000 / avg_time
    
    print("\n📊 Résultats:")
    print(f"  Temps moyen: {avg_time:.2f}ms ({fps:.1f} FPS)")
    print(f"  Temps min:   {min_time:.2f}ms")
    print(f"  Temps max:   {max_time:.2f}ms")
    print(f"  Écart-type:  {std_time:.2f}ms")
    
    return avg_time, fps


def main():
    print("=" * 60)
    print("🚀 BENCHMARK YOLO - RTX 5070")
    print("=" * 60)
    
    # GPU Info
    if torch.cuda.is_available():
        print(f"\n💻 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Benchmark PyTorch
    pt_time, pt_fps = benchmark_model("yolov10n.pt", num_iterations=50)
    
    # Benchmark ONNX
    onnx_time, onnx_fps = benchmark_model("yolov10n.onnx", num_iterations=50)
    
    # Comparaison
    print("\n" + "=" * 60)
    print("📈 COMPARAISON")
    print("=" * 60)
    print(f"\n{'Format':<15} {'Temps (ms)':<15} {'FPS':<15} {'Speedup'}")
    print("-" * 60)
    print(f"{'PyTorch (.pt)':<15} {pt_time:>10.2f} ms  {pt_fps:>10.1f} FPS  {1.0:.2f}x")
    print(f"{'ONNX (.onnx)':<15} {onnx_time:>10.2f} ms  {onnx_fps:>10.1f} FPS  {pt_time/onnx_time:.2f}x")
    
    speedup = pt_time / onnx_time
    if speedup > 1.5:
        print(f"\n🎉 ONNX est {speedup:.2f}x plus rapide ! Gain massif !")
    elif speedup > 1.2:
        print(f"\n✅ ONNX est {speedup:.2f}x plus rapide. Bon gain.")
    else:
        print(f"\n⚠️  ONNX est seulement {speedup:.2f}x plus rapide. Gain modeste.")
    
    print("\n💡 Pour utiliser ONNX automatiquement:")
    print("   Le système détectera yolov10n.onnx si présent")
    print("=" * 60)


if __name__ == "__main__":
    main()
