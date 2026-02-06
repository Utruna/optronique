"""
Script d'optimisation du modèle YOLO pour RTX 5070
Convertit le modèle en TensorRT pour gains de performance massifs
"""

import torch
import functools
from ultralytics import YOLO
import time

# Fix PyTorch 2.6 weights loading
torch.load = functools.partial(torch.load, weights_only=False)

def optimize_model_for_rtx5070():
    """
    Optimise YOLO avec TensorRT (NVIDIA)
    Gains attendus: 2-3x plus rapide que ONNX
    """
    print("🔧 OPTIMISATION YOLO POUR RTX 5070 (12GB VRAM)")
    print("="*60)
    
    # Charger le modèle de base
    model = YOLO("yolov10n.pt")
    
    # Export en TensorRT (optimisé NVIDIA)
    print("\n1️⃣ Export TensorRT en cours...")
    model.export(
        format='engine',  # TensorRT
        imgsz=416,
        half=True,        # FP16 pour RTX
        device=0,
        workspace=4,      # 4GB de workspace VRAM
        simplify=True,
        batch=1
    )
    
    print("✅ Modèle optimisé: yolov10n.engine")
    
    # Benchmark
    print("\n2️⃣ Benchmark de performance...")
    model_original = YOLO("yolov10n.pt")
    model_optimized = YOLO("yolov10n.engine")
    
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    
    # Test modèle original
    times_original = []
    for i in range(20):
        start = time.perf_counter()
        model_original.predict(dummy_frame, verbose=False, device=0, half=True)
        elapsed = time.perf_counter() - start
        if i >= 5:  # Skip warmup
            times_original.append(elapsed)
    
    # Test modèle optimisé
    times_optimized = []
    for i in range(20):
        start = time.perf_counter()
        model_optimized.predict(dummy_frame, verbose=False)
        elapsed = time.perf_counter() - start
        if i >= 5:  # Skip warmup
            times_optimized.append(elapsed)
    
    avg_original = np.mean(times_original) * 1000
    avg_optimized = np.mean(times_optimized) * 1000
    speedup = avg_original / avg_optimized
    
    print("\n" + "="*60)
    print("📊 RÉSULTATS BENCHMARK")
    print("="*60)
    print(f"Modèle Original (.pt):   {avg_original:.2f}ms ({1000/avg_original:.1f} FPS)")
    print(f"Modèle TensorRT (.engine): {avg_optimized:.2f}ms ({1000/avg_optimized:.1f} FPS)")
    print(f"🚀 Accélération: {speedup:.2f}x plus rapide !")
    print("="*60)
    
    return "yolov10n.engine"

if __name__ == "__main__":
    optimized_path = optimize_model_for_rtx5070()
    print(f"\n✅ Utilisez maintenant: VisionSystem(model_path='{optimized_path}')")
