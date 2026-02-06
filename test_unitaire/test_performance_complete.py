#!/usr/bin/env python3
"""
Test de performance du système complet
Mesure FPS réels avec tous les composants
"""

import time
import numpy as np
import torch
import functools
from VisionSystem import VisionSystem
from DecisionEngine import DecisionEngine
from KalmanFilterMouse import KalmanFilterMouse
from aimbot import BezierGenerator
import config_hardware as cfg

torch.load = functools.partial(torch.load, weights_only=False)

def test_full_system():
    print("="*60)
    print("🧪 TEST PERFORMANCE SYSTÈME COMPLET")
    print("="*60)
    
    # Init (sans capture X11 réelle)
    print("\n1️⃣ Initialisation des composants...")
    engine = DecisionEngine()
    kalman = KalmanFilterMouse()
    bezier = BezierGenerator()
    
    # Simuler des détections YOLO
    center = (208, 208)
    
    detections_frames = [
        [{'id': 0, 'x': 180, 'y': 180, 'w': 50, 'h': 80, 'conf': 0.9}],
        [{'id': 0, 'x': 185, 'y': 185, 'w': 50, 'h': 80, 'conf': 0.9}],
        [{'id': 0, 'x': 190, 'y': 190, 'w': 50, 'h': 80, 'conf': 0.9}],
        [],  # Frame perdue
        [{'id': 0, 'x': 200, 'y': 200, 'w': 50, 'h': 80, 'conf': 0.9}],
    ]
    
    print("\n2️⃣ Benchmark du pipeline de tracking...")
    print(f"   Config: BEZIER_STEPS={cfg.BEZIER_STEPS}, SMOOTH={cfg.SMOOTH_FACTOR}")
    
    num_iterations = 1000
    times = []
    last_valid_target = None
    
    for i in range(num_iterations):
        detections = detections_frames[i % len(detections_frames)]
        
        start = time.perf_counter()
        
        # Pipeline complet
        if detections:
            target = engine.choose_target(detections, center)
            if target:
                kalman.update(np.array([target['x'], target['y']]))
                last_valid_target = target
        else:
            # Prédiction Kalman
            if last_valid_target:
                predicted = kalman.predict()
                target = last_valid_target.copy()
                target['x'], target['y'] = int(predicted[0]), int(predicted[1])
        
        # Générer mouvement
        if last_valid_target:
            dx = last_valid_target['x'] - center[0]
            dy = last_valid_target['y'] - center[1]
            curve = bezier.generate_curve(dx, dy, steps=cfg.BEZIER_STEPS)
            
            # Simuler l'exécution des micro-mouvements
            for step_x, step_y in curve:
                move_x_int = int(step_x)
                move_y_int = int(step_y)
                # Pas de vrai mouvement souris ici
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # Stats
    times_ms = [t * 1000 for t in times]
    avg = np.mean(times_ms)
    min_t = np.min(times_ms)
    max_t = np.max(times_ms)
    fps = 1000 / avg
    
    print("\n📊 Résultats Pipeline Tracking:")
    print(f"   Temps moyen: {avg:.3f}ms ({fps:.0f} FPS)")
    print(f"   Temps min:   {min_t:.3f}ms")
    print(f"   Temps max:   {max_t:.3f}ms")
    
    # Estimation système complet
    print("\n3️⃣ Estimation FPS système complet:")
    yolo_time = 3.73  # Mesuré précédemment
    tracking_time = avg
    total_time = yolo_time + tracking_time
    system_fps = 1000 / total_time
    
    print(f"   YOLO (asynchrone): {yolo_time:.2f}ms")
    print(f"   Tracking:          {tracking_time:.2f}ms")
    print(f"   TOTAL (séquentiel): {total_time:.2f}ms ({1000/total_time:.0f} FPS)")
    print(f"   TOTAL (asynchrone): {max(yolo_time, tracking_time):.2f}ms ({1000/max(yolo_time, tracking_time):.0f} FPS)")
    
    print("\n" + "="*60)
    print("💡 ANALYSE")
    print("="*60)
    
    if tracking_time < 0.5:
        print("✅ Tracking ultra-rapide (<0.5ms)")
    elif tracking_time < 1.0:
        print("✅ Tracking très rapide (<1ms)")
    else:
        print(f"⚠️  Tracking pourrait être optimisé ({tracking_time:.2f}ms)")
    
    if yolo_time < 5:
        print("✅ YOLO ultra-rapide (<5ms)")
    elif yolo_time < 10:
        print("✅ YOLO très rapide (<10ms)")
    else:
        print(f"⚠️  YOLO pourrait être optimisé ({yolo_time:.2f}ms)")
    
    # Estimation avec threading
    main_loop_overhead = 0.2  # Gestion événements, etc.
    async_fps = 1000 / (max(yolo_time, tracking_time) + main_loop_overhead)
    
    print(f"\n🚀 FPS estimé avec threading asynchrone: {async_fps:.0f} FPS")
    print("="*60)


if __name__ == "__main__":
    test_full_system()
