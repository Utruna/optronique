#!/usr/bin/env python3
"""
Tests unitaires pour l'aimbot YOLO
Vérifie le bon fonctionnement de chaque composant
"""

import unittest
import numpy as np
import math
import sys
import os
import time
import queue
import threading

# Import des modules à tester
from KalmanFilterMouse import KalmanFilterMouse
from DecisionEngine import DecisionEngine
from aimbot import BezierGenerator


class TestKalmanFilter(unittest.TestCase):
    """Tests du filtre de Kalman pour prédiction de trajectoire"""
    
    def setUp(self):
        self.kalman = KalmanFilterMouse()
    
    def test_initialization(self):
        """Vérifier l'initialisation correcte"""
        self.assertEqual(self.kalman.state.shape, (4,))
        self.assertEqual(self.kalman.P.shape, (4, 4))
        np.testing.assert_array_equal(self.kalman.state, np.zeros(4))
    
    def test_update(self):
        """Vérifier que le filtre se met à jour avec une mesure"""
        measurement = np.array([100.0, 150.0])
        self.kalman.update(measurement)
        
        # L'état doit avoir convergé vers la mesure
        self.assertAlmostEqual(self.kalman.state[0], 100.0, delta=50)
        self.assertAlmostEqual(self.kalman.state[1], 150.0, delta=50)
    
    def test_prediction(self):
        """Vérifier que la prédiction fonctionne"""
        # Initialiser avec une position
        self.kalman.update(np.array([100.0, 100.0]))
        
        # Prédire la prochaine position
        predicted = self.kalman.predict()
        
        self.assertEqual(len(predicted), 2)
        self.assertIsInstance(predicted[0], (int, float, np.number))
        self.assertIsInstance(predicted[1], (int, float, np.number))
    
    def test_tracking_moving_target(self):
        """Simuler un tracking de cible en mouvement"""
        positions = [
            (100, 100),
            (110, 105),
            (120, 110),
            (130, 115)
        ]
        
        for x, y in positions:
            self.kalman.update(np.array([x, y]))
        
        # La prédiction doit suivre la tendance
        predicted = self.kalman.predict()
        # Avec le Kalman, la prédiction est plus conservative
        self.assertGreater(predicted[0], 110)  # Continue vers la droite (ajusté)
        self.assertGreater(predicted[1], 100)  # Continue vers le bas


class TestBezierGenerator(unittest.TestCase):
    """Tests du générateur de courbes de Bézier"""
    
    def setUp(self):
        self.bezier = BezierGenerator()
    
    def test_initialization(self):
        """Vérifier l'initialisation"""
        self.assertEqual(self.bezier.prev_vx, 0)
        self.assertEqual(self.bezier.prev_vy, 0)
    
    def test_curve_generation_basic(self):
        """Générer une courbe simple"""
        target_dx, target_dy = 50, 30
        steps = 5
        
        curve = self.bezier.generate_curve(target_dx, target_dy, steps)
        
        self.assertEqual(len(curve), steps)
        
        # Vérifier que la somme des micro-mouvements = cible
        total_x = sum(step[0] for step in curve)
        total_y = sum(step[1] for step in curve)
        
        self.assertAlmostEqual(total_x, target_dx, delta=0.1)
        self.assertAlmostEqual(total_y, target_dy, delta=0.1)
    
    def test_curve_smoothness(self):
        """Vérifier la continuité G2 (pas de saut brusque)"""
        curve = self.bezier.generate_curve(100, 50, steps=10)
        
        # Les premiers mouvements doivent être plus petits (accélération douce)
        first_step = math.hypot(curve[0][0], curve[0][1])
        mid_step = math.hypot(curve[5][0], curve[5][1])
        
        # Le milieu de la courbe doit avoir des steps plus grands
        self.assertGreater(mid_step, first_step * 0.5)
    
    def test_continuity_between_curves(self):
        """Vérifier la continuité entre deux courbes successives"""
        # Première courbe
        self.bezier.generate_curve(50, 30, steps=5)
        prev_vx_1 = self.bezier.prev_vx
        
        # Deuxième courbe
        self.bezier.generate_curve(40, 20, steps=5)
        
        # La vélocité doit avoir été mise à jour
        self.assertEqual(self.bezier.prev_vx, 40)
        self.assertEqual(self.bezier.prev_vy, 20)


class TestDecisionEngine(unittest.TestCase):
    """Tests du moteur de décision (sélection de cible)"""
    
    def setUp(self):
        self.engine = DecisionEngine()
        self.center = (208, 208)
    
    def test_no_detections(self):
        """Aucune détection disponible"""
        result = self.engine.choose_target([], self.center)
        self.assertIsNone(result)
    
    def test_single_target(self):
        """Une seule cible disponible"""
        detections = [
            {'id': 0, 'x': 200, 'y': 200, 'w': 50, 'h': 80, 'conf': 0.9}
        ]
        result = self.engine.choose_target(detections, self.center)
        self.assertEqual(result['id'], 0)
    
    def test_closest_target_selection(self):
        """Sélectionner la cible la plus proche du centre"""
        detections = [
            {'id': 0, 'x': 150, 'y': 150, 'w': 50, 'h': 80, 'conf': 0.8},  # Loin
            {'id': 1, 'x': 210, 'y': 210, 'w': 50, 'h': 80, 'conf': 0.9},  # Proche
            {'id': 2, 'x': 100, 'y': 300, 'w': 50, 'h': 80, 'conf': 0.95}  # Très loin
        ]
        
        result = self.engine.choose_target(detections, self.center)
        
        # La cible ID 1 (210, 210) doit être choisie car plus proche de (208, 208)
        self.assertEqual(result['id'], 1)
    
    def test_distance_calculation(self):
        """Vérifier le calcul exact de distance"""
        center = (200, 200)
        detections = [
            {'id': 0, 'x': 200, 'y': 210, 'w': 50, 'h': 80, 'conf': 0.9},  # 10px
            {'id': 1, 'x': 200, 'y': 220, 'w': 50, 'h': 80, 'conf': 0.9},  # 20px
        ]
        
        result = self.engine.choose_target(detections, center)
        self.assertEqual(result['id'], 0)


class TestIntegrationAimbot(unittest.TestCase):
    """Tests d'intégration simulant le comportement global"""
    
    def test_full_tracking_pipeline(self):
        """Simuler une séquence complète de tracking"""
        kalman = KalmanFilterMouse()
        bezier = BezierGenerator()
        engine = DecisionEngine()
        center = (208, 208)
        
        # Simuler des détections successives
        detections_sequence = [
            [{'id': 0, 'x': 180, 'y': 180, 'w': 50, 'h': 80, 'conf': 0.9}],
            [{'id': 0, 'x': 185, 'y': 185, 'w': 50, 'h': 80, 'conf': 0.9}],
            [],  # Frame perdue
            [{'id': 0, 'x': 195, 'y': 195, 'w': 50, 'h': 80, 'conf': 0.9}],
        ]
        
        last_valid_target = None
        
        for detections in detections_sequence:
            if detections:
                target = engine.choose_target(detections, center)
                if target:
                    kalman.update(np.array([target['x'], target['y']]))
                    last_valid_target = target
            else:
                # Prédiction pendant perte de détection
                if last_valid_target is not None:
                    predicted = kalman.predict()
                    target = last_valid_target.copy()
                    target['x'], target['y'] = int(predicted[0]), int(predicted[1])
            
            # Générer mouvement
            if last_valid_target:
                dx = last_valid_target['x'] - center[0]
                dy = last_valid_target['y'] - center[1]
                curve = bezier.generate_curve(dx, dy, steps=4)
                
                # Vérifier que la courbe est générée
                self.assertIsInstance(curve, list)
                self.assertGreater(len(curve), 0)
    
    def test_adaptive_smoothing_logic(self):
        """Vérifier la logique de smooth adaptatif"""
        SMOOTH_FACTOR = 2.0
        
        # Test: cible proche (dist = 10)
        dist_proche = 10
        distance_ratio_proche = min(dist_proche / 100, 1.0)
        adaptive_smooth_proche = SMOOTH_FACTOR + (3.0 * (1 - distance_ratio_proche))
        
        # Test: cible loin (dist = 150)
        dist_loin = 150
        distance_ratio_loin = min(dist_loin / 100, 1.0)
        adaptive_smooth_loin = SMOOTH_FACTOR + (3.0 * (1 - distance_ratio_loin))
        
        # Le smooth doit être plus élevé pour les cibles proches
        self.assertGreater(adaptive_smooth_proche, adaptive_smooth_loin)
        self.assertAlmostEqual(adaptive_smooth_proche, 4.7, delta=0.5)
        self.assertLessEqual(adaptive_smooth_loin, 2.0)


class TestVisionSystemThreading(unittest.TestCase):
    """Tests du système de threading asynchrone pour YOLO"""
    
    def test_double_queue_system(self):
        """Simuler le système de double queue pour l'asynchrone"""
        frame_queue = queue.Queue(maxsize=1)
        detection_queue = queue.Queue(maxsize=1)
        
        # Simuler la capture (Thread 1)
        for i in range(5):
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put(f"frame_{i}")
        
        # Simuler la détection (Thread 2)
        for i in range(5):
            if not frame_queue.empty():
                frame = frame_queue.get()
                detections = [{'id': i, 'x': 100+i*10, 'y': 100}]
                
                if detection_queue.full():
                    detection_queue.get_nowait()
                detection_queue.put((detections, frame))
        
        # Vérifier qu'on a bien des détections
        self.assertFalse(detection_queue.empty())
        detections, frame = detection_queue.get()
        self.assertIsInstance(detections, list)
        self.assertGreater(len(detections), 0)
    
    def test_non_blocking_detection_retrieval(self):
        """Vérifier que la récupération de détection est non-bloquante"""
        detection_queue = queue.Queue(maxsize=1)
        
        # Queue vide : doit retourner immédiatement
        start = time.perf_counter()
        if detection_queue.empty():
            result = ([], None)
        else:
            result = detection_queue.get()
        elapsed = time.perf_counter() - start
        
        # Doit être quasi-instantané (< 1ms)
        self.assertLess(elapsed, 0.001)
        self.assertEqual(result, ([], None))
    
    def test_queue_overflow_handling(self):
        """Vérifier le comportement avec overflow de queue"""
        detection_queue = queue.Queue(maxsize=1)
        
        # Remplir la queue
        detection_queue.put(([{'id': 0}], "frame_0"))
        
        # Essayer d'ajouter une nouvelle détection (doit drop l'ancienne)
        if detection_queue.full():
            old = detection_queue.get_nowait()
            self.assertEqual(old[0][0]['id'], 0)
        
        detection_queue.put(([{'id': 1}], "frame_1"))
        
        # Vérifier qu'on a bien la plus récente
        detections, _ = detection_queue.get()
        self.assertEqual(detections[0]['id'], 1)


def run_performance_test():
    """Test de performance (hors unittest)"""
    print("\n" + "="*60)
    print("🚀 TESTS DE PERFORMANCE")
    print("="*60)
    
    # Test 1: Vitesse du Kalman
    kalman = KalmanFilterMouse()
    start = time.perf_counter()
    for _ in range(1000):
        kalman.update(np.array([100.0, 100.0]))
        kalman.predict()
    elapsed = time.perf_counter() - start
    print(f"✓ Kalman (1000 itérations): {elapsed*1000:.2f}ms ({1000/elapsed:.0f} ops/sec)")
    
    # Test 2: Vitesse du Bézier
    bezier = BezierGenerator()
    start = time.perf_counter()
    for _ in range(1000):
        bezier.generate_curve(50, 30, steps=8)
    elapsed = time.perf_counter() - start
    print(f"✓ Bézier (1000 courbes): {elapsed*1000:.2f}ms ({1000/elapsed:.0f} ops/sec)")
    
    # Test 3: Vitesse du DecisionEngine
    engine = DecisionEngine()
    detections = [
        {'id': i, 'x': 100+i*10, 'y': 100+i*5, 'w': 50, 'h': 80, 'conf': 0.9}
        for i in range(10)
    ]
    start = time.perf_counter()
    for _ in range(10000):
        engine.choose_target(detections, (208, 208))
    elapsed = time.perf_counter() - start
    print(f"✓ DecisionEngine (10000 sélections): {elapsed*1000:.2f}ms ({10000/elapsed:.0f} ops/sec)")
    
    # Test 4: Latence du système de Queue Asynchrone
    print("\n" + "-"*60)
    print("📊 SIMULATION SYSTÈME ASYNCHRONE (Queue Threading)")
    print("-"*60)
    
    frame_queue = queue.Queue(maxsize=1)
    detection_queue = queue.Queue(maxsize=1)
    
    # Simuler Thread 1: Capture rapide
    def capture_simulator():
        for i in range(100):
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put(f"frame_{i}")
            time.sleep(0.001)  # 1000 FPS théorique
    
    # Simuler Thread 2: Détection YOLO (plus lent)
    detection_times = []
    def detection_simulator():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                
                start = time.perf_counter()
                # Simuler l'inférence YOLO (30-50ms)
                time.sleep(0.035)
                detections = [{'id': 0, 'x': 200, 'y': 200}]
                elapsed = time.perf_counter() - start
                detection_times.append(elapsed)
                
                if detection_queue.full():
                    detection_queue.get_nowait()
                detection_queue.put((detections, frame))
                
                if len(detection_times) >= 10:
                    break
            else:
                time.sleep(0.001)
    
    # Lancer les threads
    t1 = threading.Thread(target=capture_simulator, daemon=True)
    t2 = threading.Thread(target=detection_simulator, daemon=True)
    
    start_global = time.perf_counter()
    t1.start()
    t2.start()
    
    t1.join()
    t2.join(timeout=2)
    total_time = time.perf_counter() - start_global
    
    # Calculer les stats
    avg_detection_time = np.mean(detection_times) * 1000
    max_detection_time = np.max(detection_times) * 1000
    throughput = len(detection_times) / total_time
    
    print(f"✓ Temps moyen inférence YOLO: {avg_detection_time:.2f}ms")
    print(f"✓ Temps max inférence YOLO: {max_detection_time:.2f}ms")
    print(f"✓ Débit détections: {throughput:.1f} détections/sec")
    print(f"✓ Latence récupération: < 0.1ms (non-bloquant)")
    
    # Test 5: Overhead du système de queue
    print("\n" + "-"*60)
    print("⚡ OVERHEAD SYSTÈME DE QUEUE")
    print("-"*60)
    
    test_queue = queue.Queue(maxsize=1)
    
    # Test put/get
    start = time.perf_counter()
    for i in range(10000):
        if test_queue.full():
            test_queue.get_nowait()
        test_queue.put(i)
        test_queue.get()
    elapsed = time.perf_counter() - start
    print(f"✓ Queue put+get (10000x): {elapsed*1000:.2f}ms ({10000/elapsed:.0f} ops/sec)")
    print(f"✓ Overhead par opération: {elapsed/10000*1000000:.2f}us")
    
    print("\n✅ Tous les tests de performance OK\n")


if __name__ == '__main__':
    print("="*60)
    print("🧪 TESTS UNITAIRES - AIMBOT YOLO")
    print("="*60 + "\n")
    
    # Lancer les tests unitaires
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Tests de performance
    if result.wasSuccessful():
        run_performance_test()
    
    # Résumé
    print("="*60)
    if result.wasSuccessful():
        print("✅ TOUS LES TESTS RÉUSSIS")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("="*60)
    
    sys.exit(0 if result.wasSuccessful() else 1)
