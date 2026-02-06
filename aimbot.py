import cv2
import keyboard
import time
import math
import threading
import numpy as np
import re
from pathlib import Path

from VisionSystem import VisionSystem
from DecisionEngine import DecisionEngine
from InputHandler import InputHandler
from KalmanFilterMouse import KalmanFilterMouse
import config_hardware as cfg  # Réglages matériels et tuning

# --- Générateur de courbes (continuité G2) ---
class BezierGenerator:
    def __init__(self):
        # On garde la vitesse précédente pour rendre la courbe plus naturelle
        self.prev_vx = 0
        self.prev_vy = 0

    def generate_curve(self, target_dx, target_dy, steps):
        """Découpe le déplacement en micro‑pas via une Bézier quadratique."""
        # Point de contrôle dans le prolongement du mouvement précédent
        # 0.4 = inertie (plus c'est haut, plus la courbe est large)
        p1_x = self.prev_vx * 0.4
        p1_y = self.prev_vy * 0.4

        micro_movements = []
        last_x, last_y = 0, 0

        # On découpe le déplacement en `steps` étapes
        for i in range(1, steps + 1):
            t = i / steps  # t va de 0.0 à 1.0
            
            # Bézier quadratique (P0 est à 0,0)
            # B(t) = 2(1-t)t*P1 + t^2*P2
            bx = 2 * (1 - t) * t * p1_x + (t**2) * target_dx
            by = 2 * (1 - t) * t * p1_y + (t**2) * target_dy

            # Le pas courant = différence avec la position précédente
            step_x = bx - last_x
            step_y = by - last_y
            
            micro_movements.append((step_x, step_y))
            last_x, last_y = bx, by

        # Mise à jour de la vitesse pour la prochaine frame (continuité G2)
        self.prev_vx = target_dx
        self.prev_vy = target_dy

        return micro_movements

# --- Boucle principale ---
def main_live():
    # --- Configuration de base ---
    SMOOTH_FACTOR = cfg.SMOOTH_FACTOR    
    SENS_MULTIPLIER = cfg.SENS_MULTIPLIER   
    TRIGGER_DISTANCE = 12   
    COOLDOWN_TIR = 0.15     
    HEAD_OFFSET_PCT = 0.32  
    
    BEZIER_STEPS = cfg.BEZIER_STEPS
    MICRO_DELAY = cfg.MICRO_MOVEMENT_DELAY

    # 1. Initialisation (TensorRT si dispo)
    vision = VisionSystem(
        fov_size=416,
        use_tensorrt=cfg.USE_TENSORRT,
        game_window_x=cfg.GAME_WINDOW_X,
        game_window_y=cfg.GAME_WINDOW_Y,
        conf=cfg.YOLO_CONFIDENCE,
        min_y=cfg.MIN_Y_THRESHOLD
    )
    
    engine = DecisionEngine()
    mouse = InputHandler()
    bezier = BezierGenerator() 
    kalman = KalmanFilterMouse()
    
    remainder_x = 0
    remainder_y = 0

    aim_offset_x = cfg.AIM_OFFSET_X
    aim_offset_y = cfg.AIM_OFFSET_Y
    offset_step = 1
    offset_mode = False
    config_path = Path(__file__).resolve().parent / "config_hardware.py"

    tracking_actif = False
    afficher_debug = False
    dernier_tir = 0
    last_valid_target = None

    center = vision.fov_size // 2
    virtual_center = (center, center)
    
    print("🚀 SYSTÈME OPTIQUE X11 - HARDWARE OPTIMISÉ (RTX 5070)")
    print(f"⚙️ Config: Smooth={SMOOTH_FACTOR} | Steps={BEZIER_STEPS} | Sens={SENS_MULTIPLIER}")
    print(f"🔥 TensorRT: {'ACTIF' if cfg.USE_TENSORRT else 'DÉSACTIVÉ'}")
    print("[HOME] ON/OFF | [END] Quitter | [PAGE DOWN] Debug")
    print("[F9] Offset mode | [F10] Save offsets")

    try:
        while True:
            # Raccourcis clavier
            if keyboard.is_pressed('end'): break
            
            if keyboard.is_pressed('home'):
                tracking_actif = not tracking_actif
                # Reset de l'inertie pour éviter un "saut" visuel
                bezier.prev_vx, bezier.prev_vy = 0, 0 
                print(f"📡 Tracking: {'ACTIF' if tracking_actif else 'PAUSE'}")
                time.sleep(0.3)

            if keyboard.is_pressed('page down'):
                afficher_debug = not afficher_debug
                if not afficher_debug: cv2.destroyAllWindows()
                print(f"📺 Debug: {'ON' if afficher_debug else 'OFF'}")
                time.sleep(0.3)

            if keyboard.is_pressed('f9'):
                offset_mode = not offset_mode
                print(f"🎯 Offset mode: {'ON' if offset_mode else 'OFF'} | X={aim_offset_x} Y={aim_offset_y}")
                time.sleep(0.3)

            if offset_mode:
                if keyboard.is_pressed('left'):
                    aim_offset_x -= offset_step
                    print(f"🎯 Offset X={aim_offset_x} Y={aim_offset_y}")
                    time.sleep(0.05)
                elif keyboard.is_pressed('right'):
                    aim_offset_x += offset_step
                    print(f"🎯 Offset X={aim_offset_x} Y={aim_offset_y}")
                    time.sleep(0.05)
                elif keyboard.is_pressed('up'):
                    aim_offset_y -= offset_step
                    print(f"🎯 Offset X={aim_offset_x} Y={aim_offset_y}")
                    time.sleep(0.05)
                elif keyboard.is_pressed('down'):
                    aim_offset_y += offset_step
                    print(f"🎯 Offset X={aim_offset_x} Y={aim_offset_y}")
                    time.sleep(0.05)

                if keyboard.is_pressed('f10'):
                    try:
                        content = config_path.read_text(encoding='utf-8')
                        content = re.sub(r"AIM_OFFSET_X\s*=\s*-?\d+", f"AIM_OFFSET_X = {aim_offset_x}", content)
                        content = re.sub(r"AIM_OFFSET_Y\s*=\s*-?\d+", f"AIM_OFFSET_Y = {aim_offset_y}", content)
                        config_path.write_text(content, encoding='utf-8')
                        print("✅ Offsets sauvegardés dans config_hardware.py")
                    except Exception as e:
                        print(f"⚠️ Sauvegarde offsets échouée: {e}")
                    time.sleep(0.3)

            # 2. Capture + détection
            detections, frame = vision.detect_targets()

            if tracking_actif:
                target = None
                
                if detections:
                    target = engine.choose_target(detections, virtual_center)
                    if target:
                        # Mise à jour Kalman (gardé pour plus tard)
                        kalman.update(np.array([target['x'], target['y']]))
                        last_valid_target = target
                # Désactivé : prédiction Kalman quand YOLO perd la cible
                # (provoquait des rotations) — à réactiver quand mieux réglé
                
                if target:
                    # --- Calcul de la cible (tête) ---
                    dx = (target['x'] + aim_offset_x) - center
                    offset_hauteur = target['h'] * HEAD_OFFSET_PCT
                    dy = (target['y'] + aim_offset_y - offset_hauteur) - center
                    dist = math.hypot(dx, dy)

                    # --- Génération du mouvement ---
                    
                    # 1. Lissage adaptatif basé sur la distance
                    distance_ratio = min(dist / 100, 1.0)  # 0 = très proche, 1 = loin
                    adaptive_smooth = SMOOTH_FACTOR + (1.5 * (1 - distance_ratio))  # Moins de frein à courte distance
                    
                    target_move_x = (dx * SENS_MULTIPLIER) / adaptive_smooth
                    target_move_y = (dy * SENS_MULTIPLIER) / adaptive_smooth

                    # 2. Courbe de Bézier vers ce point
                    curve_steps = bezier.generate_curve(target_move_x, target_move_y, steps=BEZIER_STEPS)

                    # 3. Exécution des micro‑mouvements
                    for step_x, step_y in curve_steps:
                        
                        # On conserve les décimales pour la précision sur la durée
                        total_x = step_x + remainder_x
                        total_y = step_y + remainder_y

                        move_x_int = int(total_x)
                        move_y_int = int(total_y)

                        remainder_x = total_x - move_x_int
                        remainder_y = total_y - move_y_int

                        if move_x_int != 0 or move_y_int != 0:
                            mouse.move_mouse(move_x_int, move_y_int)
                        
                        # Micro‑pause pour garder un mouvement fluide
                        time.sleep(MICRO_DELAY)

                    # --- Logique de tir ---
                    if dist < TRIGGER_DISTANCE:
                        maintenant = time.time()
                        if maintenant - dernier_tir > COOLDOWN_TIR:
                            mouse.click()
                            dernier_tir = maintenant

                    # Debug
                    if afficher_debug and frame is not None:
                        head_x, head_y = int(target['x']), int(target['y'] - offset_hauteur)
                        adj_head_x = head_x + aim_offset_x
                        adj_head_y = head_y + aim_offset_y
                        cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), 2)
                        cv2.circle(frame, (adj_head_x, adj_head_y), 5, (0, 255, 255), 2)
                        cv2.line(frame, virtual_center, (adj_head_x, adj_head_y), (0, 255, 255), 1)
                        cv2.putText(
                            frame,
                            f"Offset X={aim_offset_x} Y={aim_offset_y}",
                            (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1
                        )

            # 3. Affichage
            if afficher_debug and frame is not None:
                cv2.drawMarker(frame, virtual_center, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                offset_center = (center + aim_offset_x, center + aim_offset_y)
                cv2.drawMarker(frame, offset_center, (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
                cv2.putText(
                    frame,
                    f"Offset X={aim_offset_x} Y={aim_offset_y}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
                cv2.imshow("Vision Debug", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    afficher_debug = False

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
    finally:
        if hasattr(vision, 'stop'): vision.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_live()