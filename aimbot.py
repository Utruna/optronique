import cv2
import numpy as np
import keyboard
from mss import mss
import time
import math # Pour calculer la distance du centre

from VisionSystem import VisionSystem
from DecisionEngine import DecisionEngine
from InputHandler import InputHandler

# Note: On n'utilise plus MovementHandler pour des courbes pré-calculées,
# on génère la courbe dynamiquement frame par frame.

def main_live():
    vision = VisionSystem()
    engine = DecisionEngine()
    mouse = InputHandler()

    tracking_actif = False
    afficher_debug = False # Par défaut sur FALSE pour la performance max

    print("🚀 SYSTÈME DYNAMIQUE - PRÊT")
    print("[HOME] ON/OFF | [END] Quitter | [D] Afficher/Cacher Debug")

    with mss() as sct:
        monitor = sct.monitors[1]
        
        # Paramètres de capture
        crop_size = 640
        x_center = crop_size // 2
        y_center = crop_size // 2
        virtual_center = (x_center, y_center)
        
        # --- PARAMÈTRES D'HUMANISATION ---
        # Plus c'est haut, plus c'est lent et fluide.
        # 3 = Robotique / 6 = Pro Player / 10 = Legit / 15 = Très lent
        SMOOTH_FACTOR = 6.0 
        
        # Distance en pixels pour autoriser le tir (Triggerbot)
        TRIGGER_DISTANCE = 10 

        while True:
            if keyboard.is_pressed('end'): break
            
            # Gestion des Toggles
            if keyboard.is_pressed('home'):
                tracking_actif = not tracking_actif
                print(f"📡 Tracking: {'ACTIF' if tracking_actif else 'PAUSE'}")
                time.sleep(0.3)

            if keyboard.is_pressed('d'):
                afficher_debug = not afficher_debug
                if not afficher_debug: cv2.destroyAllWindows()
                print(f"📺 Debug: {'ON' if afficher_debug else 'OFF'}")
                time.sleep(0.3)

            # Optimisation : Si tracking OFF et Debug OFF, on dort (0% CPU)
            if not tracking_actif and not afficher_debug:
                time.sleep(0.1)
                continue

            try:
                # 1. Capture Rapide
                sct_img = sct.grab(monitor)
                full_frame = np.array(sct_img)
                
                # Crop au centre (méthode rapide)
                h, w = full_frame.shape[:2]
                mid_x, mid_y = w // 2, h // 2
                frame = full_frame[mid_y - y_center : mid_y + y_center, 
                                   mid_x - x_center : mid_x + x_center]
                
                # On ne convertit les couleurs QUE si nécessaire (gain de perf)
                if afficher_debug:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    debug_frame = frame_bgr.copy()
                else:
                    # Pour YOLO, on peut souvent garder le format brut ou juste enlever l'alpha
                    frame_bgr = frame[:, :, :3] 

                # 2. Détection & Logique
                if tracking_actif:
                    detections = vision.detect_targets(frame_bgr)
                    target = engine.choose_target(detections, virtual_center)

                    if target:
                        # Calcul de la distance brute (erreur)
                        dx = target['x'] - x_center
                        dy = target['y'] - y_center
                        dist = math.hypot(dx, dy)

                        # --- TRIGGERBOT (Sécurité) ---
                        # On ne tire que si on est très proche du centre
                        if dist < TRIGGER_DISTANCE:
                            mouse.click()
                        else:
                            # --- MOUVEMENT DYNAMIQUE ---
                            # On ne bouge que d'une fraction de la distance (1/SMOOTH)
                            # Cela crée une courbe d'approche logarithmique naturelle
                            move_x = dx / SMOOTH_FACTOR
                            move_y = dy / SMOOTH_FACTOR
                            
                            mouse.move_mouse(move_x, move_y)

                        # Debug visuel simplifié
                        if afficher_debug:
                            cv2.circle(debug_frame, (int(target['x']), int(target['y'])), 5, (0, 255, 0), 2)
                            cv2.line(debug_frame, virtual_center, (int(target['x']), int(target['y'])), (0, 255, 0), 1)

                # 3. Affichage (Optionnel)
                if afficher_debug:
                    cv2.imshow("Vision", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        afficher_debug = False
                        cv2.destroyAllWindows()

            except Exception as e:
                pass # On ignore les erreurs mineures de frame drop

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_live()