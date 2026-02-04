import cv2
import keyboard
import time
import math
import threading

from VisionSystem import VisionSystem
from DecisionEngine import DecisionEngine
from InputHandler import InputHandler

def main_live():
    # --- CONFIGURATION (DÉFINIE ICI POUR ÉVITER LES ERREURS) ---
    SMOOTH_FACTOR = 3.2    
    SENS_MULTIPLIER = 1.4 
    TRIGGER_DISTANCE= 15   
    COOLDOWN_TIR = 0.2     

    # 1. Initialisation
    vision = VisionSystem(fov_size=416) 
    vision.left = 1074
    vision.top = 1420
    
    engine = DecisionEngine()
    mouse = InputHandler()

    tracking_actif = False
    afficher_debug = False
    dernier_tir = 0

    center = vision.fov_size // 2
    virtual_center = (center, center)

    print("🚀 SYSTÈME OPTIQUE X11 - PRÊT")
    print(f"⚙️ Config active : Smooth={SMOOTH_FACTOR}, Sens={SENS_MULTIPLIER}")
    print("[HOME] ON/OFF | [END] Quitter | [D] Debug")

    try:
        while True:
            # Sortie
            if keyboard.is_pressed('end'): 
                break
            
            # Activation/Désactivation
            if keyboard.is_pressed('home'):
                tracking_actif = not tracking_actif
                status = "ACTIF" if tracking_actif else "PAUSE"
                print(f"📡 Tracking: {status}")
                time.sleep(0.3)

            # Debug
            if keyboard.is_pressed('page down'):
                afficher_debug = not afficher_debug
                if not afficher_debug: cv2.destroyAllWindows()
                print(f"📺 Debug: {'ON' if afficher_debug else 'OFF'}")
                time.sleep(0.3)

            # 2. Capture et Détection
            detections, frame = vision.detect_targets()

            if tracking_actif and detections:
                target = engine.choose_target(detections, virtual_center)

                if target:
                    # Calcul des écarts de base
                    dx = target['x'] - center
                    dy = target['y'] - center
                    dist = math.hypot(dx, dy)
                    
                    # --- LE FIX POUR LES BIJOUX DE FAMILLE ---
                    # target['h'] est la hauteur totale de l'ennemi.
                    # On soustrait une partie de cette hauteur pour "monter" le viseur.
                    # 0.35 = environ le haut du torse / cou. 
                    # 0.42 = environ la tête.
                    offset_hauteur = target['h'] * 0.38 
                    
                    dy = (target['y'] - offset_hauteur) - center
                    dist = math.hypot(dx, dy)

                    # --- CALCUL DU MOUVEMENT (VARIABLES LOCALES) ---
                    # On utilise les variables définies en haut de la fonction
                    move_x = (dx * SENS_MULTIPLIER) / SMOOTH_FACTOR
                    move_y = (dy * SENS_MULTIPLIER) / SMOOTH_FACTOR

                    if abs(move_x) < 0.5: move_x = 0
                    if abs(move_y) < 0.5: move_y = 0

                    if move_x != 0 or move_y != 0:
                        mouse.move_mouse(move_x, move_y)

                    # --- LOGIQUE DE TIR ---
                    if dist < TRIGGER_DISTANCE:
                        maintenant = time.time()
                        if maintenant - dernier_tir > COOLDOWN_TIR:
                            mouse.click()
                            dernier_tir = maintenant

                    # Dessin si mode Debug activé
                    if afficher_debug and frame is not None:
                        cv2.circle(frame, (int(target['x']), int(target['y'])), 5, (0, 255, 0), 2)
                        cv2.line(frame, virtual_center, (int(target['x']), int(target['y'])), (0, 255, 0), 1)

            # 3. Affichage Debug
            if afficher_debug and frame is not None:
                cv2.imshow("Vision Debug", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    afficher_debug = False

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
    finally:
        # Nettoyage automatique à la fermeture
        if hasattr(vision, 'stop'):
            vision.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_live()