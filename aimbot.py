import cv2
import keyboard
import time
import math
import threading

from VisionSystem import VisionSystem
from DecisionEngine import DecisionEngine
from InputHandler import InputHandler

# --- CLASSE DE GÉNÉRATION DE COURBES (G2 CONTINUITY) ---
class BezierGenerator:
    def __init__(self):
        # On stocke la vélocité précédente pour assurer la continuité de la courbe
        self.prev_vx = 0
        self.prev_vy = 0

    def generate_curve(self, target_dx, target_dy, steps):
        """
        Génère une liste de micro-mouvements (dx, dy) basés sur une courbe de Bézier quadratique.
        P0 = (0,0) (Position actuelle)
        P1 = Point de contrôle (basé sur l'élan précédent)
        P2 = (target_dx, target_dy) (Cible)
        """
        # Le point de contrôle P1 est situé dans le prolongement du mouvement précédent
        # 0.4 est le facteur d'inertie (plus il est haut, plus la courbe est large)
        p1_x = self.prev_vx * 0.4
        p1_y = self.prev_vy * 0.4

        micro_movements = []
        last_x, last_y = 0, 0

        # On découpe le mouvement en 'steps' étapes
        for i in range(1, steps + 1):
            t = i / steps # t va de 0.0 à 1.0
            
            # Formule de Bézier Quadratique simplifiée (P0 étant à 0,0)
            # B(t) = 2(1-t)t*P1 + t^2*P2
            bx = 2 * (1 - t) * t * p1_x + (t**2) * target_dx
            by = 2 * (1 - t) * t * p1_y + (t**2) * target_dy

            # Le mouvement pour cette étape est la différence avec la position précédente
            step_x = bx - last_x
            step_y = by - last_y
            
            micro_movements.append((step_x, step_y))
            last_x, last_y = bx, by

        # Mise à jour de la vélocité pour la prochaine frame (Continuité G2)
        self.prev_vx = target_dx
        self.prev_vy = target_dy

        return micro_movements

# --- MAIN LOOP ---
def main_live():
    # --- CONFIGURATION EXPERT (CS2 / 180Hz+) ---
    SMOOTH_FACTOR = 2.0    
    SENS_MULTIPLIER = 1.65   
    TRIGGER_DISTANCE = 12   
    COOLDOWN_TIR = 0.15     
    HEAD_OFFSET_PCT = 0.42  
    
    BEZIER_STEPS = 4        

    # 1. Initialisation
    vision = VisionSystem(fov_size=416) 
    vision.left = 1074
    vision.top = 1420
    
    engine = DecisionEngine()
    mouse = InputHandler()
    bezier = BezierGenerator() 
    
    remainder_x = 0
    remainder_y = 0

    tracking_actif = False
    afficher_debug = False
    dernier_tir = 0

    center = vision.fov_size // 2
    virtual_center = (center, center)

    print("🚀 SYSTÈME OPTIQUE X11 - PRÊT (v2.0 BEZIER)")
    print(f"⚙️ Config: Smooth={SMOOTH_FACTOR} | Steps={BEZIER_STEPS} | Sens={SENS_MULTIPLIER}")
    print("[HOME] ON/OFF | [END] Quitter | [PAGE DOWN] Debug")

    try:
        while True:
            # Gestion des inputs clavier
            if keyboard.is_pressed('end'): break
            
            if keyboard.is_pressed('home'):
                tracking_actif = not tracking_actif
                # Reset de l'inertie quand on active/désactive pour éviter un saut
                bezier.prev_vx, bezier.prev_vy = 0, 0 
                print(f"📡 Tracking: {'ACTIF' if tracking_actif else 'PAUSE'}")
                time.sleep(0.3)

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
                    # --- CALCUL CIBLE (TÊTE) ---
                    dx = target['x'] - center
                    offset_hauteur = target['h'] * HEAD_OFFSET_PCT
                    dy = (target['y'] - offset_hauteur) - center
                    dist = math.hypot(dx, dy)

                    # --- GÉNÉRATION DU MOUVEMENT COMPLEXE ---
                    
                    # 1. On applique le multiplicateur de sensi et le smooth GLOBAL d'abord
                    target_move_x = (dx * SENS_MULTIPLIER) / SMOOTH_FACTOR
                    target_move_y = (dy * SENS_MULTIPLIER) / SMOOTH_FACTOR

                    # 2. On génère la courbe de Bézier vers ce point
                    curve_steps = bezier.generate_curve(target_move_x, target_move_y, steps=BEZIER_STEPS)

                    # 3. On exécute les micro-mouvements
                    for step_x, step_y in curve_steps:
                        
                        # Accumulation des restes (Pixel Perfect Logic) à chaque micro-pas
                        total_x = step_x + remainder_x
                        total_y = step_y + remainder_y

                        move_x_int = int(total_x)
                        move_y_int = int(total_y)

                        remainder_x = total_x - move_x_int
                        remainder_y = total_y - move_y_int

                        if move_x_int != 0 or move_y_int != 0:
                            mouse.move_mouse(move_x_int, move_y_int)
                        
                        # Micro-pause essentielle pour laisser le jeu "respirer" entre les pas
                        # C'est ça qui crée la fluidité perçue par le moteur Source 2
                        time.sleep(0.0005) 

                    # --- LOGIQUE DE TIR ---
                    if dist < TRIGGER_DISTANCE:
                        maintenant = time.time()
                        if maintenant - dernier_tir > COOLDOWN_TIR:
                            mouse.click()
                            dernier_tir = maintenant

                    # Debug
                    if afficher_debug and frame is not None:
                        head_x, head_y = int(target['x']), int(target['y'] - offset_hauteur)
                        cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), 2)
                        cv2.line(frame, virtual_center, (head_x, head_y), (0, 255, 0), 1)

            # 3. Affichage
            if afficher_debug and frame is not None:
                cv2.drawMarker(frame, virtual_center, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
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