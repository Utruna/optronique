import numpy as np
import random

class MovementHandler:
    def __init__(self, fuzziness=0.1):
        self.fuzziness = fuzziness

    def get_human_offset(self, center, size):
        """Calcule l'impact avec l'imprécision de 10%"""
        cx, cy = center
        w, h = size
        off_x = cx + random.uniform(-w * self.fuzziness, w * self.fuzziness)
        off_y = cy + random.uniform(-h * self.fuzziness, h * self.fuzziness)
        return int(off_x), int(off_y)

    def generate_bezier_path(self, start, end, steps=30):
        """Génère la trajectoire avec 10% de chance d'effet ressort"""
        if random.random() < 0.10:
            # Calcul du point de dépassement (Overshoot)
            overshoot_factor = 1.07 
            overshoot_end = (
                int(start[0] + (end[0] - start[0]) * overshoot_factor),
                int(start[1] + (end[1] - start[1]) * overshoot_factor)
            )
            
            # Trajectoire vers le dépassement puis retour
            path = self._build_curve(start, overshoot_end, int(steps * 0.8))
            return_path = self._build_curve(overshoot_end, end, int(steps * 0.2))
            return np.vstack((path, return_path))
        
        return self._build_curve(start, end, steps)

    def _build_curve(self, p0, p2, steps):
        """Logique interne de courbe de Bézier"""
        # Point de contrôle aléatoire pour la courbure humaine
        p1 = (
            (p0[0] + p2[0]) / 2 + random.randint(-40, 40),
            (p0[1] + p2[1]) / 2 + random.randint(-40, 40)
        )
        t = np.linspace(0, 1, steps)
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        return np.column_stack((x, y)).astype(np.int32)
    
    def move_mouse(self, x, y):
        """Effectue un micro-mouvement immédiat (pour le temps réel)"""
        if x == 0 and y == 0: return
        self.device.emit(uinput.REL_X, int(x))
        self.device.emit(uinput.REL_Y, int(y))