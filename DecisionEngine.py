import math

class DecisionEngine:
    def choose_target(self, detections, center_screen):
        """
        Sélectionne STRICTEMENT la cible la plus proche du centre (viseur).
        Plus aucun aléatoire : c'est mathématique.
        """
        if not detections:
            return None

        best_target = None
        min_dist = float('inf') # On commence avec une distance infinie

        cx, cy = center_screen

        for target in detections:
            # Calcul de la distance réelle (Théorème de Pythagore)
            # entre le centre de la cible et le centre de ton écran
            dx = target['x'] - cx
            dy = target['y'] - cy
            dist = math.hypot(dx, dy)

            # Comparaison simple : est-ce que celle-ci est plus près que la précédente ?
            if dist < min_dist:
                min_dist = dist
                best_target = target

        return best_target