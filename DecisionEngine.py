import math

class DecisionEngine:
    def choose_target(self, detections, center_screen):
        """
        Choisit la cible la plus proche du centre (viseur).
        Pas d'aléatoire : on prend la distance minimale.
        """
        if not detections:
            return None

        best_target = None
        min_dist = float('inf')  # point de départ simple

        cx, cy = center_screen

        for target in detections:
            # Distance euclidienne entre la cible et le centre
            dx = target['x'] - cx
            dy = target['y'] - cy
            dist = math.hypot(dx, dy)

            # On garde la plus proche
            if dist < min_dist:
                min_dist = dist
                best_target = target

        return best_target