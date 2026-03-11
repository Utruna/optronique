import math

class DecisionEngine:
    def choose_target(self, detections, center_screen):
        """Select the highest-priority target from a list of detections.

        Priority is defined as minimum Euclidean distance from the ROI centre point
        (crosshair position).  No stochastic or confidence-weighted ranking is applied
        at this stage; the closest target always wins.

        Parameters
        ----------
        detections : list[dict]
            Detection dicts produced by ``VisionSystem.detect_targets``.
        center_screen : tuple[int, int]
            (cx, cy) pixel coordinates of the ROI centre (virtual crosshair origin).

        Returns
        -------
        dict or None
            The detection entry with the smallest Euclidean distance to
            ``center_screen``, or ``None`` if ``detections`` is empty.
        """
        if not detections:
            return None

        best_target = None
        min_dist = float('inf')

        cx, cy = center_screen

        for target in detections:
            # Euclidean distance from target centre to ROI centre
            dx = target['x'] - cx
            dy = target['y'] - cy
            dist = math.hypot(dx, dy)

            if dist < min_dist:
                min_dist = dist
                best_target = target

        return best_target