import numpy as np
import random

class MovementHandler:
    def __init__(self, fuzziness=0.1):
        self.fuzziness = fuzziness

    def get_human_offset(self, center, size):
        """Add a small stochastic offset to model fine-motor imprecision.

        Parameters
        ----------
        center : tuple[int, int]
            (cx, cy) base target position in pixels.
        size : tuple[int, int]
            (w, h) of the target bounding box, used to scale the noise magnitude.

        Returns
        -------
        tuple[int, int]
            Noisy (x, y) position within ±fuzziness of the bounding box half-extents.
        """
        cx, cy = center
        w, h = size
        off_x = cx + random.uniform(-w * self.fuzziness, w * self.fuzziness)
        off_y = cy + random.uniform(-h * self.fuzziness, h * self.fuzziness)
        return int(off_x), int(off_y)

    def generate_bezier_path(self, start, end, steps=30):
        """Generate a Bézier trajectory with an optional overshoot component.

        With 10 % probability an overshoot segment is prepended to simulate the
        spring-like dynamics of rapid arm movements that slightly over-extend and
        self-correct.

        Parameters
        ----------
        start : tuple[int, int]
            Starting pixel position.
        end : tuple[int, int]
            Target pixel position.
        steps : int
            Total number of interpolation steps.

        Returns
        -------
        numpy.ndarray, shape (steps, 2), dtype int32
            Sequence of (x, y) waypoints along the trajectory.
        """
        if random.random() < 0.10:
            # Overshoot by 7 % beyond the target, then correct (spring dynamics)
            overshoot_factor = 1.07
            overshoot_end = (
                int(start[0] + (end[0] - start[0]) * overshoot_factor),
                int(start[1] + (end[1] - start[1]) * overshoot_factor)
            )

            path = self._build_curve(start, overshoot_end, int(steps * 0.8))
            return_path = self._build_curve(overshoot_end, end, int(steps * 0.2))
            return np.vstack((path, return_path))

        return self._build_curve(start, end, steps)

    def _build_curve(self, p0, p2, steps):
        """Construct a single quadratic Bézier segment with a random control point.

        Parameters
        ----------
        p0 : tuple[int, int]
            Curve start point.
        p2 : tuple[int, int]
            Curve end point.
        steps : int
            Number of interpolation samples.

        Returns
        -------
        numpy.ndarray, shape (steps, 2), dtype int32
            Sampled (x, y) positions along the curve.
        """
        # Randomised control point near the midpoint introduces natural curvature
        p1 = (
            (p0[0] + p2[0]) / 2 + random.randint(-40, 40),
            (p0[1] + p2[1]) / 2 + random.randint(-40, 40)
        )
        t = np.linspace(0, 1, steps)
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        return np.column_stack((x, y)).astype(np.int32)

    def move_mouse(self, x, y):
        """Emit an immediate relative motion event (real-time micro-step)."""
        if x == 0 and y == 0: return
        self.device.emit(uinput.REL_X, int(x))
        self.device.emit(uinput.REL_Y, int(y))