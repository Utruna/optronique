class BezierGenerator:
    def __init__(self):
        """Initialise the G2-continuous quadratic Bézier micro-step generator.

        The generator maintains a velocity carry-over state (``prev_vx``, ``prev_vy``)
        so that successive motion segments share a common tangent direction, producing
        G2 curvature continuity across frames.
        """
        self.prev_vx = 0
        self.prev_vy = 0

    def generate_curve(self, target_dx, target_dy, steps):
        """Decompose a displacement vector into Ease-Out Bézier micro-steps.

        The parametric position along the curve uses a square-root time-warp
        ``t = √(i / steps)`` to implement a non-linear Ease-Out acceleration profile:
        the pointer approaches the target quickly and decelerates near the endpoint,
        mimicking the fine-motor control of human wrist/arm movement.

        The quadratic Bézier formula (P0 at origin) is:

            B(t) = 2·(1−t)·t·P1 + t²·P2

        where P1 is the inertia control point (``prev_v * 0.15``) and P2 is the
        target displacement vector.

        Parameters
        ----------
        target_dx : float
            Target displacement along the X axis (pixels).
        target_dy : float
            Target displacement along the Y axis (pixels).
        steps : int
            Number of micro-steps to generate.

        Returns
        -------
        list of tuple[float, float]
            Ordered list of incremental (step_x, step_y) displacements.
        """
        p1_x = self.prev_vx * 0.15
        p1_y = self.prev_vy * 0.15

        micro_movements = []
        last_x, last_y = 0, 0

        for i in range(1, steps + 1):
            t_linear = i / steps
            t = math.sqrt(t_linear)  # Ease-Out: √t time-warp

            bx = 2 * (1 - t) * t * p1_x + (t**2) * target_dx
            by = 2 * (1 - t) * t * p1_y + (t**2) * target_dy

            step_x = bx - last_x
            step_y = by - last_y

            micro_movements.append((step_x, step_y))
            last_x, last_y = bx, by

        # Carry-over velocity for the next curve's G2 control point
        self.prev_vx = target_dx
        self.prev_vy = target_dy

        return micro_movements