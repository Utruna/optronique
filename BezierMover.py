class BezierGenerator:
    def __init__(self):
        self.prev_vx = 0
        self.prev_vy = 0

    def generate_curve(self, target_dx, target_dy, steps):
        p1_x = self.prev_vx * 0.15
        p1_y = self.prev_vy * 0.15

        micro_movements = []
        last_x, last_y = 0, 0

        for i in range(1, steps + 1):
            t_linear = i / steps
            t = math.sqrt(t_linear)

            bx = 2 * (1 - t) * t * p1_x + (t**2) * target_dx
            by = 2 * (1 - t) * t * p1_y + (t**2) * target_dy

            step_x = bx - last_x
            step_y = by - last_y
            
            micro_movements.append((step_x, step_y))
            last_x, last_y = bx, by

        self.prev_vx = target_dx
        self.prev_vy = target_dy

        return micro_movements