import numpy as np

class KalmanFilterMouse:
    def __init__(self):
        # État initial [x, y, vx, vy]
        self.state = np.zeros(4)
        # Matrice de transition (vitesse constante entre deux frames)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Matrice de mesure (on observe x et y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Incertitude de l'état
        self.P = np.eye(4) * 1000
        # Bruit du processus (confiance dans le modèle)
        self.Q = np.eye(4) * 0.1
        # Bruit de mesure (confiance dans la détection YOLO)
        self.R = np.eye(2) * 1.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[0:2]

    def update(self, measurement):
        # measurement = [x, y] venant de YOLO
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
