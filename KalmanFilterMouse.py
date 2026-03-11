import numpy as np

class KalmanFilterMouse:
    """Constant-velocity Kalman filter for 2-D target position tracking.

    State vector: [x, y, vx, vy]
    Observation:  [x, y]

    The filter predicts the next position by assuming constant velocity between
    frames and then corrects the estimate using the YOLO detection measurement.
    It is used as a fall-back predictor when the detector temporarily loses a target.

    Matrices
    --------
    F : 4×4 state-transition matrix (constant-velocity kinematics)
    H : 2×4 observation matrix (we observe x and y only)
    P : 4×4 error covariance matrix (initialised with high uncertainty)
    Q : 4×4 process noise covariance (model confidence)
    R : 2×2 measurement noise covariance (detector confidence)
    """

    def __init__(self):
        # Initial state: [x, y, vx, vy] = [0, 0, 0, 0]
        self.state = np.zeros(4)
        # State-transition matrix (constant-velocity model, Δt = 1 frame)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Observation matrix: maps 4-D state to 2-D measurement
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Initial error covariance (high value = high initial uncertainty)
        self.P = np.eye(4) * 1000
        # Process noise covariance (confidence in the kinematic model)
        self.Q = np.eye(4) * 0.1
        # Measurement noise covariance (confidence in YOLO detections)
        self.R = np.eye(2) * 1.0

    def predict(self):
        """Project the state estimate one step forward in time.

        Returns
        -------
        numpy.ndarray, shape (2,)
            Predicted [x, y] position for the next frame.
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[0:2]

    def update(self, measurement):
        """Correct the state estimate with a new [x, y] observation.

        Parameters
        ----------
        measurement : numpy.ndarray, shape (2,)
            Observed [x, y] position from the YOLOv10 detector.
        """
        # Innovation (residual between measurement and predicted observation)
        y = measurement - self.H @ self.state
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Optimal Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # State update
        self.state = self.state + K @ y
        # Covariance update (Joseph form)
        self.P = (np.eye(4) - K @ self.H) @ self.P
