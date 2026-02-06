import Xlib.display
import Xlib.X
import numpy as np
import cv2
if not hasattr(cv2, 'setNumThreads'):
    cv2.setNumThreads = lambda x: None
import torch
import functools
import threading
import queue
import screeninfo
import time
from ultralytics import YOLO

class VisionSystem:
    def __init__(self, model_path="yolov10n.pt", fov_size=416, monitor_index=1, use_tensorrt=True, game_window_x=None, game_window_y=None, conf=0.25, min_y=0):
        torch.load = functools.partial(torch.load, weights_only=False)
        
        # On force un modèle PyTorch (l'ONNX a posé des soucis de float16/float32)
        if not model_path.endswith('.pt'):
            model_path = "yolov10n.pt"
        
        # 1. Connexion X11
        self.display = Xlib.display.Display()
        
        if monitor_index >= self.display.screen_count():
            monitor_index = 0
        
        self.screen = self.display.screen(monitor_index)
        self.target_drawable = self.screen.root
        
        geo = self.target_drawable.get_geometry()
        self.screen_w = geo.width
        self.screen_h = geo.height
        
        print(f"🖥️ Xlib Screen {monitor_index} détecté : {self.screen_w}x{self.screen_h}")

        # 2. Calcul de la zone ROI
        self.fov_size = fov_size
        
        # Si des coordonnées de fenêtre sont fournies, on les utilise
        if game_window_x is not None and game_window_y is not None:
            self.left = game_window_x
            self.top = game_window_y
        else:
            # Sinon, on centre la ROI à l'écran
            self.left = (self.screen_w // 2) - (self.fov_size // 2)
            self.top = (self.screen_h // 2) - (self.fov_size // 2)

        self.left = max(0, min(self.left, self.screen_w - self.fov_size))
        self.top = max(0, min(self.top, self.screen_h - self.fov_size))

        print(f"🎯 ROI Calibrée : {self.left},{self.top}")

        # 3. Modèle (mode synchrone pour fiabilité)
        self.model = YOLO(model_path)
        
        # Optimisations GPU (si dispo)
        if torch.cuda.is_available():
            self.model.to('cuda')
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"⚡ GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        self.running = True
        self.conf = conf
        self.min_y = min_y

        print("🧵 Mode synchrone : capture + YOLO dans la boucle principale")

    def capture_frame(self):
        """Capture rapide via XGetImage."""
        raw_img = self.target_drawable.get_image(
            self.left, self.top, 
            self.fov_size, self.fov_size, 
            Xlib.X.ZPixmap, 0xffffffff
        )
        frame = np.frombuffer(raw_img.data, dtype=np.uint8)
        frame = frame.reshape(self.fov_size, self.fov_size, 4)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def detect_targets(self):
        """Capture + inférence YOLO en synchrone (fiable)."""
        frame = self.capture_frame()

        # Inférence YOLO
        with torch.cuda.amp.autocast():
            results = self.model.predict(
                frame,
            conf=self.conf,
                classes=[0],
                imgsz=416,
                device=0,
                half=True,
                verbose=False,
                augment=False,
                agnostic_nms=True,
                max_det=10
            )

        detections = []
        if len(results) > 0:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if y2 > self.fov_size - 5:
                    continue
                if y1 < self.min_y:
                    continue

                detections.append({
                    'id': i,
                    'x': int((x1 + x2) / 2),
                    'y': int((y1 + y2) / 2),
                    'w': int(x2 - x1),
                    'h': int(y2 - y1),
                    'conf': float(box.conf)
                })

        return detections, frame
    
    def stop(self):
        """Arrêt propre (pas de thread en mode synchrone)."""
        self.running = False

        print("🛑 VisionSystem arrêté")
