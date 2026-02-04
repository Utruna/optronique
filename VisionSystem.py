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
    def __init__(self, model_path="yolov10n.pt", fov_size=416, monitor_index=1):
        torch.load = functools.partial(torch.load, weights_only=False)
        
        # 1. Connexion X11
        self.display = Xlib.display.Display()
        
        # --- LA MODIFICATION EST ICI ---
        # Au lieu de prendre la racine globale, on cible l'écran spécifique
        if monitor_index >= self.display.screen_count():
            monitor_index = 0
        
        self.screen = self.display.screen(monitor_index)
        self.target_drawable = self.screen.root # On capture la racine de CET écran uniquement
        
        # On récupère la géométrie RÉELLE de cet écran précis via X11
        geo = self.target_drawable.get_geometry()
        self.screen_w = geo.width
        self.screen_h = geo.height
        
        print(f"🖥️ Xlib Screen {monitor_index} détecté : {self.screen_w}x{self.screen_h}")

        # 2. Calcul du centre RELATIF (0,0 est maintenant le coin de DP-2)
        self.fov_size = fov_size
        self.left = (self.screen_w // 2) - (self.fov_size // 2)
        self.top = (self.screen_h // 2) - (self.fov_size // 2)

        # Sécurité pour ne pas sortir des pixels de l'écran local
        self.left = max(0, min(self.left, self.screen_w - self.fov_size))
        self.top = max(0, min(self.top, self.screen_h - self.fov_size))

        print(f"🎯 ROI Relatif validé : {self.left},{self.top}")

        # 3. IA & Threading
        self.model = YOLO(model_path)
        if torch.cuda.is_available(): self.model.to('cuda')
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def capture_frame(self):
        """Capture ultra-haute performance via XGetImage"""
        # On utilise le drawable de l'écran cible
        raw_img = self.target_drawable.get_image(
            self.left, self.top, 
            self.fov_size, self.fov_size, 
            Xlib.X.ZPixmap, 0xffffffff
        )
        frame = np.frombuffer(raw_img.data, dtype=np.uint8)
        frame = frame.reshape(self.fov_size, self.fov_size, 4)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _capture_loop(self):
        while self.running:
            try:
                frame = self.capture_frame()
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
            except Exception as e:
                time.sleep(0.01)

    def detect_targets(self):
        if self.frame_queue.empty(): 
            return [], None
            
        frame = self.frame_queue.get()
        
        results = self.model.predict(
            frame, 
            conf=0.40,  
            classes=[0],
            imgsz=416,        
            device=0,         
            half=True,        
            verbose=False,    
            augment=False,    
            agnostic_nms=True 
        )
        
        detections = []
        if len(results) > 0:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if y2 > self.fov_size - 5: continue 
                
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
        """Arrête proprement le thread de capture"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            # On ne bloque pas si le thread est déjà mort
            if self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1)
        print("🛑 VisionSystem arrêté.")