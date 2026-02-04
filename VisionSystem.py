from ultralytics import YOLO
import torch
import functools

class VisionSystem:
    def __init__(self, model_path="yolov10n.pt"):
        # Patch pour éviter le warning PyTorch 2.6
        torch.load = functools.partial(torch.load, weights_only=False)
        
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("👁️ VisionSystem : CUDA activé (RTX 5070)")

    def detect_targets(self, frame):
        # On garde la sensibilité élevée (conf=0.25)
        results = self.model.predict(frame, conf=0.25, verbose=False)
        
        detections = []
        height, width = frame.shape[:2] # Taille de l'image (640x640)

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            w = int(x2 - x1)
            h = int(y2 - y1)
            
            # --- FILTRES ANTI-ARME / ANTI-MAIN ---
            
            # 1. Filtre de Bordure : Si la boîte touche le bas de l'image
            # L'arme vient toujours du bas de l'écran. Si y2 est proche de la hauteur max, on ignore.
            if y2 > height - 5: 
                continue 

            # 2. Filtre de Taille : Si la boîte est gigantesque
            # Une cible distante ne prend jamais 50% de la largeur de l'écran. L'arme si.
            if w > width * 0.4 or h > height * 0.4:
                continue

            # 3. Filtre de Position : Si la boîte est collée tout en bas
            # Si le centre de l'objet est dans les 10% inférieurs de l'écran, c'est sûrement l'arme.
            center_y = int((y1 + y2) / 2)
            if center_y > height * 0.90:
                continue

            # -------------------------------------

            detections.append({
                'id': i,
                'x': int((x1 + x2) / 2),
                'y': int((y1 + y2) / 2),
                'w': w,
                'h': h,
                'conf': float(box.conf)
            })
            
        return detections