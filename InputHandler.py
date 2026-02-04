import uinput
import time
import numpy as np

class InputHandler:
    def __init__(self):
        # Initialisation de la souris virtuelle
        self.device = uinput.Device([
            uinput.REL_X, 
            uinput.REL_Y, 
            uinput.BTN_LEFT
        ])
        print("🖱️ Souris InputHandler prête.")

    def move_mouse(self, x, y):
        """Mouvement relatif immédiat (pour le lissage dynamique)"""
        # On convertit en entier car uinput refuse les virgules
        ix = int(x)
        iy = int(y)
        
        # Optimisation : On n'envoie rien si c'est 0 (évite de spammer le kernel)
        if ix == 0 and iy == 0:
            return

        self.device.emit(uinput.REL_X, ix)
        self.device.emit(uinput.REL_Y, iy)

    def move_to_target(self, path):
        """(Ancienne méthode pour les courbes de Bézier)"""
        for i in range(1, len(path)):
            rel_x = int(path[i][0] - path[i-1][0])
            rel_y = int(path[i][1] - path[i-1][1])
            self.device.emit(uinput.REL_X, rel_x)
            self.device.emit(uinput.REL_Y, rel_y)
            time.sleep(0.005) 

    def click(self):
        self.device.emit(uinput.BTN_LEFT, 1)
        # Clic très court pour être réactif
        time.sleep(np.random.uniform(0.02, 0.04))
        self.device.emit(uinput.BTN_LEFT, 0)