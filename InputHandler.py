import uinput
import time
import numpy as np
import threading

class InputHandler:
    def __init__(self):
        # Création du périphérique virtuel au niveau kernel
        try:
            self.device = uinput.Device([
                uinput.REL_X, 
                uinput.REL_Y, 
                uinput.BTN_LEFT
            ])
            print("🖱️ Souris InputHandler prête (Mode Asynchrone).")
        except OSError:
            print("❌ Erreur : Permission refusée pour /dev/uinput.")
            print("Tape : sudo chmod 666 /dev/uinput")

    def move_mouse(self, x, y):
        """Envoie un mouvement relatif au moteur de jeu."""
        # Conversion en entiers (obligatoire pour uinput)
        ix, iy = int(x), int(y)
        
        if ix == 0 and iy == 0:
            return

        self.device.emit(uinput.REL_X, ix)
        self.device.emit(uinput.REL_Y, iy)

    def _click_logic(self):
        """Logique interne du clic avec une durée réaliste."""
        self.device.emit(uinput.BTN_LEFT, 1)  # Appuyer
        # Durée de pression entre 20ms et 50ms
        time.sleep(np.random.uniform(0.02, 0.05))
        self.device.emit(uinput.BTN_LEFT, 0)  # Relâcher

    def click(self):
        """Déclenche un clic sans bloquer le script principal."""
        # On lance le clic dans un thread séparé pour ne pas bloquer la boucle
        threading.Thread(target=self._click_logic, daemon=True).start()

    def scroll_up(self):
        # Optionnel : changement d'arme ou action secondaire
        pass