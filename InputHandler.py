import uinput
import time
import numpy as np
import threading

class InputHandler:
    def __init__(self):
        # Create a virtual input device at the kernel level via uinput
        try:
            self.device = uinput.Device([
                uinput.REL_X,
                uinput.REL_Y,
                uinput.BTN_LEFT
            ])
            print("🖱️ InputHandler virtual device ready (kernel-level uinput).")
        except OSError:
            print("❌ Error: Permission denied for /dev/uinput.")
            print("Run: sudo chmod 666 /dev/uinput")

    def move_mouse(self, x, y):
        """Emit a relative motion event to the kernel input subsystem.

        Parameters
        ----------
        x : int
            Horizontal displacement in device units (REL_X).
        y : int
            Vertical displacement in device units (REL_Y).
        """
        ix, iy = int(x), int(y)

        if ix == 0 and iy == 0:
            return

        self.device.emit(uinput.REL_X, ix)
        self.device.emit(uinput.REL_Y, iy)

    def _click_logic(self):
        """Internal click sequence with realistic hold duration.

        Press duration is sampled uniformly from [20 ms, 50 ms] to model
        the natural variability of human finger actuation.
        """
        self.device.emit(uinput.BTN_LEFT, 1)  # Button press
        time.sleep(np.random.uniform(0.02, 0.05))
        self.device.emit(uinput.BTN_LEFT, 0)  # Button release

    def click(self):
        """Trigger a non-blocking left-click via a daemon thread.

        The click executes in a separate thread so it does not stall the
        main acquisition loop.
        """
        threading.Thread(target=self._click_logic, daemon=True).start()

    def scroll_up(self):
        # Optional: secondary action placeholder (e.g., weapon switch)
        pass