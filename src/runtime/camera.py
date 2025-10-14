import numpy as np
import cv2
import time
from picamera2 import Picamera2


class Camera:

  def __init__(self):
    self.picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")

  def __enter__(self):
    """Start the camera when entering the with-block."""
    self.picam2.start()
    return self

  def capture(self):
    """Capture a single frame and return it as a NumPy array."""
    return picam2.capture_array()


  def __exit__(self, exception_type, exception_value, exception_traceback):
    self.picam2.stop()
