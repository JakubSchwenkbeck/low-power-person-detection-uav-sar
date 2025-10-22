import numpy as np
import cv2
import time
# from picamera2 import Picamera2



class FrameSource:
    def __init__(self, path=None):
        self.path = path
        self.cap = None
        self.cam_ctx = None

    def __enter__(self):
        if not self.path:
            self.cam_ctx = Camera()
            self.cam_ctx.__enter__()
            return self
        else:
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {self.path}")
            return self

    def capture(self):
        if not self.path:
            return self.cam_ctx.capture()
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def __exit__(self, exc_type, exc, tb):
        if not self.path and self.cam_ctx:
            self.cam_ctx.__exit__(exc_type, exc, tb)
        if self.cap:
            self.cap.release()



class Camera:

  def __init__(self):
    self.picam2 = Picamera2()
    self.picam2.preview_configuration.main.size = (640, 480)
    self.picam2.preview_configuration.main.format = "RGB888"
    self.picam2.preview_configuration.align()
    self.picam2.configure("preview")

  def __enter__(self):
    """Start the camera when entering the with-block."""
    self.picam2.start()
    return self

  def capture(self):
    """Capture a single frame and return it as a NumPy array."""
    frame = self.picam2.capture_array()
    frame = cv2.flip(frame, -1)
    return frame

  def __exit__(self, exception_type, exception_value, exception_traceback):
    self.picam2.stop()
