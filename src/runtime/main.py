
from model import YoloModel
from camera import Camera
import cv2
import os

if __name__ == '__main__':
    model = YoloModel(path='./data/models/yolo11n_latency_dynamic.tflite')


    for filename in os.listdir('temp'):
        file_path = os.path.join('temp', filename)
        os.remove(file_path)
    
    with Camera() as cam:
        i = 0
        while True:
            frame = cam.capture()
            image = model.inference(frame)

            if image is not None:
                cv2.imwrite(f"temp/output{i}.jpg", image)
                i += 1
                print(f"Person detected! Saved output{i}.jpg")
            
            else:
                print("No person detected in this frame.")
