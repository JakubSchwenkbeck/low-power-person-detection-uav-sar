from model import Model
from camera import FrameSource

import cv2
import os
import time
from moviepy import *
import numpy as np

if __name__ == '__main__':
    # model = Model(model_type='yolo', path='./data/models/yolo11n_latency_dynamic.tflite')
    model = Model(model_type='fomo', path='./data/models/tinyml-linux-aarch64-v1-int8.eim')
    
    with FrameSource(path='data/images/remote_drone_footage.mp4') as src:
        i = 0
        frames = []
        durations = []
        try: 
            while True:
                frame = src.capture()


                start_time = time.time()
                
                image = model.inference(frame)
                end_time = time.time()
                inference_time = (end_time - start_time)

                
                image = frame.copy() if image is None else image
                image_path = f'temp/frame_{i}.jpg'
                cv2.imwrite(image_path, image)
                i+=1

                frames.append(image_path)
                durations.append(inference_time)


        except KeyboardInterrupt:
            print("Interrupted — building video...")




        clip = ImageSequenceClip(frames, fps=1/np.mean(durations))
        clip.write_videofile("temp/output_variable_timing.mp4", codec="libx264")

        for filename in os.listdir('temp'):
            file_path = os.path.join('temp', filename)
            if file_path.endswith('.jpg'):
                os.remove(file_path)    


