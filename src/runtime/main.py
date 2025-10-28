from model import Model
from camera import FrameSource

import cv2
import os
import time
from moviepy import *
import numpy as np
import argparse
import sys


def run(model_path: str, video_path: str, output_video_path: str):

    model = Model(model_type='yolo', path=model_path)

    with FrameSource(path=video_path) as src:
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

                # print(f"Inference time: {inference_time*1000:.2f} ms")
                
                image = frame.copy() if image is None else image
                image_path = f'temp/frame_{i}.jpg'
                cv2.imwrite(image_path, image)
                i+=1

                frames.append(image_path)
                durations.append(inference_time)

                # cv2.imshow("YOLO", image)
                # cv2.waitKey(1)

        except:
            print("Interrupted — building video...")
        

        clip = ImageSequenceClip(frames, fps=1/np.mean(durations))
        clip.write_videofile(output_video_path, codec="libx264")

        for filename in os.listdir('temp'):
            file_path = os.path.join('temp', filename)
            if file_path.endswith('.jpg'):
                os.remove(file_path)    


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("model-path", help="Path to the model file")
    parser.add_argument("video-path", help="Path to the input video file")
    parser.add_argument("output-video-path", help="Path to the output video file")

    args = parser.parse_args(argv)

    run(args.model_path, args.video_path, args.output_video_path)

if __name__ == '__main__':
    main(sys.argv[1:])
