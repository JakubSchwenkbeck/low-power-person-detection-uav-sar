from model import Model
import os
import cv2
import time

MODEL_PATH = 'data/models'
IMAGES_PATH = 'data/images/val2017'

def main():
    for model in os.listdir(MODEL_PATH):
        file_path = os.path.join(MODEL_PATH, model)
        if file_path.endswith('.tflite'):
            benchmark(file_path)

def get_model_type(model_path: str) -> str:
    if 'yolo' in model_path:
        return 'yolo'
    elif 'fomo' in model_path:
        return 'fomo'
    elif 'mobilenet' in model_path:
        return 'mobilenet'
    else:
        return None

def getCPUtemperature():
    res = os.popen('vcgencmd measure_temp').readline()
    return(res.replace("temp=","").replace("'C\n",""))

def getRAMinfo():
    p = os.popen('free')
    i = 0
    while 1:
        i = i + 1
        line = p.readline()
        if i==2:
            return(line.split()[1:4])

#def getCPUuse():
#    return(str(os.popen("top -n1 | awk '/Cpu\(s\):/ {print $2}'").readline().strip(\)))


def benchmark(model_path: str):

    model_type = get_model_type(model_path)
    if not model_type:
        print(f"Unknown model type for model: {model_path}")
        return

    model = Model(model_type=model_type, path=model_path)
    
    for image in os.listdir(IMAGES_PATH):
        image_path = os.path.join(IMAGES_PATH, image)
        img = cv2.imread(image_path)

        start_time = time.time()
        model.inference(img, postprocess=False)
        end_time = time.time()

        inference_time = end_time - start_time
        cpu_temp = getCPUtemperature()
        ram_info = getRAMinfo()

        print(f"Model: {model_path}, Image: {image}, Inference Time: {inference_time:.4f} seconds")
        print(f"CPU Temperature: {cpu_temp} °C")
        print(f"RAM Info (total, used, free in KB): {ram_info}")



main()
