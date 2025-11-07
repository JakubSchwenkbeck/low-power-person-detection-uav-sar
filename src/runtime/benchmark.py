from model import Model
import os
import cv2
import time
from memory_profiler import memory_usage
import numpy as np
import psutil
from tqdm import tqdm
import json
from datetime import datetime

MODEL_PATH = 'models'
IMAGES_PATH = 'images'

P_idle = 1.5
P_max = 4.0

timestamp = datetime.now().strftime("%m_%d_%H%M%S")
OUTPUT_DIR = os.path.join('output', 'benchmarks')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'benchmark_results_{timestamp}.json')

def run(amount_of_images=50):
    results = {
        'images_amount': amount_of_images,
        'results': {
        }
    }

    for model in os.listdir(MODEL_PATH):
        file_path = os.path.join(MODEL_PATH, model)
        if file_path.endswith('.tflite') and get_model_type(file_path):
            print(f'Benchmarking model: {model}')
            results['results'][model] = benchmark(file_path, amount_of_images)
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=4)

def get_model_type(model_path: str) -> str:
    if 'yolo' in model_path:
        return 'yolo'
    elif 'fomo' in model_path:
        return 'fomo'
    else:
        return None

def get_CPU_temp():
    res = os.popen('vcgencmd measure_temp').readline()
    return float(res.replace("temp=","").replace("'C\n",""))

def benchmark(model_path: str, amount_of_images):
    model_type = get_model_type(model_path)
    model = Model(model_type=model_type, path=model_path)

    time_values = []
    memory_values = []
    temp_values = []
    cpu_usage_values = []
    energy_values = []
    
    for image in tqdm(os.listdir(IMAGES_PATH)[:amount_of_images]):
        image_path = os.path.join(IMAGES_PATH, image)
        img = cv2.imread(image_path)

        start_time = time.time()
        model.inference(img, postprocess=False)
        end_time = time.time()


        mem_usage = memory_usage((model.inference, (img, False, )))

        time_values.append((end_time - start_time) * 1000)
        cpu_usage_values.append(psutil.cpu_percent())
        temp_values.append(get_CPU_temp())
        memory_values.append(np.mean(mem_usage))
        energy_values.append(P_idle + (P_max - P_idle) * (cpu_usage_values[-1] / 100))



    return {
        'inference_time (ms)': np.mean(time_values),
        'memory_usage (MiB)': np.mean(memory_values),
        'cpu_temperature (C)': np.mean(temp_values),
        'cpu_usage (%)': np.mean(cpu_usage_values),
        'energy_consumption (W)': np.mean(energy_values)
    }

if __name__ == '__main__':
    run()