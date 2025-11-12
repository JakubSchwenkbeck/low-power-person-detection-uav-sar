from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import sys
import asyncio
import base64
from pathlib import Path
import time
import shutil
import psutil
from memory_profiler import memory_usage
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'runtime'))
from model import Model

P_IDLE = 1.5
P_MAX = 4.0

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "webui" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
BENCHMARK_DIR = BASE_DIR / "output" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = BASE_DIR / "output" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/plots", StaticFiles(directory=PLOT_DIR), name="plots")
app.mount("/benchmarks", StaticFiles(directory=BENCHMARK_DIR), name="benchmarks")

active_models = {}

def get_available_models():
    return [{"name": f.name, "path": str(f), "size": f"{f.stat().st_size / (1024 * 1024):.2f} MB"} 
            for f in MODELS_DIR.glob("*.tflite")]

def load_model(model_path: str):
    if model_path not in active_models:
        active_models[model_path] = Model(model_type='yolo', path=model_path)
    return active_models[model_path]

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def get_cpu_temp():
    try:
        res = os.popen('vcgencmd measure_temp').readline()
        return float(res.replace("temp=","").replace("'C\n",""))
    except:
        # Fallback for non-Raspberry Pi systems
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
        except:
            pass
        return None

@app.get("/", response_class=HTMLResponse)
async def home():
    return open(Path(__file__).parent / "static" / "index.html").read()

@app.get("/api/models")
async def get_models():
    return {"models": get_available_models()}

@app.get("/api/benchmarks")
async def get_benchmarks():
    benchmarks = []
    for f in BENCHMARK_DIR.glob("benchmark_*.json"):
        with open(f, 'r') as file:
            data = json.load(file)
            benchmarks.append({
                "filename": f.name,
                "path": str(f),
                "timestamp": data.get("timestamp"),
                "model": data.get("model"),
                "video": data.get("video"),
                "avg_inference_time": data["results"].get("avg_inference_time_ms")
            })
    benchmarks.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"benchmarks": benchmarks}

@app.post("/api/plot_benchmark")
async def create_plot(data: dict):
    import subprocess
    benchmark_file = data.get("benchmark_file")
    if not benchmark_file or not Path(benchmark_file).exists():
        raise HTTPException(status_code=404, detail="Benchmark file not found")
    
    # Run plotting script
    plot_script = Path(__file__).parent / "plot_benchmark.py"
    result = subprocess.run(
        [sys.executable, str(plot_script), benchmark_file],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Extract plot filename from output
        plot_file = result.stdout.strip().split(": ")[-1]
        return {"plot_file": plot_file, "success": True}
    else:
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {result.stderr}")

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "filename": file.filename,
        "path": str(file_path),
        "fps": fps,
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": f"{frame_count / fps:.2f}s" if fps > 0 else "unknown"
    }


@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    try:
        config = await websocket.receive_json()
        print(f"[DEBUG] Received config: {config}")
        
        model = load_model(config["model_path"])
        cap = cv2.VideoCapture(config["video_path"])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        benchmark_mode = config.get("benchmark", False)
        
        print(f"[DEBUG] Benchmark mode: {benchmark_mode}")
        print(f"[DEBUG] Total frames: {total_frames}")
        
        frame_idx = 0
        inference_times = []
        detection_count = 0
        memory_values = []
        temp_values = []
        cpu_usage_values = []
        energy_values = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            if benchmark_mode:
                print(f"[DEBUG] Frame {frame_idx}: Running in benchmark mode")
                cpu_before = psutil.cpu_percent(interval=None)
                mem_usage = memory_usage((model.inference, (frame.copy(), True, config.get("conf_threshold", 0.5), config.get("nms_threshold", 0.45))), interval=0.01, max_usage=True)
                result_frame = model.inference(frame.copy(), postprocess=True, 
                                              conf_threshold=config.get("conf_threshold", 0.5),
                                              nms_threshold=config.get("nms_threshold", 0.45))
                cpu_after = psutil.cpu_percent(interval=None)
                cpu_usage = (cpu_before + cpu_after) / 2
                
                temp = get_cpu_temp()
                if temp:
                    temp_values.append(temp)
                    print(f"[DEBUG] Frame {frame_idx}: Temp={temp}°C")
                
                memory_values.append(mem_usage)
                cpu_usage_values.append(cpu_usage)
                energy = P_IDLE + (P_MAX - P_IDLE) * (cpu_usage / 100)
                energy_values.append(energy)
                
                print(f"[DEBUG] Frame {frame_idx}: Memory={mem_usage:.2f} MiB, CPU={cpu_usage:.1f}%, Energy={energy:.2f}W")
            else:
                result_frame = model.inference(frame.copy(), postprocess=True, 
                                              conf_threshold=config.get("conf_threshold", 0.5),
                                              nms_threshold=config.get("nms_threshold", 0.45))
            
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            display_frame = result_frame if result_frame is not None else frame
            if result_frame is not None:
                detection_count += 1
            
            response = {
                "frame": frame_to_base64(display_frame),
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "inference_time": f"{inference_time:.2f}",
                "avg_inference_time": f"{np.mean(inference_times):.2f}",
                "progress": (frame_idx + 1) / total_frames * 100
            }
            
            if benchmark_mode and len(memory_values) > 0:
                response.update({
                    "memory_usage": f"{np.mean(memory_values):.2f}",
                    "cpu_usage": f"{np.mean(cpu_usage_values):.2f}",
                    "energy": f"{np.mean(energy_values):.2f}",
                })
                if temp_values:
                    response["temperature"] = f"{np.mean(temp_values):.2f}"
            
            await websocket.send_json(response)
            
            frame_idx += 1
            await asyncio.sleep(0.01)
        
        cap.release()
        
        print(f"[DEBUG] Inference complete. Total frames: {frame_idx}")
        print(f"[DEBUG] Benchmark mode was: {benchmark_mode}")
        print(f"[DEBUG] Memory values collected: {len(memory_values)}")
        
        result = {
            "complete": True,
            "total_frames": frame_idx,
            "frames_with_detections": detection_count,
            "avg_inference_time": f"{np.mean(inference_times):.2f}",
            "total_time": f"{sum(inference_times) / 1000:.2f}"
        }
        
        if benchmark_mode and len(memory_values) > 0:
            print(f"[DEBUG] Creating benchmark results...")
            benchmark_data = {
                "avg_memory_usage_MiB": f"{np.mean(memory_values):.2f}",
                "avg_cpu_usage_percent": f"{np.mean(cpu_usage_values):.2f}",
                "avg_energy_consumption_W": f"{np.mean(energy_values):.2f}",
            }
            if temp_values:
                benchmark_data["avg_temperature_C"] = f"{np.mean(temp_values):.2f}"
            
            result["benchmark"] = benchmark_data
            print(f"[DEBUG] Benchmark data added to result")
            
            # Save full benchmark results to JSON
            timestamp = datetime.now().strftime("%m_%d_%H%M%S")
            model_name = Path(config["model_path"]).stem
            video_name = Path(config["video_path"]).stem
            
            print(f"[DEBUG] Model: {model_name}, Video: {video_name}, Timestamp: {timestamp}")
            
            full_benchmark = {
                "timestamp": timestamp,
                "model": model_name,
                "video": video_name,
                "config": {
                    "conf_threshold": config.get("conf_threshold", 0.5),
                    "nms_threshold": config.get("nms_threshold", 0.45)
                },
                "results": {
                    "total_frames": frame_idx,
                    "frames_with_detections": detection_count,
                    "inference_times_ms": [float(t) for t in inference_times],
                    "avg_inference_time_ms": float(np.mean(inference_times)),
                    "min_inference_time_ms": float(np.min(inference_times)),
                    "max_inference_time_ms": float(np.max(inference_times)),
                    "std_inference_time_ms": float(np.std(inference_times)),
                    "total_time_s": float(sum(inference_times) / 1000),
                    "memory_usage_MiB": [float(m) for m in memory_values],
                    "avg_memory_usage_MiB": float(np.mean(memory_values)),
                    "cpu_usage_percent": [float(c) for c in cpu_usage_values],
                    "avg_cpu_usage_percent": float(np.mean(cpu_usage_values)),
                    "energy_consumption_W": [float(e) for e in energy_values],
                    "avg_energy_consumption_W": float(np.mean(energy_values)),
                }
            }
            
            if temp_values:
                full_benchmark["results"]["temperature_C"] = [float(t) for t in temp_values]
                full_benchmark["results"]["avg_temperature_C"] = float(np.mean(temp_values))
            
            benchmark_file = BENCHMARK_DIR / f"benchmark_{model_name}_{video_name}_{timestamp}.json"
            print(f"[DEBUG] Saving benchmark to: {benchmark_file}")
            
            try:
                with open(benchmark_file, 'w') as f:
                    json.dump(full_benchmark, f, indent=4)
                print(f"[DEBUG] ✓ Benchmark file saved successfully!")
                print(f"[DEBUG] File size: {benchmark_file.stat().st_size} bytes")
            except Exception as e:
                print(f"[ERROR] Failed to save benchmark file: {e}")
            
            result["benchmark_file"] = str(benchmark_file)
            print(f"[DEBUG] Benchmark file path added to result: {result['benchmark_file']}")
        else:
            print(f"[DEBUG] NOT saving benchmark - mode={benchmark_mode}, memory_values={len(memory_values)}")
        
        print(f"[DEBUG] Sending final result: {result.keys()}")
        await websocket.send_json(result)
    except Exception as e:
        print(f"[ERROR] Exception in websocket_inference: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
