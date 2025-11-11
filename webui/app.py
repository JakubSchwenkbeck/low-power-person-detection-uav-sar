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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'runtime'))
from model import Model

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "webui" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

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

@app.get("/", response_class=HTMLResponse)
async def home():
    return open(Path(__file__).parent / "static" / "index.html").read()

@app.get("/api/models")
async def get_models():
    return {"models": get_available_models()}

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
        model = load_model(config["model_path"])
        cap = cv2.VideoCapture(config["video_path"])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        inference_times = []
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            result_frame = model.inference(frame.copy(), postprocess=True, 
                                          conf_threshold=config.get("conf_threshold", 0.5),
                                          nms_threshold=config.get("nms_threshold", 0.45))
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            display_frame = result_frame if result_frame is not None else frame
            if result_frame is not None:
                detection_count += 1
            
            await websocket.send_json({
                "frame": frame_to_base64(display_frame),
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "inference_time": f"{inference_time:.2f}",
                "avg_inference_time": f"{np.mean(inference_times):.2f}",
                "progress": (frame_idx + 1) / total_frames * 100
            })
            
            frame_idx += 1
            await asyncio.sleep(0.01)
        
        cap.release()
        await websocket.send_json({
            "complete": True,
            "total_frames": frame_idx,
            "frames_with_detections": detection_count,
            "avg_inference_time": f"{np.mean(inference_times):.2f}",
            "total_time": f"{sum(inference_times) / 1000:.2f}"
        })
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
