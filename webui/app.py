"""
Modern Web UI for UAV Person Detection
FastAPI backend with WebSocket support for real-time inference
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import sys
import json
import asyncio
import base64
from pathlib import Path
from typing import Optional
import time
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'runtime'))
from model import Model

app = FastAPI(title="UAV Person Detection", description="Real-time person detection for SAR operations")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "webui" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Global state
active_models = {}


def get_available_models():
    """Get list of available TFLite models"""
    models = []
    try:
        for model_file in MODELS_DIR.glob("*.tflite"):
            model_info = {
                "name": model_file.name,
                "path": str(model_file),
                "size": f"{model_file.stat().st_size / (1024 * 1024):.2f} MB"
            }
            models.append(model_info)
        logger.info(f"Found {len(models)} models")
    except Exception as e:
        logger.error(f"Error scanning models directory: {e}")
    return models


def load_model(model_path: str):
    """Load a model and cache it"""
    if model_path not in active_models:
        logger.info(f"Loading model: {model_path}")
        model = Model(model_type='yolo', path=model_path)
        active_models[model_path] = model
        logger.info(f"Model loaded successfully")
    return active_models[model_path]


def frame_to_base64(frame):
    """Convert frame to base64 encoded JPEG"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_available": len(get_available_models()),
        "active_models": len(active_models)
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "static" / "index.html"
    with open(html_path, "r") as f:
        return f.read()


@app.get("/api/models")
async def get_models():
    """Get list of available models"""
    return {"models": get_available_models()}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file"""
    logger.info(f"Receiving upload: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: mp4, avi, mov, mkv, webm")
    
    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    # Get video info
    try:
        cap = cv2.VideoCapture(str(file_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"Video info: {width}x{height}, {fps} fps, {frame_count} frames")
    except Exception as e:
        logger.error(f"Error reading video info: {e}")
        raise HTTPException(status_code=500, detail="Failed to read video information")
    
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
    """WebSocket endpoint for real-time inference"""
    await websocket.accept()
    
    try:
        # Receive configuration
        config = await websocket.receive_json()
        video_path = config.get("video_path")
        model_path = config.get("model_path")
        conf_threshold = config.get("conf_threshold", 0.5)
        nms_threshold = config.get("nms_threshold", 0.45)
        
        if not video_path or not os.path.exists(video_path):
            await websocket.send_json({"error": "Invalid video path"})
            return
        
        if not model_path or not os.path.exists(model_path):
            await websocket.send_json({"error": "Invalid model path"})
            return
        
        # Load model
        model = load_model(model_path)
        logger.info(f"Processing with conf_threshold={conf_threshold}, nms_threshold={nms_threshold}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        inference_times = []
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference with custom thresholds
            start_time = time.time()
            result_frame = model.inference(
                frame.copy(), 
                postprocess=True,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)
            
            # Use original frame if no detections
            if result_frame is not None:
                display_frame = result_frame
                detection_count += 1
            else:
                display_frame = frame
            
            # Convert to base64
            frame_b64 = frame_to_base64(display_frame)
            
            # Send frame data
            await websocket.send_json({
                "frame": frame_b64,
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "inference_time": f"{inference_time:.2f}",
                "avg_inference_time": f"{np.mean(inference_times):.2f}",
                "progress": (frame_idx + 1) / total_frames * 100
            })
            
            frame_idx += 1
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        cap.release()
        
        # Send completion message
        logger.info(f"Inference complete: {detection_count}/{frame_idx} frames had detections")
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


@app.get("/api/video/{filename}")
async def get_video(filename: str):
    """Stream video file"""
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    def iter_file():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):  # 1MB chunks
                yield chunk
    
    return StreamingResponse(iter_file(), media_type="video/mp4")


@app.delete("/api/uploads")
async def clear_uploads():
    """Clear all uploaded files"""
    for file in UPLOAD_DIR.glob("*"):
        if file.is_file():
            file.unlink()
    return {"message": "All uploads cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
