from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid
from pathlib import Path
import time
import cv2
import numpy as np

from src.runtime.model import Model

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
OUTPUT_DIR = ROOT / 'output' / 'webui'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Model Deployment UI')
app.mount('/static', StaticFiles(directory=ROOT / 'webui' / 'static'), name='static')
templates = Jinja2Templates(directory=ROOT / 'webui' / 'templates')


def list_models():
    files = []
    for p in MODELS_DIR.iterdir():
        if p.is_file() and p.suffix in ['.tflite', '.pb']:
            files.append(p.name)
        elif p.is_dir():
            files.append(p.name + '/')
    return sorted(files)


def run_inference(model_path: str, video_path: str, output_path: str):
    # Use the existing Model class to perform inference on a video file and write an output video
    model_file = str(MODELS_DIR / model_path)
    model = Model(model_type='yolo', path=model_file)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            start = time.time()
            res = model.inference(frame)
            # If postprocess returned an annotated image, use it
            annotated = frame if res is None else res
            out.write(annotated)
        except Exception:
            # on error, write original frame
            out.write(frame)

    cap.release()
    out.release()


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    models = list_models()
    benchmarks = []
    plots = []
    bench_dir = ROOT / 'output' / 'benchmarks'
    plots_dir = ROOT / 'output' / 'plots'
    if bench_dir.exists():
        benchmarks = sorted([p.name for p in bench_dir.iterdir() if p.is_file()])
    if plots_dir.exists():
        plots = sorted([p.name for p in plots_dir.iterdir() if p.is_file()])

    return templates.TemplateResponse('index.html', { 'request': request, 'models': models, 'benchmarks': benchmarks, 'plots': plots })


@app.post('/run')
def run_model(request: Request, background_tasks: BackgroundTasks, model: str = Form(...), video: UploadFile = File(...)):
    # save uploaded video
    uid = uuid.uuid4().hex
    temp_dir = OUTPUT_DIR / uid
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / video.filename
    with open(video_path, 'wb') as f:
        shutil.copyfileobj(video.file, f)

    output_video = temp_dir / f'out_{video.filename.rsplit('.',1)[0]}.mp4'

    background_tasks.add_task(run_inference, model, str(video_path), str(output_video))

    return RedirectResponse(url=f'/result/{uid}', status_code=303)


@app.get('/result/{uid}', response_class=HTMLResponse)
def result(request: Request, uid: str):
    temp_dir = OUTPUT_DIR / uid
    if not temp_dir.exists():
        return templates.TemplateResponse('result.html', {'request': request, 'ready': False, 'path': None})

    files = list(temp_dir.iterdir())
    out_video = next((p for p in files if p.suffix == '.mp4'), None)
    ready = out_video is not None and out_video.exists()
    return templates.TemplateResponse('result.html', {'request': request, 'ready': ready, 'path': out_video.name if ready else None, 'uid': uid})


@app.get('/download/{uid}/{filename}')
def download(uid: str, filename: str):
    file_path = OUTPUT_DIR / uid / filename
    if file_path.exists():
        return FileResponse(path=str(file_path), filename=filename, media_type='application/octet-stream')
    return { 'error': 'not found' }


@app.get('/bench/{name}')
def get_benchmark(name: str):
    bench_file = ROOT / 'output' / 'benchmarks' / name
    if bench_file.exists():
        return FileResponse(str(bench_file), media_type='application/json')
    return { 'error': 'not found' }
