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
import json


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
OUTPUT_DIR = ROOT / 'output' / 'webui'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Model Deployment UI')
app.mount('/static', StaticFiles(directory=ROOT / 'webui' / 'static'), name='static')
# Serve generated outputs (plots, benchmarks, results) under /output
app.mount('/output', StaticFiles(directory=ROOT / 'output'), name='output')

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
    # Import Model lazily to avoid requiring tflite_runtime at web server startup.
    from src.runtime.model import Model
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


@app.get('/inference', response_class=HTMLResponse)
def inference(request: Request):
    """Simple inference page with a model selector and upload form.
    The form posts to /run and uses the existing background runner. This page is intentionally minimal (dummy Run/Benchmark controls).
    """
    models = list_models()
    return templates.TemplateResponse('inference.html', { 'request': request, 'models': models })


@app.post('/run')
def run_model(request: Request, background_tasks: BackgroundTasks, model: str = Form(...), video: UploadFile = File(...), benchmark: str = Form(None), images: int = Form(50)):
    # save uploaded video
    uid = uuid.uuid4().hex
    temp_dir = OUTPUT_DIR / uid
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / video.filename
    with open(video_path, 'wb') as f:
        shutil.copyfileobj(video.file, f)

    output_video = temp_dir / f'out_{video.filename.rsplit('.',1)[0]}.mp4'

    # schedule inference
    background_tasks.add_task(run_inference, model, str(video_path), str(output_video))

    # optionally schedule a benchmark run (runs on all models and writes JSON to output/benchmarks)
    if benchmark:
        def _run_benchmark(amount):
            import importlib
            bmod = importlib.import_module('src.runtime.benchmark')
            try:
                bmod.run(amount_of_images=amount)
            except TypeError:
                # older signature may accept a single positional arg
                bmod.run(amount)

        background_tasks.add_task(_run_benchmark, int(images or 50))

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


@app.get('/compare', response_class=HTMLResponse)
def compare(request: Request, name: str = None):
    bench_dir = ROOT / 'output' / 'benchmarks'
    benchmarks = []
    if bench_dir.exists():
        benchmarks = sorted([p.name for p in bench_dir.iterdir() if p.is_file()])

    data = None
    if name:
        path = bench_dir / name
        if path.exists():
            import json
            with open(path, 'r') as f:
                data = json.load(f)

    return templates.TemplateResponse('compare.html', { 'request': request, 'benchmarks': benchmarks, 'data': data, 'selected': name })


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


@app.post('/bench/run')
def run_single_benchmark(background_tasks: BackgroundTasks, model: str = Form(...), images: int = Form(50)):
    """Start benchmark for a single model and stream progress via SSE (see /bench/stream/{id})."""
    # validate model exists
    model_path = MODELS_DIR / model
    if not model_path.exists() or not model_path.suffix == '.tflite':
        return { 'error': 'model not found or unsupported (expect .tflite file)' }

    run_id = uuid.uuid4().hex
    status_dir = ROOT / 'output' / 'benchmarks' / 'status'
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / f'{run_id}.log'
    result_file = ROOT / 'output' / 'benchmarks' / f'benchmark_{run_id}.json'

    def _write_progress(current, total, interim):
        # append one JSON line per update
        try:
            with open(status_file, 'a') as f:
                f.write(json.dumps({'current': current, 'total': total, 'metrics': interim}) + '\n')
        except Exception:
            pass

    def _bench_task(mpath, amount, status_p, result_p):
        import importlib
        bmod = importlib.import_module('src.runtime.benchmark')
        try:
            res = bmod.benchmark(str(mpath), int(amount), progress_callback=_write_progress)
            # write final result JSON
            with open(result_p, 'w') as f:
                json.dump({'model': mpath.name, 'result': res}, f, indent=4)
            # notify completion
            with open(status_p, 'a') as f:
                f.write(json.dumps({'finished': True, 'result_file': result_p.name}) + '\n')
        except Exception as e:
            with open(status_p, 'a') as f:
                f.write(json.dumps({'error': str(e)}) + '\n')

    # schedule background benchmark task
    background_tasks.add_task(_bench_task, model_path, images, status_file, result_file)

    return { 'run_id': run_id, 'stream_url': f'/bench/stream/{run_id}' }


@app.get('/bench/stream/{run_id}')
def bench_stream(run_id: str):
    """Server-Sent Events stream that tails the status log for a run id."""
    from fastapi.responses import StreamingResponse

    status_dir = ROOT / 'output' / 'benchmarks' / 'status'
    status_file = status_dir / f'{run_id}.log'

    def event_generator(path):
        # wait for file to be created
        import time as _time
        while not path.exists():
            _time.sleep(0.1)

        with open(path, 'r') as f:
            # yield existing lines
            for line in f:
                yield f'data: {line.strip()}\n\n'

            # then tail
            while True:
                where = f.tell()
                line = f.readline()
                if not line:
                    _time.sleep(0.2)
                    f.seek(where)
                else:
                    yield f'data: {line.strip()}\n\n'

    return StreamingResponse(event_generator(status_file), media_type='text/event-stream')
