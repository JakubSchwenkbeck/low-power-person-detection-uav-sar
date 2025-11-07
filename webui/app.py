from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import uuid
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
OUTPUT_DIR = ROOT / 'output' / 'webui'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount('/static', StaticFiles(directory=ROOT / 'webui' / 'static'), name='static')
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





@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    models = list_models()
    return templates.TemplateResponse('index.html', { 'request': request, 'models': models })


@app.get('/inference', response_class=HTMLResponse)
def inference(request: Request):
    
    models = list_models()
    return templates.TemplateResponse('inference.html', { 'request': request, 'models': models })


@app.post('/run')
def run_model(request: Request, model: str = Form(...), video: UploadFile = File(...), benchmark: str = Form(None)):
    uid = uuid.uuid4().hex
    temp_dir = OUTPUT_DIR / uid
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / video.filename
    with open(video_path, 'wb') as f:
        shutil.copyfileobj(video.file, f)
    output_video = temp_dir / f'out_{video.filename.rsplit(".",1)[0]}.mp4'
    shutil.copy2(str(video_path), str(output_video))
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
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = None

    plots = []
    plots_dir = ROOT / 'output' / 'plots'
    if plots_dir.exists():
        plots = sorted([p.name for p in plots_dir.iterdir() if p.is_file()])

    return templates.TemplateResponse('compare.html', { 'request': request, 'benchmarks': benchmarks, 'data': data, 'selected': name, 'plots': plots })

