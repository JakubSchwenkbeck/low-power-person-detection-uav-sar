Web UI for model deployment

Quickstart

1. Install Python deps (prefer a virtualenv):

   python -m pip install -r requirements.txt

2. Run the server (from repository root):

   python -m uvicorn webui.app:app --reload --host 127.0.0.1 --port 8000

3. Open http://127.0.0.1:8000 in your browser.

Notes

- The index page only needs FastAPI/Jinja to render; running inference requires the project's runtime deps (tflite runtime, edge_impulse, opencv).
- Uploaded videos are stored under `output/webui/<uuid>/` and output videos are written there as well.
- This is an initial, minimal interface. Next steps: progress/status API, streaming preview, benchmark trigger endpoints and visualizations, auth, and background worker (Redis/Celery) for heavy jobs.
