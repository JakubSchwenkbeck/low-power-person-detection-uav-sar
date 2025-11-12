#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "runtime"))

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("UAV Person Detection - Web UI")
    print("http://0.0.0.0:8000")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
