#!/usr/bin/env python3
"""
Launcher script for UAV Person Detection Web UI
"""

import os
import sys
from pathlib import Path

# Add parent src directory to path
parent_dir = Path(__file__).resolve().parent.parent
src_runtime = parent_dir / "src" / "runtime"
sys.path.insert(0, str(src_runtime))

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("UAV Person Detection - SAR System")
    print("Web UI Server")
    print("=" * 60)
    print()
    print("Starting server on http://0.0.0.0:8000")
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
