#!/bin/bash

# UAV Person Detection Web UI Launcher
# This script starts the web interface for real-time inference

cd "$(dirname "$0")"

echo "======================================"
echo "UAV Person Detection - Web UI"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "../tinyml" ]; then
    echo "Error: Virtual environment not found at ../tinyml"
    echo "Please create a virtual environment and install dependencies first."
    exit 1
fi

# Activate virtual environment (works for both bash and fish)
if [ -n "$BASH_VERSION" ]; then
    source ../tinyml/bin/activate
elif [ -n "$FISH_VERSION" ]; then
    source ../tinyml/bin/activate.fish
else
    # Try bash activation by default
    source ../tinyml/bin/activate
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

# Start the server
echo "Starting web server..."
echo "Open your browser and navigate to: http://localhost:8000"
echo ""
python run.py
