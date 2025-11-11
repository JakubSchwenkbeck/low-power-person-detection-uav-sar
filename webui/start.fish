#!/usr/bin/env fish

# UAV Person Detection Web UI Launcher (Fish Shell)
# This script starts the web interface for real-time inference

cd (dirname (status --current-filename))

echo "======================================"
echo "UAV Person Detection - Web UI"
echo "======================================"
echo ""

# Check if virtual environment exists
if not test -d "../tinyml"
    echo "Error: Virtual environment not found at ../tinyml"
    echo "Please create a virtual environment and install dependencies first."
    exit 1
end

# Activate virtual environment
source ../tinyml/bin/activate.fish

# Check if dependencies are installed
if not python -c "import fastapi" 2>/dev/null
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
end

# Start the server
echo "Starting web server..."
echo "Open your browser and navigate to: http://localhost:8000"
echo ""
python run.py
