#!/bin/bash
cd "$(dirname "$0")"
[ ! -d "../tinyml" ] && exit 1
source ../tinyml/bin/activate 2>/dev/null || source ../tinyml/bin/activate.fish
python -c "import fastapi" 2>/dev/null || pip install -q -r requirements.txt
python run.py
