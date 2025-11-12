#!/usr/bin/env fish
cd (dirname (status --current-filename))
test -d "../tinyml"; or exit 1
source ../tinyml/bin/activate.fish
python -c "import fastapi" 2>/dev/null; or pip install -q -r requirements.txt
python run.py
