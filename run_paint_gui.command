#!/bin/zsh
set -e

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 was not found."
    echo "Install Python 3.10 or newer from https://www.python.org/downloads/macos/"
    echo "Then run this file again."
    read -r "?Press Return to close..."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating local Python environment..."
    python3 -m venv .venv
fi

echo "Installing or updating required packages..."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

echo "Starting PAINT analysis GUI..."
".venv/bin/python" paint_analysis_gui.py

read -r "?PAINT analysis GUI closed. Press Return to close this window..."
