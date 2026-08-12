#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_SUPPORT_DIR="${PAINT_ANALYSIS_HOME:-${HOME}/Library/Application Support/PaintAnalysis}"
VENV_DIR="${APP_SUPPORT_DIR}/venv"
export PAINT_ANALYSIS_HOME="$APP_SUPPORT_DIR"
export PYTHONPYCACHEPREFIX="${APP_SUPPORT_DIR}/cache"

cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
    echo "Python 3.10 or newer was not found."
    echo "Install Python 3.10 or newer from https://www.python.org/downloads/macos/"
    echo "Then run this file again."
    read -r "?Press Return to close..."
    exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating Python environment in:"
    echo "$VENV_DIR"
    if ! mkdir -p "$APP_SUPPORT_DIR" || ! python3 -m venv "$VENV_DIR"; then
        echo "Failed to create the Python environment."
        read -r "?Press Return to close..."
        exit 1
    fi
fi

echo "Installing or updating required packages..."
if ! "$VENV_DIR/bin/python" -m pip install --upgrade pip || ! "$VENV_DIR/bin/python" -m pip install -r requirements.txt; then
    echo "Failed to install dependencies."
    read -r "?Press Return to close..."
    exit 1
fi

echo "Starting PAINT analysis GUI..."
"$VENV_DIR/bin/python" paint_analysis_gui.py

read -r "?PAINT analysis GUI closed. Press Return to close this window..."
