@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    py --version >nul 2>nul
    if errorlevel 1 (
        echo Python was not found on PATH.
        echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
        echo During install, enable "Add python.exe to PATH", then run this file again.
        pause
        exit /b 1
    ) else (
        set PYTHON=py
    )
) else (
    set PYTHON=python
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

echo Installing or updating required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo Starting PAINT analysis GUI...
".venv\Scripts\python.exe" paint_analysis_gui.py
pause
