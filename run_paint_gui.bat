@echo off
setlocal
cd /d "%~dp0"

if defined PAINT_ANALYSIS_HOME (
    set "APP_HOME=%PAINT_ANALYSIS_HOME%"
) else if defined LOCALAPPDATA (
    set "APP_HOME=%LOCALAPPDATA%\PaintAnalysis"
) else (
    set "APP_HOME=%USERPROFILE%\AppData\Local\PaintAnalysis"
)
set "VENV_DIR=%APP_HOME%\venv"
set "PAINT_ANALYSIS_HOME=%APP_HOME%"
set "PYTHONPYCACHEPREFIX=%APP_HOME%\cache"

py -3 -c "import sys; raise SystemExit(sys.version_info ^< (3, 10))" >nul 2>nul
if errorlevel 1 (
    python -c "import sys; raise SystemExit(sys.version_info ^< (3, 10))" >nul 2>nul
    if errorlevel 1 (
        echo Python 3.10 or newer was not found.
        echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
        echo Then run this file again.
        pause
        exit /b 1
    ) else (
        set "PYTHON_CMD=python"
    )
) else (
    set "PYTHON_CMD=py -3"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python environment in:
    echo "%VENV_DIR%"
    if not exist "%APP_HOME%" (
        mkdir "%APP_HOME%"
        if errorlevel 1 (
            echo Failed to create the application data directory.
            pause
            exit /b 1
        )
    )
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create the Python environment.
        pause
        exit /b 1
    )
)

echo Installing or updating required packages...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to update pip.
    pause
    exit /b 1
)
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo Starting PAINT analysis GUI...
"%VENV_DIR%\Scripts\python.exe" paint_analysis_gui.py
pause
