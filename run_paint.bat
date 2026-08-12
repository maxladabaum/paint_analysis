@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "MIN_PYTHON=3.10"
set "BOOTSTRAP_PYTHON=3.12.10"

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

call :find_python
if not defined PYTHON_CMD (
    echo A compatible Python %MIN_PYTHON% through 3.14 installation was not found.
    echo Installing Python %BOOTSTRAP_PYTHON% alongside any existing Python versions...
    call :install_python
    if errorlevel 1 goto :python_install_failed

    call :find_python
    if not defined PYTHON_CMD goto :python_install_failed
)

echo Using Python: %PYTHON_CMD% %PYTHON_ARGS%
if not exist "%APP_HOME%" (
    mkdir "%APP_HOME%"
    if errorlevel 1 goto :app_home_failed
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python environment in:
    echo "%VENV_DIR%"
    "%PYTHON_CMD%" %PYTHON_ARGS% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :venv_failed
) else (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 10) ^<= sys.version_info[:2] ^< (3, 15) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Rebuilding an incompatible application environment...
        rmdir /s /q "%VENV_DIR%"
        "%PYTHON_CMD%" %PYTHON_ARGS% -m venv "%VENV_DIR%"
        if errorlevel 1 goto :venv_failed
    )
)

echo Installing or updating required packages...
"%VENV_DIR%\Scripts\python.exe" -m pip --version >nul 2>nul
if errorlevel 1 (
    "%VENV_DIR%\Scripts\python.exe" -m ensurepip --upgrade
    if errorlevel 1 goto :dependency_failed
)
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto :dependency_failed
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
if errorlevel 1 goto :dependency_failed
"%VENV_DIR%\Scripts\python.exe" -m pip check
if errorlevel 1 goto :dependency_failed

echo Checking the application installation...
"%VENV_DIR%\Scripts\python.exe" -c "import h5py, matplotlib, numpy, pandas, picasso, scipy, sklearn, tifffile, tkinter, tqdm, yaml"
if errorlevel 1 goto :dependency_failed

echo Starting PAINT analysis GUI...
"%VENV_DIR%\Scripts\python.exe" "%~dp0paint_analysis_gui.py"
set "APP_EXIT=%ERRORLEVEL%"
echo.
if not "%APP_EXIT%"=="0" echo PAINT analysis exited with error code %APP_EXIT%.
pause
exit /b %APP_EXIT%

:find_python
set "PYTHON_CMD="
set "PYTHON_ARGS="
py -3.12 -c "import ensurepip, tkinter, venv, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py"
    set "PYTHON_ARGS=-3.12"
    exit /b 0
)
py -3 -c "import ensurepip, tkinter, venv, sys; raise SystemExit(0 if (3, 10) ^<= sys.version_info[:2] ^< (3, 15) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py"
    set "PYTHON_ARGS=-3"
    exit /b 0
)
python -c "import ensurepip, tkinter, venv, sys; raise SystemExit(0 if (3, 10) ^<= sys.version_info[:2] ^< (3, 15) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do call :consider_python "%%~fD\python.exe"
for /d %%D in ("%ProgramFiles%\Python3*") do call :consider_python "%%~fD\python.exe"
for /d %%D in ("%ProgramFiles(x86)%\Python3*") do call :consider_python "%%~fD\python.exe"
exit /b 0

:consider_python
if defined PYTHON_CMD exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import ensurepip, tkinter, venv, sys; raise SystemExit(0 if (3, 10) ^<= sys.version_info[:2] ^< (3, 15) else 1)" >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=%~1"
exit /b 0

:install_python
set "PYTHON_ARCH="
if /i "%PROCESSOR_ARCHITECTURE%"=="AMD64" set "PYTHON_ARCH=-amd64"
if /i "%PROCESSOR_ARCHITEW6432%"=="AMD64" set "PYTHON_ARCH=-amd64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "PYTHON_ARCH=-arm64"
if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "PYTHON_ARCH=-arm64"
set "PYTHON_INSTALLER=%TEMP%\paint-analysis-python-%BOOTSTRAP_PYTHON%.exe"
set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/%BOOTSTRAP_PYTHON%/python-%BOOTSTRAP_PYTHON%%PYTHON_ARCH%.exe"

echo Downloading Python from python.org...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri $env:PYTHON_INSTALLER_URL -OutFile $env:PYTHON_INSTALLER"
if errorlevel 1 exit /b 1

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "if ((Get-AuthenticodeSignature -LiteralPath $env:PYTHON_INSTALLER).Status -ne 'Valid') { exit 1 }"
if errorlevel 1 (
    del /q "%PYTHON_INSTALLER%" >nul 2>nul
    echo The downloaded Python installer did not have a valid digital signature.
    exit /b 1
)

start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_pip=1 Include_test=0 Include_tcltk=1
set "INSTALL_EXIT=%ERRORLEVEL%"
del /q "%PYTHON_INSTALLER%" >nul 2>nul
if not "%INSTALL_EXIT%"=="0" exit /b 1
exit /b 0

:python_install_failed
echo.
echo Python could not be installed automatically.
echo Your existing Python installation has not been changed.
echo Check your internet connection, or install Python %MIN_PYTHON% through 3.14 from:
echo https://www.python.org/downloads/windows/
echo Then run this file again.
pause
exit /b 1

:app_home_failed
echo.
echo Failed to create the application data directory:
echo "%APP_HOME%"
pause
exit /b 1

:venv_failed
echo.
echo Failed to create or update the Python environment:
echo "%VENV_DIR%"
pause
exit /b 1

:dependency_failed
echo.
echo Failed to install or verify the required Python packages.
echo Check the messages above and your internet connection, then run this file again.
pause
exit /b 1
