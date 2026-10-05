@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "BRAND=Navine AI - Python"
set "PKG=navine"
set "PORT=8766"
set "LAUNCHER="

if exist "%ROOT%\venv\Scripts\python.exe" (
    set "PY=%ROOT%\venv\Scripts\python.exe"
) else (
    set "PY=python"
)

for %%F in ("%ROOT%\scripts\*-launcher.ps1") do (
    if not defined LAUNCHER set "LAUNCHER=%%~fF"
)
if not defined LAUNCHER if exist "%ROOT%\scripts\navine-launcher.ps1" set "LAUNCHER=%ROOT%\scripts\navine-launcher.ps1"

set "WEB_URL=http://127.0.0.1:%PORT%"

if "%~1"=="" goto :launch_gui
set "CMD=%~1"
if /i "%CMD%"=="gui" goto :launch_gui
if /i "%CMD%"=="menu" goto :launch_gui
if /i "%CMD%"=="help" goto :show_help
if /i "%CMD%"=="-h" goto :show_help
if /i "%CMD%"=="--help" goto :show_help
if /i "%CMD%"=="ui" goto :web
if /i "%CMD%"=="web" goto :web
if /i "%CMD%"=="server" goto :server
if /i "%CMD%"=="serve" goto :server
if /i "%CMD%"=="setup" goto :setup
if /i "%CMD%"=="train" goto :train
if /i "%CMD%"=="quick-train" goto :quick_train

"%PY%" -m %PKG%.cli %*
exit /b %ERRORLEVEL%

:launch_gui
if not defined LAUNCHER (
    echo Launcher script not found in scripts\
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%"
exit /b %ERRORLEVEL%

:web
echo Starting %BRAND% server at %WEB_URL% ...
start "%BRAND% Server" "%PY%" -m %PKG%.server
timeout /t 3 /nobreak >nul
start "" "%WEB_URL%"
exit /b 0

:server
echo Starting %BRAND% API server at %WEB_URL% ...
"%PY%" -m %PKG%.server
exit /b %ERRORLEVEL%

:setup
if not exist "%ROOT%\venv\Scripts\python.exe" (
    echo Creating virtual environment ...
    python -m venv "%ROOT%\venv"
    set "PY=%ROOT%\venv\Scripts\python.exe"
)
echo Installing dependencies ...
"%PY%" -m pip install --upgrade pip
if exist "%ROOT%\requirements.txt" "%PY%" -m pip install -r "%ROOT%\requirements.txt"
echo Setup complete. Double-click this bat file to open the GUI.
exit /b %ERRORLEVEL%

:train
echo Starting comprehensive training for %BRAND% ...
"%PY%" "%ROOT%\scripts\train_comprehensive.py" --mode full
exit /b %ERRORLEVEL%

:quick_train
echo Starting quick training for %BRAND% ...
"%PY%" "%ROOT%\scripts\train_comprehensive.py" --mode quick
exit /b %ERRORLEVEL%

:show_help
echo.
echo   %BRAND% launcher
echo.
echo   Double-click this file to open the GUI.
echo.
echo   Commands:
echo     "%%~nx0"              Open the graphical launcher
echo     "%%~nx0" gui          Open the graphical launcher
echo     "%%~nx0" web          Start the server and open chat/image UI
echo     "%%~nx0" server       Start the API server
echo     "%%~nx0" train        Train on all available data sources
echo     "%%~nx0" quick-train  Shorter training pass
echo     "%%~nx0" setup        Create venv and install dependencies
echo     "%%~nx0" help         Show this help
echo.
echo   Web UI: %WEB_URL%
echo   Chat, image, video, voice, think, detective, and analyze modes are in the GUI.
echo.
exit /b 0
