@echo off
setlocal
set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "PY=%ROOT%\venv\Scripts\python.exe"
set "SCRIPT=%ROOT%\scripts\run_think_train_loop.py"
set "LOGDIR=%ROOT%\logs\think_train"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if exist "%LOGDIR%\STOP" del /f /q "%LOGDIR%\STOP"
"%PY%" -u "%SCRIPT%" --max-cycles 48 --max-hours 96 --text-target 0.70 --pass-streak 3 >> "%LOGDIR%\console.out.log" 2>> "%LOGDIR%\console.err.log"
