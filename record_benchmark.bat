@echo off
title RoundWatch Benchmark Recording
rem Records a CARLA + SUMO clip with exact instance-segmentation labels and
rem exports YOLO datasets. Start CARLA first (scripts\start_carla.bat).
rem Any arguments are passed through, e.g.:
rem   record_benchmark.bat --name clear_noon_01 --fps 1 5 20
cd /d "%~dp0"
call scripts\load_env.bat
if not defined PYTHON_EXE set "PYTHON_EXE=python"

if not defined CARLA_ROOT (
    echo CARLA_ROOT is not configured. Copy .env.example to .env and set the path.
    pause
    exit /b 1
)
if not defined SUMO_HOME (
    echo SUMO_HOME is not configured. Copy .env.example to .env and set the path.
    pause
    exit /b 1
)
set "PATH=%SUMO_HOME%\bin;%PATH%"
set "PYTHONPATH=%SUMO_HOME%\tools;%CARLA_ROOT%\Co-Simulation\Sumo;%~dp0"
"%PYTHON_EXE%" scripts\capture_yolo_benchmark.py %*
pause
