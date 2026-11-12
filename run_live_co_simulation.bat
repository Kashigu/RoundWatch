@echo off
title RoundWatch Live CARLA + SUMO Monitor
echo ============================================================
echo   Starting Live CARLA + SUMO Roundabout Simulation...
echo ============================================================
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
if not exist "%CARLA_ROOT%\CarlaUE4.exe" (
    echo CARLA executable not found: "%CARLA_ROOT%\CarlaUE4.exe"
    pause
    exit /b 1
)
if not exist "%SUMO_HOME%\tools" (
    echo SUMO tools folder not found: "%SUMO_HOME%\tools"
    pause
    exit /b 1
)
set "PATH=%SUMO_HOME%\bin;%PATH%"
set "PYTHONPATH=%SUMO_HOME%\tools;%CARLA_ROOT%\Co-Simulation\Sumo;%~dp0"
"%PYTHON_EXE%" scripts\watch_live_simulation.py
pause
