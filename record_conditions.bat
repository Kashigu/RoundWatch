@echo off
title RoundWatch Condition Recordings
rem Records 30 s south-camera clips of the 2-lane roundabout under several weather / light
rem conditions, all with the same traffic (route seed 101), for the YOLO conditions benchmark.
rem Start CARLA first (scripts\start_carla.bat). Then export and benchmark:
rem   python scripts\export_conditions_benchmark.py --clips cond_s101_ClearNoon ... --name conditions_real
rem   python scripts\run_yolo_benchmark.py data\yolo_benchmark\conditions_real --imgsz 1280 1920
cd /d "%~dp0"
call scripts\load_env.bat
if not defined PYTHON_EXE set "PYTHON_EXE=python"
if not defined CARLA_ROOT (
    echo CARLA_ROOT is not configured. Copy .env.example to .env and set the path.
    exit /b 1
)
set "PATH=%SUMO_HOME%\bin;%PATH%"

for %%W in (ClearNoon HardRainNoon Storm Dusk ClearNight HardRainNight) do (
    echo.
    echo ===== %%W =====
    "%PYTHON_EXE%" scripts\record_tracking_dataset.py --weather %%W --cameras south --seconds 30 ^
        --sumo-warmup 20 --route-seed 101 --out data\eval\cond_s101_%%W
)
echo ALL_CONDITION_RECORDINGS_DONE
