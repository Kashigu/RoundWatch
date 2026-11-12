@echo off
call "%~dp0load_env.bat"
if not defined CARLA_ROOT (
    echo CARLA_ROOT is not configured. Copy .env.example to .env and set the path.
    pause
    exit /b 1
)
if not exist "%CARLA_ROOT%\CarlaUE4.exe" (
    echo CARLA executable not found: "%CARLA_ROOT%\CarlaUE4.exe"
    pause
    exit /b 1
)

REM Start CARLA 0.9.15 Server in Fixed 20 FPS Synchronous Mode
echo ===================================================
echo Starting CARLA Simulator 0.9.15 on NVIDIA RTX 4060...
echo Directory: %CARLA_ROOT%
echo ===================================================

cd /d "%CARLA_ROOT%"
start "" CarlaUE4.exe -carla-server -fps=20 -quality-level=Epic
