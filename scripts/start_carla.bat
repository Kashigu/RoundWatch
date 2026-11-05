@echo off
REM Start CARLA 0.9.15 Server in Fixed 20 FPS Synchronous Mode
echo ===================================================
echo Starting CARLA Simulator 0.9.15...
if exist "C:\Simulation_Engines\CARLA_0.9.15" (
    set "CARLA_DIR=C:\Simulation_Engines\CARLA_0.9.15"
) else (
    set "CARLA_DIR=E:\Simulation_Engines\CARLA_0.9.15"
)
echo Directory: %CARLA_DIR%
echo ===================================================

cd /d "%CARLA_DIR%"
start CarlaUE4.exe -carla-server -fps=20 -quality-level=Epic


