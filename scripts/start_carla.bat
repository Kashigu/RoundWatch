@echo off
REM Start CARLA 0.9.15 Server in Fixed 20 FPS Synchronous Mode
echo ===================================================
echo Starting CARLA Simulator 0.9.15 on NVIDIA RTX 4060...
echo Directory: E:\Simulation_Engines\CARLA_0.9.15
echo ===================================================

cd /d "E:\Simulation_Engines\CARLA_0.9.15"
start CarlaUE4.exe -carla-server -fps=20 -quality-level=Epic

