@echo off
title RoundWatch Live CARLA + SUMO Monitor
echo ============================================================
echo   Starting Live CARLA + SUMO Roundabout Simulation...
echo ============================================================
cd /d "%~dp0"
if exist "C:\Simulation_Engines\SUMO" (
    set "SUMO_HOME=C:\Simulation_Engines\SUMO"
) else (
    set "SUMO_HOME=E:\Simulation_Engines\SUMO"
)
if exist "C:\Simulation_Engines\CARLA_0.9.15\Co-Simulation\Sumo" (
    set "PYTHONPATH=C:\Simulation_Engines\CARLA_0.9.15\Co-Simulation\Sumo;%~dp0"
) else (
    set "PYTHONPATH=E:\Simulation_Engines\CARLA_0.9.15\Co-Simulation\Sumo;%~dp0"
)
.venv310\Scripts\python.exe scripts\watch_live_simulation.py
pause


