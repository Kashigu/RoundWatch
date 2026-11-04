@echo off
title RoundWatch Live CARLA + SUMO Monitor
echo ============================================================
echo   Starting Live CARLA + SUMO Roundabout Simulation...
echo ============================================================
cd /d "%~dp0"
set SUMO_HOME=E:\Simulation_Engines\SUMO
set PYTHONPATH=E:\Simulation_Engines\CARLA_0.9.15\Co-Simulation\Sumo;%~dp0
.venv310\Scripts\python.exe scripts\watch_live_simulation.py
pause

