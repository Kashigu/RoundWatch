@echo off
title RoundWatch AR Desktop Viewer
echo ============================================================
echo   Launching RoundWatch AR Desktop Video Viewer...
echo ============================================================
cd /d "%~dp0"
call scripts\load_env.bat
if not defined PYTHON_EXE set "PYTHON_EXE=python"
set PYTHONPATH=%~dp0
"%PYTHON_EXE%" scripts\view_live.py
pause
