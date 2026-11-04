@echo off
title RoundWatch AR Desktop Viewer
echo ============================================================
echo   Launching RoundWatch AR Desktop Video Viewer...
echo ============================================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0
.venv310\Scripts\python.exe scripts\view_live.py
pause

