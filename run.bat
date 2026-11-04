@echo off
title Driver Drowsiness and Distraction Warning System
cd /d "%~dp0"

echo =====================================================================
echo  DRIVER DROWSINESS AND DISTRACTION WARNING SYSTEM
echo  Laboratory Research Prototype (AIML / Computer Vision)
echo =====================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    echo Please install dependencies first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting real-time monitoring...
echo [INFO] Press 'q' or 'ESC' to quit cleanly.
echo [INFO] Press 'm' to toggle audio mute.
echo [INFO] Press 'c' to recalibrate open-eye baseline.
echo.

".venv\Scripts\python.exe" -m src.main %*

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)

