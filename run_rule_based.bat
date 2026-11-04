@echo off
title Driver Drowsiness System - Rule-Based Demo
cd /d "%~dp0"

echo =====================================================================
echo  DRIVER DROWSINESS WARNING SYSTEM - RULE-BASED DEMO
echo =====================================================================
echo.

".venv\Scripts\python.exe" -m src.main --rule-based-only %*

if errorlevel 1 (
    pause
)

