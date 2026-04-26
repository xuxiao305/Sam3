@echo off
chcp 65001 >nul 2>&1
title SAM3 Segment Tool

echo ══════════════════════════════════════════════════════
echo   SAM3 Segment Tool - Launcher
echo ══════════════════════════════════════════════════════
echo.

:: ComfyUI Python environment (symlinked from .venv)
set PYTHON=d:\AI\ComfyUI-Easy-Install\python_embeded\python.exe

:: Verify Python exists
if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    echo Please check your ComfyUI installation path.
    pause
    exit /b 1
)

:: Change to project directory
cd /d "%~dp0"

echo [INFO] Python: %PYTHON%
echo [INFO] Working directory: %cd%
echo [INFO] Starting SAM3 Segment Tool...
echo.

:: Run the application
"%PYTHON%" "%~dp0sam3_app\main.py" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
