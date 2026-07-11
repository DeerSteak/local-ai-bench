@echo off
setlocal

set SCRIPT_DIR=%~dp0
set VENV=%SCRIPT_DIR%bench-env

if not exist "%VENV%\Scripts\activate.bat" (
    echo Virtual environment not found at %VENV% -- run setup.bat first.
    exit /b 1
)

call "%VENV%\Scripts\activate.bat"
python "%SCRIPT_DIR%scripts\benchmark.py" %*
