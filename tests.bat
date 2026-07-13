@echo off
setlocal

set SCRIPT_DIR=%~dp0
set VENV=%SCRIPT_DIR%bench-env

if not exist "%VENV%\Scripts\activate.bat" (
    echo Virtual environment not found at %VENV% -- run setup.bat first.
    exit /b 1
)

call "%VENV%\Scripts\activate.bat"
pip install --quiet -r "%SCRIPT_DIR%tests\requirements.txt"
python -m pytest "%SCRIPT_DIR%tests" %*
