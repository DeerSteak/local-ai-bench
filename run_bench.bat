@echo off
setlocal

set SCRIPT_DIR=%~dp0
set VENV=%SCRIPT_DIR%bench-env
set "PAUSE_ON_EXIT="
set CMDCMDLINE | %SystemRoot%\System32\findstr.exe /l /i /c:"%~f0" >nul
if not errorlevel 1 set "PAUSE_ON_EXIT=1"

if not exist "%VENV%\Scripts\activate.bat" (
    for /f "tokens=1 delims=." %%T in ("%TIME: =0%") do echo [%%T] Virtual environment not found at %VENV% -- run setup.bat first.
    set "BENCH_EXIT_CODE=1"
    goto finish
)

call "%VENV%\Scripts\activate.bat"
if "%~1"=="" goto frontend
python "%SCRIPT_DIR%scripts\benchmark.py" %*
set "BENCH_EXIT_CODE=%errorlevel%"
goto finish

:frontend
python "%SCRIPT_DIR%scripts\benchmark_frontend.py"
set "BENCH_EXIT_CODE=%errorlevel%"

:finish
if defined PAUSE_ON_EXIT pause
exit /b %BENCH_EXIT_CODE%
