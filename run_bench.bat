@echo off
setlocal

set SCRIPT_DIR=%~dp0
set VENV=%SCRIPT_DIR%bench-env
cd /d "%SCRIPT_DIR%"

if not exist "%VENV%\Scripts\activate.bat" (
    for /f "tokens=1 delims=." %%T in ("%TIME: =0%") do echo [%%T] Virtual environment not found at %VENV% -- run setup.bat first.
    set "BENCH_EXIT_CODE=1"
    goto finish
)

call "%VENV%\Scripts\activate.bat"
if "%~1"=="" goto frontend
if /i "%~1"=="--ui" goto frontend_with_args
if /i "%~1"=="--interface" goto frontend_with_args
python -m scripts.app.benchmark %*
set "BENCH_EXIT_CODE=%errorlevel%"
goto finish

:frontend
python -m scripts.app.benchmark_launcher --ui auto
set "BENCH_EXIT_CODE=%errorlevel%"
goto finish

:frontend_with_args
python -m scripts.app.benchmark_launcher %*
set "BENCH_EXIT_CODE=%errorlevel%"

:finish
exit /b %BENCH_EXIT_CODE%
