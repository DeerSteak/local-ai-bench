@echo off
:: setup.bat — local-ai-bench setup for Windows
:: Usage: double-click or run from a terminal in the repo directory

setlocal EnableDelayedExpansion
set VENV_DIR=bench-env

echo.
echo ==================================================
echo   local-ai-bench setup
echo ==================================================

:: ── 1. Find Python 3.11+ ──────────────────────────────────────────────────────
echo.
echo [Python]

set PYTHON=
for %%C in (python3.13 python3.12 python3.11 python3 python) do (
    where %%C >nul 2>&1
    if !errorlevel! == 0 (
        for /f "tokens=*" %%V in ('%%C -c "import sys; ok = sys.version_info >= (3,11); print(sys.executable if ok else '')" 2^>nul') do (
            if not "%%V"=="" (
                set PYTHON=%%C
                for /f "tokens=*" %%W in ('%%C --version 2^>^&1') do echo   OK  %%W found
                goto :python_found
            )
        )
    )
)

:: Python not found or too old — ask before installing via winget
echo   !  Python 3.11+ not found
echo.
echo   This will:
echo     - Install Python 3.11 via winget
echo.
set /p _PY_REPLY="  Proceed? [Y/n] "
echo.
if /i "%_PY_REPLY%"=="n" goto :python_cancelled
if /i "%_PY_REPLY%"=="no" goto :python_cancelled

echo   -^>  Installing Python 3.11 via winget...
winget install --id Python.Python.3.11 --source winget --accept-package-agreements --accept-source-agreements
if %errorlevel% neq 0 (
    echo   X  winget install failed. Please install Python 3.11+ from https://python.org and re-run.
    pause
    exit /b 1
)
:: Refresh PATH
call refreshenv >nul 2>&1 || (
    echo   !  Please close and reopen this terminal, then re-run setup.bat
    pause
    exit /b 1
)
set PYTHON=python
echo   OK  Python installed
goto :python_found

:python_cancelled
echo   X  Setup cancelled -- Python 3.11+ is required.
pause
exit /b 1

:python_found

:: ── 2. Create venv ─────────────────────────────────────────────────────────────
echo.
echo [Virtual Environment]

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo   OK  Venv already exists at %VENV_DIR%
) else (
    echo   -^>  Creating venv at %VENV_DIR%...
    %PYTHON% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo   X  Failed to create venv. Is python3-venv installed?
        pause
        exit /b 1
    )
    echo   OK  Venv created
)

set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe
set VENV_PIP=%VENV_DIR%\Scripts\pip.exe

:: ── 3. Base Python dependencies ───────────────────────────────────────────────
echo.
echo [Python Packages]
echo   -^>  Installing from requirements.txt ...
%VENV_PIP% install -r requirements.txt
if %errorlevel% neq 0 (
    echo   X  pip install -r requirements.txt failed. Check your internet connection.
    pause
    exit /b 1
)
echo   OK  Base dependencies installed

:: ── 4. Run setup_check.py ─────────────────────────────────────────────────────
:: (Ollama detection/install happens inside setup_check.py, gated behind its
:: own approval prompt, so it isn't installed here without asking.)
echo.
echo [Running setup_check.py]
echo.

%VENV_PYTHON% scripts\setup_check.py %*
if %errorlevel% neq 0 (
    echo.
    echo   X  setup_check.py exited with errors. See above.
    pause
    exit /b 1
)

:: ── 5. Done ────────────────────────────────────────────────────────────────────
echo.
echo ==================================================
echo   Setup complete.
echo ==================================================
echo.
echo   To run benchmarks:
echo     run_bench.bat
echo.
set /p _REPLY="  Run the benchmark now? [y/N] "
echo.
if /i "%_REPLY%"=="y"   call "%~dp0run_bench.bat"
if /i "%_REPLY%"=="yes" call "%~dp0run_bench.bat"
