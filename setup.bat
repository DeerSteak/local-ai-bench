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

:: Python not found or too old — try to install via winget
echo   !  Python 3.11+ not found -- installing via winget...
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

:: ── 3. Install Ollama if missing ───────────────────────────────────────────────
echo.
echo [Ollama]

where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo   !  Ollama not found -- installing via winget...
    winget install --id Ollama.Ollama --source winget --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo   X  winget install failed. Please install Ollama from https://ollama.com/download and re-run.
        pause
        exit /b 1
    )
    echo   OK  Ollama installed -- you may need to restart your terminal for it to be on PATH
) else (
    for /f "tokens=*" %%V in ('ollama --version 2^>^&1') do echo   OK  %%V
)

:: ── 4. Run setup_check.py ─────────────────────────────────────────────────────
echo.
echo [Running setup_check.py]
echo.

%VENV_PYTHON% setup_check.py
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
echo     %VENV_DIR%\Scripts\activate
echo     python benchmark.py
echo.
pause
