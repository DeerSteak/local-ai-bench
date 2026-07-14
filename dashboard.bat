@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set DASHBOARD_DIR=%SCRIPT_DIR%dashboard
set DIST_DIR=%DASHBOARD_DIR%\dist
set RESULTS_DIR=%SCRIPT_DIR%results
set PORT=3000
set REBUILD=0

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--rebuild" (
    set REBUILD=1
    shift
    goto parse_args
)
echo Unknown option: %~1
exit /b 1
:args_done

if not exist "%DASHBOARD_DIR%" (
    echo Error: dashboard directory not found at %DASHBOARD_DIR%
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo Error: npm not found in PATH.
    echo Install Node.js from https://nodejs.org/ and re-run.
    exit /b 1
)

if not exist "%DASHBOARD_DIR%\node_modules" (
    echo Installing dependencies ^(npm install^) ...
    pushd "%DASHBOARD_DIR%"
    call npm install
    if errorlevel 1 (
        echo npm install failed -- fix the errors above and try again.
        popd
        exit /b 1
    )
    popd
    echo Dependencies installed.
    echo.
)

set NEEDS_BUILD=0
if "%REBUILD%"=="1" set NEEDS_BUILD=1
if not exist "%DIST_DIR%\index.html" set NEEDS_BUILD=1

if "%NEEDS_BUILD%"=="1" (
    echo Building dashboard ...
    pushd "%DASHBOARD_DIR%"
    call npm run build
    if errorlevel 1 (
        echo Build failed -- fix the errors above and try again.
        popd
        exit /b 1
    )
    popd
    echo Build complete.
    echo.
)

echo Dashboard -^> http://localhost:%PORT%
echo Drop your results JSON files onto the page to analyze them.
echo Ctrl-C to stop.
echo.

if exist "%RESULTS_DIR%" (
    start "" explorer "%RESULTS_DIR%"
)

call npm --prefix "%DASHBOARD_DIR%" run preview -- --port %PORT% --open
