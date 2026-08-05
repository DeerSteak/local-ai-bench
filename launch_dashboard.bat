@echo off
setlocal

set SCRIPT_DIR=%~dp0
set DASHBOARD_DIR=%SCRIPT_DIR%dashboard
set RESULTS_DIR=%SCRIPT_DIR%results
set PORT=3000
set RESULT_ARGS=
set HAS_RESULTS=0

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--port" (
    if "%~2"=="" (
        echo Missing value for --port
        exit /b 1
    )
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--result" (
    if "%~2"=="" (
        echo Missing value for --result
        exit /b 1
    )
    set RESULT_ARGS=%RESULT_ARGS% "%~2"
    set HAS_RESULTS=1
    shift
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

echo Building dashboard ...
pushd "%DASHBOARD_DIR%"
call npm run build
if errorlevel 1 (
    echo Build failed -- fix the errors above and try again.
    popd
    exit /b 1
)
popd
node "%DASHBOARD_DIR%\stage_selected_results.mjs" "%DASHBOARD_DIR%\dist" %RESULT_ARGS%
if errorlevel 1 exit /b 1
echo Build complete.
echo.

echo Dashboard -^> http://localhost:%PORT%
echo Drop your results JSON files onto the page to analyze them.
echo Ctrl-C to stop.
echo.

if "%HAS_RESULTS%"=="0" if exist "%RESULTS_DIR%" (
    start "" explorer "%RESULTS_DIR%"
)

if "%HAS_RESULTS%"=="1" (
    call npm --prefix "%DASHBOARD_DIR%" run preview -- --port %PORT% --open "/?autoload=1"
) else (
    call npm --prefix "%DASHBOARD_DIR%" run preview -- --port %PORT% --open
)
set PREVIEW_STATUS=%errorlevel%
node "%DASHBOARD_DIR%\stage_selected_results.mjs" "%DASHBOARD_DIR%\dist" >nul 2>nul
exit /b %PREVIEW_STATUS%
