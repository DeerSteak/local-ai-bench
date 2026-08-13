@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if not "%~1"=="" goto validate_interval
call "%~f0" 1.0
if errorlevel 1 exit /b %errorlevel%
timeout /t 30 /nobreak >nul
call "%~f0" 0.5
if errorlevel 1 exit /b %errorlevel%
timeout /t 30 /nobreak >nul
call "%~f0" 0.25
if errorlevel 1 exit /b %errorlevel%
echo All Windows qualification intervals completed.
exit /b 0

:validate_interval
if "%~1"=="1.0" goto interval_ok
if "%~1"=="0.5" goto interval_ok
if "%~1"=="0.25" goto interval_ok
echo Usage: qual_windows.bat [1.0^|0.5^|0.25]
exit /b 2

:interval_ok
set "INTERVAL=%~1"
if "%INTERVAL%"=="1.0" set "INTERVAL_DIR=1s"
if "%INTERVAL%"=="0.5" set "INTERVAL_DIR=0.5s"
if "%INTERVAL%"=="0.25" set "INTERVAL_DIR=0.25s"
set "OUTDIR=results\qualification\rtx-5090-windows\%INTERVAL_DIR%"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

for /L %%P in (1,1,20) do (
    call :run_pair %%P
    if errorlevel 1 exit /b !errorlevel!
)

echo Qualification interval %INTERVAL% completed.
exit /b 0

:run_pair
set "PAIR=%1"
set "NUMBER=0%1"
set "NUMBER=%NUMBER:~-2%"
set "OFF_FILE=%OUTDIR%\off-%NUMBER%.json"
set "ON_FILE=%OUTDIR%\on-%NUMBER%.json"

if not exist "%OFF_FILE%" goto check_on_only
if not exist "%ON_FILE%" goto incomplete_pair
echo Pair %NUMBER% already complete - skipping.
exit /b 0

:check_on_only
if exist "%ON_FILE%" goto incomplete_pair
set "LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC=%INTERVAL%"
set /A "PARITY=%PAIR% %% 2"
if "%PARITY%"=="1" goto run_off_on

call :run_on %NUMBER%
if errorlevel 1 exit /b %errorlevel%
call :wait_between
call :run_off %NUMBER%
if errorlevel 1 exit /b %errorlevel%
if %PAIR% LSS 20 call :wait_between
exit /b 0

:run_off_on
call :run_off %NUMBER%
if errorlevel 1 exit /b %errorlevel%
call :wait_between
call :run_on %NUMBER%
if errorlevel 1 exit /b %errorlevel%
if %PAIR% LSS 20 call :wait_between
exit /b 0

:run_off
echo Running pair %1 telemetry OFF...
call run_bench.bat --ui none --tests llm --llm-models qwen3.5:4b-q4_K_M --max-prompt-tokens 2048 --warmup 2 --runs 3 --out "%OUTDIR%\off-%1.json"
exit /b %errorlevel%

:run_on
echo Running pair %1 telemetry ON...
call run_bench.bat --ui none --tests llm --llm-models qwen3.5:4b-q4_K_M --max-prompt-tokens 2048 --warmup 2 --runs 3 --memory-telemetry --out "%OUTDIR%\on-%1.json"
exit /b %errorlevel%

:wait_between
timeout /t 30 /nobreak >nul
exit /b 0

:incomplete_pair
echo A partial pair exists in %OUTDIR%. Remove or relocate that pair's files, then retry.
exit /b 3
