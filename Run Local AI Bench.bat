@echo off
cd /d "%~dp0"
call run_bench.bat --ui gui
set "BENCH_EXIT_CODE=%errorlevel%"
pause
exit /b %BENCH_EXIT_CODE%
