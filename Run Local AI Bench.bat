@echo off
cd /d "%~dp0"
call run_bench.bat --ui gui
exit /b %errorlevel%
