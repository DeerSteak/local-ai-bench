@echo off
cd /d "%~dp0"
call run_bench.bat --interface gui
exit /b %errorlevel%
