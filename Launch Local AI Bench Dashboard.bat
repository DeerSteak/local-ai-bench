@echo off
cd /d "%~dp0"
call launch_dashboard.bat
exit /b %errorlevel%
