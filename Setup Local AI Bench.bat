@echo off
cd /d "%~dp0"

call setup.bat --interface gui
set SETUP_STATUS=%errorlevel%

if %SETUP_STATUS% equ 10 exit /b 0
exit /b %SETUP_STATUS%
