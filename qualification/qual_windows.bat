@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0qual_windows.ps1" %*
exit /b %errorlevel%
