@echo off
setlocal EnableExtensions
for %%D in ("%~dp0..") do set "ROOT=%%~fD"
cd /d "%ROOT%"

if "%~1"=="--list-targets" goto list_targets
if "%~1"=="" goto usage
if not "%~3"=="" goto usage

set "TARGET=%~1"
set "RESULT=%~2"
if "%RESULT%"=="" set "RESULT=%CD%\qualification-evidence\%TARGET%\results_qualification_%TARGET%.json"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\release\qualification_targets.ps1" -Target "%TARGET%" || (
  echo Unknown qualification target: %TARGET% 1>&2
  exit /b 2
)
set "ENGINE=llamacpp"
echo %TARGET% | findstr /c:"-vllm-" >nul && set "ENGINE=vllm"

for %%D in ("%RESULT%") do set "EVIDENCE_DIR=%%~dpD"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\release\qualification_setup.ps1" -Target "%TARGET%" -Engine "%ENGINE%" -EvidenceDir "%EVIDENCE_DIR%" -SetupPath "%CD%\setup.bat"
if errorlevel 1 exit /b %ERRORLEVEL%
bench-env\Scripts\python.exe -m scripts.release.qualification_run "%TARGET%" --root "%CD%" --result "%RESULT%"
exit /b %ERRORLEVEL%

:list_targets
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\release\qualification_targets.ps1"
exit /b %ERRORLEVEL%

:usage
echo Usage: qualification\%~nx0 TARGET [RESULT_JSON]
echo        qualification\%~nx0 --list-targets
exit /b 2
