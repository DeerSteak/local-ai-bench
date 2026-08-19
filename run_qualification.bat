@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="--list-targets" goto list_targets
if "%~1"=="" goto auto_preview
if "%~1"=="--execute" goto auto_execute
if "%~3"=="" goto usage

set "TARGET=%~1"
set "BASELINE_VERSION=%~2"
set "TARGET_VERSION=%~3"
if "%~4"=="" (
  set "OUTPUT_DIR=%CD%\qualification-evidence\%TARGET%"
) else (
  set "OUTPUT_DIR=%~f4"
)
set "EXECUTE=%~5"
if not "%EXECUTE%"=="" if not "%EXECUTE%"=="--execute" goto usage

if not exist "qualification-env\Scripts\python.exe" (
  py -3 -m venv qualification-env || exit /b 1
  qualification-env\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
)
qualification-env\Scripts\python.exe -m pip install --quiet -r requirements.txt || exit /b 1

qualification-env\Scripts\python.exe -m scripts.release.qualification_recipe --target "%TARGET%" --root "%CD%" --output "%OUTPUT_DIR%" --baseline-version "%BASELINE_VERSION%" --target-version "%TARGET_VERSION%" || exit /b 1
if "%EXECUTE%"=="--execute" (
  qualification-env\Scripts\python.exe -m scripts.release.qualification_automation "%OUTPUT_DIR%\qualification-recipe.json" --output "%OUTPUT_DIR%" --execute
) else (
  qualification-env\Scripts\python.exe -m scripts.release.qualification_automation "%OUTPUT_DIR%\qualification-recipe.json" --output "%OUTPUT_DIR%"
)
exit /b %ERRORLEVEL%

:auto_preview
call bootstrap_qualification.bat || exit /b 1
call :ensure_env || exit /b 1
qualification-env\Scripts\python.exe -m scripts.release.qualification_auto --root "%CD%"
exit /b %ERRORLEVEL%

:auto_execute
call bootstrap_qualification.bat --execute || exit /b 1
call :ensure_env || exit /b 1
qualification-env\Scripts\python.exe -m scripts.release.qualification_auto --root "%CD%" --execute
exit /b %ERRORLEVEL%

:ensure_env
if not exist "qualification-env\Scripts\python.exe" (
  py -3 -m venv qualification-env || exit /b 1
  qualification-env\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
)
qualification-env\Scripts\python.exe -m pip install --quiet -r requirements.txt || exit /b 1
exit /b 0

:list_targets
if exist "qualification-env\Scripts\python.exe" (
  qualification-env\Scripts\python.exe -m scripts.release.qualification_recipe --list-targets
) else (
  py -3 -m scripts.release.qualification_recipe --list-targets
)
exit /b %ERRORLEVEL%

:usage
echo Usage: %~nx0 TARGET BASELINE_VERSION TARGET_VERSION [OUTPUT_DIR] [--execute]
echo        %~nx0 --list-targets
exit /b 2
