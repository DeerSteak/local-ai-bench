@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="--list-targets" goto list_targets
if "%~1"=="" goto usage
if not "%~3"=="" goto usage

set "TARGET=%~1"
set "RESULT=%~2"
if "%RESULT%"=="" set "RESULT=%CD%\qualification-evidence\%TARGET%\results_qualification_%TARGET%.json"
powershell.exe -NoProfile -Command "$text = Get-Content -Raw 'scripts\release\qualification_targets.py'; $ids = @([regex]::Matches($text, '\"id\": \"([^\"]+)\"').ForEach({$_.Groups[1].Value})); if ($ids -notcontains $env:TARGET) { exit 1 }" || (
  echo Unknown qualification target: %TARGET% 1>&2
  exit /b 2
)
set "ENGINE=llamacpp"
echo %TARGET% | findstr /c:"-vllm-" >nul && set "ENGINE=vllm"

call setup.bat --qualification %ENGINE% || exit /b 1
bench-env\Scripts\python.exe -m scripts.release.qualification_run "%TARGET%" --root "%CD%" --result "%RESULT%"
exit /b %ERRORLEVEL%

:list_targets
powershell.exe -NoProfile -Command "$text = Get-Content -Raw 'scripts\release\qualification_targets.py'; [regex]::Matches($text, '\"id\": \"([^\"]+)\"').ForEach({$_.Groups[1].Value})"
exit /b %ERRORLEVEL%

:usage
echo Usage: %~nx0 TARGET [RESULT_JSON]
echo        %~nx0 --list-targets
exit /b 2
