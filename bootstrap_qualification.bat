@echo off
setlocal
echo Qualification host bootstrap checks Python and Git. GPU drivers and SDKs are not changed.
where py >nul 2>nul && where git >nul 2>nul && goto ready
echo Preview: winget install --id Python.Python.3.12 --exact
echo Preview: winget install --id Git.Git --exact
if not "%~1"=="--execute" (
  echo Repeat with --execute from an Administrator terminal after reviewing the commands.
  exit /b 0
)
where py >nul 2>nul || winget install --id Python.Python.3.12 --exact --accept-package-agreements --accept-source-agreements || exit /b 1
where git >nul 2>nul || winget install --id Git.Git --exact --accept-package-agreements --accept-source-agreements || exit /b 1
:ready
py -3 --version || exit /b 1
git --version || exit /b 1
