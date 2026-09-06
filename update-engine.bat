@echo off
REM ===========================================================================
REM  Met a jour le moteur stable-diffusion.cpp (sd-cli) vers la DERNIERE version.
REM  A lancer quand un nouveau modele (ex. Krea 2) n'est pas reconnu par le
REM  moteur actuel ("get sd version from file failed").
REM ===========================================================================
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo   Mise a jour du moteur stable-diffusion.cpp (derniere version)
echo ============================================================
"%PY%" scripts\get_sdcpp.py --variant cuda --force
echo.
echo Termine. Relancez run.bat.
pause
