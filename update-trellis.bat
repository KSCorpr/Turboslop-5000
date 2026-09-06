@echo off
REM ===========================================================================
REM  Met a jour le moteur trellis.cpp (image -> 3D) vers la DERNIERE version.
REM
REM  Telecharge la derniere release Windows CUDA de pwilkin/trellis.cpp et
REM  remplace le binaire dans bin\trellis\ (l'ancienne version est retiree
REM  avant extraction, pour ne pas melanger des DLL de deux releases).
REM
REM  Les MODELES 3D (models\trellis\, ~16 Go) ne sont PAS retelecharges :
REM  ils changent rarement. Pour les (re)installer, utilisez le bouton
REM  d'installation de l'onglet "Image -> 3D".
REM ===========================================================================
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo   Mise a jour du moteur trellis.cpp (image -^> 3D)
echo ============================================================
"%PY%" scripts\get_trellis.py --binary --force
echo.
echo Termine. Relancez run.bat.
pause
