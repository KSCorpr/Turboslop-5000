@echo off
REM ===========================================================================
REM  Met a jour l'APPLICATION (le code) depuis GitHub.
REM
REM  Ne touche jamais a vos donnees : models\, loras\, outputs\, userdata\,
REM  tools_repo\, bin\, python\ sont laisses intacts.
REM
REM    update.bat              met a jour
REM    update.bat --check      dit seulement ce qui changerait
REM    update.bat --rollback   annule la derniere mise a jour
REM
REM  FERMEZ l'application avant de lancer ce script (un fichier ouvert ne peut
REM  pas etre remplace sous Windows).
REM ===========================================================================
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\update_app.py %*
echo.
pause
