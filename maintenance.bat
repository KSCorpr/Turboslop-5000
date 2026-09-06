@echo off
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

if not "%~1"=="" goto run

echo ============================================================
echo   Maintenance - Turboslop 5000
echo ------------------------------------------------------------
echo   Nettoie ce qu'une mise a jour par copier-coller laisse
echo   derriere : code des fonctions retirees, modules orphelins,
echo   caches Python, fichiers temporaires. Verifie ensuite que
echo   tout compile ET que le moteur sd-cli sait faire ce que le
echo   code lui demande.
echo.
echo   Les DONNEES (poids, add-ons d'anciennes versions) sont
echo   seulement CHIFFREES ici, pas supprimees.
echo.
echo   Options :
echo       maintenance.bat --update-engine   aligne le moteur
echo       maintenance.bat --purge           efface les restes
echo       maintenance.bat --all             les deux
echo ============================================================
echo.

:run
"%PY%" scripts\maintenance.py %*
echo.
pause
