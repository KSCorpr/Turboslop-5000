@echo off
REM ===========================================================================
REM  Cree un ZIP portable a distribuer a vos amis.
REM  Inclut : le code, le Python portable (python\) et le moteur (bin\).
REM  Exclut : modeles, sorties, donnees perso (vos amis telechargent les
REM           modeles depuis l'onglet Bibliotheque, via HuggingFace).
REM  => Vos amis n'ont AUCUN telechargement GitHub a faire : ils decompressent
REM     et lancent run.bat.
REM ===========================================================================
setlocal
cd /d "%~dp0"

set "NAME=Turboslop-5000-portable"
set "STAGE=%TEMP%\fedimg_dist"
set "ZIP=%~dp0%NAME%.zip"

echo Preparation du paquet portable...
if exist "%STAGE%" rmdir /s /q "%STAGE%"
mkdir "%STAGE%\%NAME%"

REM Copie tout sauf les dossiers volumineux/volatils.
robocopy "%~dp0." "%STAGE%\%NAME%" /E ^
  /XD models outputs tmp userdata tools_repo .git __pycache__ dist ^
  /XF *.log "%NAME%.zip" >nul

echo Compression (cela peut prendre une minute)...
if exist "%ZIP%" del /q "%ZIP%"
powershell -NoProfile -Command ^
  "Compress-Archive -Force -Path '%STAGE%\%NAME%' -DestinationPath '%ZIP%'"

rmdir /s /q "%STAGE%"
echo.
echo ============================================================
echo   Paquet cree : %NAME%.zip
echo   Vos amis : decompresser, puis lancer run.bat.
echo   (Les modeles se telechargent depuis l'onglet Bibliotheque.)
echo ============================================================
pause
