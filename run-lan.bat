@echo off
REM ===========================================================================
REM  Atelier - Lancement en mode RESEAU LOCAL
REM  Permet a vos collegues (Mac/PC sur le meme Wi-Fi) d'ouvrir Atelier dans
REM  leur navigateur, en utilisant CE PC pour generer les images.
REM
REM  L'adresse a partager (http://<IP>:7860) s'affiche au demarrage.
REM
REM  Pour proteger par mot de passe :
REM      run-lan.bat --auth nom:motdepasse
REM ===========================================================================
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" app.py --listen %*
pause
