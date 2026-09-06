@echo off
cd /d "%~dp0"
set "PY=python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" scripts\get_sdcpp.py --rollback
pause
