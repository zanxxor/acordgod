@echo off
rem AcordGod - Inicia el servidor del transcriptor.
rem Deja esta ventana abierta: la app web detecta el servidor y activa
rem el boton "Transcribir".
cd /d "%~dp0transcriptor"
"%~dp0transcriptor\venv\Scripts\python.exe" servidor.py
pause
