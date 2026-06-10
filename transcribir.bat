@echo off
rem AcordGod - Transcriptor: arrastra o pega un link de YouTube.
rem Uso:  transcribir "https://www.youtube.com/watch?v=XXXX"
if "%~1"=="" (
  set /p URL=Pega el link de YouTube y presiona Enter:
) else (
  set URL=%~1
)
"%~dp0transcriptor\venv\Scripts\python.exe" "%~dp0transcriptor\transcribir.py" "%URL%" %2 %3 %4
pause
