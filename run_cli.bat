@echo off
REM usage: run_cli.bat            (all six models)    run_cli.bat --fast    run_cli.bat --rl
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python main.py %*
pause
