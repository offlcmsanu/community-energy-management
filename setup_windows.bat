@echo off
REM One-time setup: creates a virtual environment and installs the packages
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup finished. Now run run_dashboard.bat or run_cli.bat
pause
