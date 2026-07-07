@echo off
title Staffing Copilot Dashboard
echo.
echo ================================================
echo    Staffing Copilot — Launching Dashboard
echo ================================================
echo.

cd /d C:\Users\KUMAR\Desktop\staffing_copilot

echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/2] Starting Streamlit...
echo.
echo  Dashboard will open at: http://localhost:8501
echo  Press Ctrl+C to stop
echo.

venv\Scripts\python.exe -m streamlit run src/dashboard.py

pause