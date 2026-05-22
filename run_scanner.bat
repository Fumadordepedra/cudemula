@echo off
title Scanner B3
echo Preparando dependencias do Scanner B3...
py -m pip install -r requirements.txt
echo.
echo Iniciando o Scanner B3...
cd /d "C:\Users\HACKER\.gemini\antigravity\scratch\scanner_b3"
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
py -m streamlit run app.py
pause
