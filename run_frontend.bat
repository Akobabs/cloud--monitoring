@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
  echo The environment is missing. Run run_project.bat first.
  exit /b 1
)
echo Opening the research dashboard at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false
exit /b %errorlevel%

