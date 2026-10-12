@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "TF_CPP_MIN_LOG_LEVEL=2"
set "CONFIG=configs\research.json"
if /i "%~1"=="quick" set "CONFIG=configs\quick.json"
if /i "%~1"=="improved" set "CONFIG=configs\improved.json"
if exist ".venv\Scripts\python.exe" goto install
echo Creating the project environment using Python 3.12...
py -3.12 -m venv .venv
if errorlevel 1 (
  echo Python 3.12 is required. Install it from python.org, then run this file again.
  exit /b 1
)
:install
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-lock.txt
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-deps -e .
if errorlevel 1 goto failure
echo Downloading data and running the configured experiments...
".venv\Scripts\python.exe" -m cloud_anomaly.cli run --config "%CONFIG%"
if errorlevel 1 goto failure
if /i "%~1"=="improved" goto compare
echo Calibrating decision thresholds using validation labels...
".venv\Scripts\python.exe" scripts\recalibrate_results.py --latest
if errorlevel 1 goto failure
goto complete
:compare
".venv\Scripts\python.exe" scripts\compare_experiments.py --latest
if errorlevel 1 goto failure
:complete
echo.
echo Completed. Results are in artifacts\runs. Launch run_frontend.bat to explore them.
exit /b 0
:failure
echo.
echo The project stopped after an error. Review the message above; existing data is preserved.
exit /b 1
