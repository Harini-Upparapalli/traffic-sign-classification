@echo off
REM run.bat - Windows cmd wrapper to setup venv, install deps, generate dataset, train, and run app
SETLOCAL ENABLEDELAYEDEXPANSION

REM Find python (py launcher preferred)
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  set PYCMD=py -3
) else (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    set PYCMD=python
  ) else (
    echo No python or py launcher found. Install Python 3.10+ and ensure it is on PATH.
    exit /b 1
  )
)

if not exist .venv (
  echo Creating virtual environment...
  %PYCMD% -m venv .venv
)

set VENV_PY=.venv\Scripts\python.exe
if not exist %VENV_PY% (
  set VENV_PY=%PYCMD%
)

%VENV_PY% -m pip install --upgrade pip
%VENV_PY% -m pip install -r requirements.txt

echo Generating dataset (may take a minute)...
%VENV_PY% -c "from dataset_generator import generate_dataset; generate_dataset(output_dir='dataset', samples_per_class=60, save_preset_samples=True)"

if not exist model.joblib (
  echo Training model (may take several minutes)...
  %VENV_PY% -c "from model import TrafficSignClassifier; TrafficSignClassifier().train(samples_per_class=60)"
) else (
  echo model.joblib found — skipping training. Delete model.joblib to retrain.
)

echo Launching app...
%VENV_PY% app.py
