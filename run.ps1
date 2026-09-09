# run.ps1 - setup, generate dataset, train model, and run app
# Usage: Right-click -> "Run with PowerShell" or open PowerShell in this folder and run: .\run.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

Write-Host "NeuroSign run script - venv -> deps -> dataset -> train -> run" -ForegroundColor Cyan

# Find a Python launcher
$pyCmd = $null
try {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { $pyCmd = 'py -3' }
} catch {}

if (-not $pyCmd) {
    try {
        $p = Get-Command python -ErrorAction SilentlyContinue
        if ($p) { $pyCmd = 'python' }
    } catch {}
}

if (-not $pyCmd) {
    Write-Error "No 'py' or 'python' launcher found. Please install Python 3.10+ and ensure 'py' or 'python' is on PATH."; exit 1
}

# Create virtual environment if missing
if (-not (Test-Path -Path .venv)) {
    Write-Host "Creating virtual environment (.venv)..."
    & $pyCmd -m venv .venv
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    # Fallback to launcher if venv creation used a different python
    $venvPython = "$pyCmd"
}

Write-Host "Upgrading pip and installing dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

# Generate dataset (moderate size)
Write-Host "Generating dataset (this may take a minute)..."
& $venvPython -c "from dataset_generator import generate_dataset; generate_dataset(output_dir='dataset', samples_per_class=60, save_preset_samples=True)"

# Train model if not present
if (-not (Test-Path -Path 'model.joblib')) {
    Write-Host "Training RandomForest model (this may take several minutes)..."
    & $venvPython -c "from model import TrafficSignClassifier; TrafficSignClassifier().train(samples_per_class=60)"
} else {
    Write-Host "Found existing model.joblib — skipping training. To retrain remove model.joblib and re-run."
}

# Launch the app
Write-Host "Launching NeuroSign GUI..."
& $venvPython app.py
