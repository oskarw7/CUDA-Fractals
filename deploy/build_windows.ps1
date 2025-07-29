# deploy\build.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[*] Starting build..."
Write-Host "[*] Make sure you have CUDA toolkit installed."

# Move to project root from deploy\
Set-Location (Join-Path $PSScriptRoot "..")

# Create virtual environment if it doesn't exist
if (-Not (Test-Path ".venv")) {
    Write-Host "[*] Creating virtual environment..."
    python -m venv .venv
}

# Activate the virtual environment
Write-Host "[*] Activating virtual environment..."
. ".\.venv\Scripts\Activate.ps1"

Write-Host "[*] Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller

Write-Host "[*] Building with PyInstaller..."
pyinstaller --onefile --name fractal_app src\main.py

Write-Host "[*] Performing cleanup..."
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Force fractal_app.spec -ErrorAction SilentlyContinue

Write-Host "`n[✓] Done. Executable is at app\fractal_app.exe"
