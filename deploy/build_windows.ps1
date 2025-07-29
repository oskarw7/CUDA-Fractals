Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[*] Starting build..."
Write-Host "[*] Make sure you have CUDA toolkit installed."

# Move to project root from deploy/
Set-Location (Join-Path $PSScriptRoot "..")

# Optional: Create and activate virtual environment (at project root)
if (-Not (Test-Path ".venv")) {
    Write-Host "[*] Creating virtual environment..."
    python -m venv .venv
} else {
    Write-Host "[*] Virtual environment already exists."
}

# Activate virtual environment
. .\.venv\Scripts\Activate.ps1

Write-Host "[*] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt pyinstaller

Write-Host "[*] Building with PyInstaller..."
pyinstaller --onefile --name fractal_generator src\main.py

Write-Host "[*] Performing cleanup..."

# Create output directory and move executable
New-Item -ItemType Directory -Force -Path app | Out-Null
Move-Item -Force dist\fractal_generator.exe app\

# Clean up build artifacts
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Remove-Item -Force fractal_generator.spec -ErrorAction SilentlyContinue

Write-Host "[✓] Done. Executable is at app\fractal_generator.exe"
