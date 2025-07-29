#!/bin/bash
set -e

# Move to the project root from deploy/
cd "$(dirname "$0")/.."

echo "[*] Starting build..."
echo "[*] Make sure you have CUDA toolkit installed."

# Optional: Create and activate virtual environment (at project root)
if [ ! -d ".venv" ]; then
  echo "[*] Creating virtual environment..."
  python3 -m venv .venv
else
  echo "[*] Virtual environment already exists."
fi
source .venv/bin/activate

echo "[*] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt pyinstaller

echo "[*] Building with PyInstaller..."
pyinstaller --onefile --name fractal_generator src/main.py

echo "[*] Performing cleanup..."
mkdir -p app
mv dist/fractal_generator app
rm -rf dist build fractal_generator.spec

echo "[✓] Done. Executable is at app/fractal_generator"