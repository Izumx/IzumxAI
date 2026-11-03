# Окружение IzumxAI (Windows / RTX 4050 6GB)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
$env:PIP_CACHE_DIR = "$root\.cache\pip"
$env:HF_HOME = "$root\.cache\hf"

& $py -m pip install --upgrade pip setuptools wheel
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt
& $py -m pip install sentencepiece

& $py -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

Write-Host ""
Write-Host "Проверка пайплайна без обучения:"
& $py tests\smoke.py
