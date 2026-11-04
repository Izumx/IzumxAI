param([string]$Out="runs/izumxai-smel-dense", [int]$Steps=64)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& "$PSScriptRoot/09_smel_setup.ps1"
$env:HF_HOME = "$root/.cache/hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
& $py -m izumxai.distill.prepare_smel_data --out data/distill/stage3-smel
if ($LASTEXITCODE -ne 0) { throw "SMEL training data preparation failed" }
& $py -u -m izumxai.distill.train_smel --model models/izumxai-2b `
    --data data/distill/stage3-smel/train.jsonl --val data/distill/stage3-smel/val.jsonl `
    --cache data/cache/smel-dense --out $Out --cut 20 --rank 16 --steps $Steps `
    --grad-accum 8 --lr 1e-5 --code-probability 0.7 --vram-budget-gb 2.8
if ($LASTEXITCODE -ne 0) { throw "SMEL dense training failed" }
