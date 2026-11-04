# SFT profile intended for 4 GiB laptops; full-model peak must be measured.
param([string]$Data = "data/distill/train.jsonl",
      [string]$Split = "data/distill/4gb",
      [string]$Out = "runs/izumxai-2b-4gb",
      [double]$BudgetGiB = 3.25,
      [int]$MaxSteps = -1)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
if (Test-Path "$Out/final") { throw "Output already contains an adapter; choose another -Out" }

& $py -m izumxai.distill.prepare_sft --data $Data --out $Split
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed" }

& $py -m izumxai.distill.train_lora --model Qwen/Qwen3-1.7B `
    --data "$Split/train.jsonl" --val "$Split/val.jsonl" --out $Out `
    --epochs 3 --batch-size 1 --grad-accum 16 --max-len 512 `
    --lora-r 16 --lora-alpha 32 --lr 1e-4 --vram-budget-gb $BudgetGiB `
    --eval-every 10 --save-every 50 --max-steps $MaxSteps
if ($LASTEXITCODE -ne 0) { throw "Training failed; inspect the log before exporting" }
