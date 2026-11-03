# ORPO: пары «хороший/плохой ответ» с самой модели + дообучение без референса
param([int]$N = 200, [int]$K = 4, [string]$Level = "hard",
      [string]$Init = "runs/izumxai-2b-lora/final",
      [string]$Out = "runs/izumxai-2b-orpo")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

# 1) текущая модель решает задачи K раз, ответы проверяются кодом
& $py -m izumxai.distill.gen_prefs --model izumxai:2b --n $N --k $K --level $Level `
    --out data/distill/prefs.jsonl --rft data/distill/rft.jsonl

# 2) ORPO от адаптера после SFT (модель-референс не нужна)
& $py -m izumxai.distill.train_orpo --pairs data/distill/prefs.jsonl `
    --init $Init --out $Out --epochs 2 --grad-accum 8 --lr 5e-5 --lam 0.1
