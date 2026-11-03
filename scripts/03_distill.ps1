# IzumxAI:2B — данные от учителя + QLoRA-дистилляция
param([int]$N = 400)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"

# 1) учитель qwen3:8b решает задачи, ответы проверяются кодом
& $py -m izumxai.distill.gen_data --model qwen3:8b --n $N --out data/distill/train.jsonl

# 2) ученик 1.7B учится на верифицированных парах (QLoRA, влезает в 6 ГБ)
& $py -m izumxai.distill.train_lora --model Qwen/Qwen3-1.7B `
    --data data/distill/train.jsonl --out runs/izumxai-2b-lora `
    --epochs 3 --batch-size 1 --grad-accum 16 --max-len 2048
