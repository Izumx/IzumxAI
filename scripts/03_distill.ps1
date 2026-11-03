# IzumxAI:2B - данные от учителя + QLoRA-дистилляция
param([int]$N = 1000, [string]$Level = "hard")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

# 1) учитель qwen3:8b решает задачи, ответы проверяются кодом
& $py -m izumxai.distill.gen_data --model qwen3:8b --n $N --level $Level `
    --out data/distill/train.jsonl --num-ctx 2048 --num-predict 512

# 2) чистим LaTeX и дубли условий
& $py -m izumxai.distill.clean_data --in data/distill/train.jsonl

# 3) ученик 1.7B учится на верифицированных парах (QLoRA, влезает в 6 ГБ VRAM)
& $py -m izumxai.distill.train_lora --model Qwen/Qwen3-1.7B `
    --data data/distill/train.jsonl --out runs/izumxai-2b-lora `
    --epochs 3 --batch-size 1 --grad-accum 8 --max-len 1024 `
    --lr 2e-4 --lora-r 64 --lora-alpha 128
