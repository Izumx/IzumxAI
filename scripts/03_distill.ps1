# IzumxAI:2B - данные от учителя + QLoRA-дистилляция
param([int]$N = 400)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$py = ".\.venv\Scripts\python.exe"

# 1) учитель qwen3:8b решает задачи, ответы проверяются кодом
& $py -m izumxai.distill.gen_data --model qwen3:8b --n $N `
    --out data/distill/train.jsonl --num-ctx 1024 --num-predict 384

# 2) ученик 1.7B учится на верифицированных парах (QLoRA, влезает в 6 ГБ VRAM)
& $py -m izumxai.distill.train_lora --model Qwen/Qwen3-1.7B `
    --data data/distill/train.jsonl --out runs/izumxai-2b-lora `
    --epochs 4 --batch-size 1 --grad-accum 4 --max-len 1024 `
    --lr 2e-4 --lora-r 64 --lora-alpha 128
