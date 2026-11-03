# Сравнение с qwen3:8b на одних и тех же задачах
param([int]$Math = 30, [int]$Code = 12, [int]$Instr = 10)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$py = ".\.venv\Scripts\python.exe"

& $py -m izumxai.eval.bench --models "izumxai:2b,qwen3:8b" `
    --n-math $Math --n-code $Code --n-instr $Instr
