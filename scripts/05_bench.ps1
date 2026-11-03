# Сравнение с qwen3:8b на одних и тех же задачах
param([int]$Math = 24, [int]$Logic = 12, [int]$Code = 16, [int]$Instr = 12,
      [string]$Level = "hard")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

& $py -m izumxai.eval.bench --models "izumxai:2b,qwen3:8b" --level $Level `
    --n-math $Math --n-logic $Logic --n-code $Code --n-instr $Instr
