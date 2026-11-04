# Замер stage4: единый бенч, быстрый прогон HumanEval и полный прогон
param([string]$Model = "izumxai:2b-v4", [int]$Rounds = 3)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

# 1) общий бенч на hard-задачах против qwen3:8b
& $py -m izumxai.eval.bench --models "$Model,qwen3:8b" --level hard `
    --n-math 24 --n-logic 12 --n-code 16 --n-instr 12
if ($LASTEXITCODE -ne 0) { throw "bench failed" }

# 2) быстрый сигнал на 12 задачах HumanEval, один проход
& $py -u -m izumxai.eval.solve --model $Model --think --rounds 1 `
    --tasks data/eval/humaneval_stage2_12.jsonl --out runs/solver/stage4-smoke.json
if ($LASTEXITCODE -ne 0) { throw "smoke solve failed" }

# 3) полный HumanEval с самопроверкой
& $py -u -m izumxai.eval.solve --model $Model --think --rounds $Rounds `
    --num-predict 2048 --out "runs/solver/humaneval-stage4-r$Rounds.json"
if ($LASTEXITCODE -ne 0) { throw "full solve failed" }
