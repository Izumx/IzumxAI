param([string]$Model="izumxai:2b-think", [int]$Rounds=3, [int]$NumPredict=2048,
      [string]$Out="runs/solver/humaneval-think-r3.json")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root/.cache/hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
& ollama create izumxai:2b-think -f models/izumxai-think/Modelfile
if ($LASTEXITCODE -ne 0) { throw "ollama create failed" }
& $py -u -m izumxai.eval.solve --model $Model --think --rounds $Rounds `
    --num-predict $NumPredict --out $Out
if ($LASTEXITCODE -ne 0) { throw "solver evaluation failed" }
