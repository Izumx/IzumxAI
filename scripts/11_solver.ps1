param([string]$Model="izumxai:4b-think", [string]$Modelfile="modelfiles/izumxai-4b-think.Modelfile",
      [int]$Rounds=3, [int]$NumPredict=2048,
      [string]$Out="runs/solver/humaneval-4b-think-r3.json")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root/.cache/hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
if ($Modelfile -like "modelfiles/izumxai-4b*") { & ollama pull qwen3:4b }
& ollama create $Model -f $Modelfile
if ($LASTEXITCODE -ne 0) { throw "ollama create failed" }
& $py -u -m izumxai.eval.solve --model $Model --think --rounds $Rounds `
    --num-predict $NumPredict --out $Out
if ($LASTEXITCODE -ne 0) { throw "solver evaluation failed" }
