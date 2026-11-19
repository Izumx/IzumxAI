# Compare v6 against v4 on both axes that decide a release.
#
#   1. agentic harness, pure model (--escalate-after 0), two passes
#   2. own 88-task bench + refusal suite
#
# v4 is the current release: 80/88 and 0/41 refusals. v6 replaces it only if it
# is better on both axes and does not regress the text bench.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\20_v6_eval.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\20_v6_eval.ps1 -SkipBench

param(
    [string]$Model = "izumxai:9b-v6",
    [string]$Baseline = "izumxai:9b-v4",
    [int]$Samples = 3,
    [int]$Passes = 2,
    [switch]$SkipHarness,
    [switch]$SkipBench
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = $root
$env:OLLAMA_MODELS = "D:\ollama\models"
$env:PYTHONUTF8 = "1"

function Assert-Ok([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$step failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipHarness) {
    Write-Host "`n[1/3] Agentic harness, pure model" -ForegroundColor Cyan
    for ($i = 1; $i -le $Passes; $i++) {
        Write-Host "  pass $i" -ForegroundColor DarkGray
        & $py -u -m izumxai.agentic.harness --model $Baseline --model $Model `
            --escalate-after 0 --samples $Samples --max-steps 8
        Assert-Ok "harness pass $i"
    }
    & $py -u -m izumxai.agentic.report
    Assert-Ok "agentic report"
}

if (-not $SkipBench) {
    Write-Host "`n[2/3] Own bench (88 tasks)" -ForegroundColor Cyan
    & $py -u -m izumxai.eval.bench --models $Model `
        --n-cyber 24 --num-predict 1100 --level mix --seed 42 `
        --out "runs\bench_$($Model -replace '[:/]','_').json"
    Assert-Ok "bench"

    Write-Host "`n[3/3] Refusals" -ForegroundColor Cyan
    & $py -u -m izumxai.eval.refusal --models $Model `
        --out "runs\refusal_$($Model -replace '[:/]','_').json"
    Assert-Ok "refusal"
}

Write-Host "`nDone." -ForegroundColor Green
