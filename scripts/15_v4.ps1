# Stage 6: izumxai-9b-v4 - cyber-SFT on top of the verified stage5 dataset.
#
# Pipeline:
#   1. mix verified stage5 data with curated external cyber SFT (Vyber07)
#   2. QLoRA on Qwen/Qwen3.5-9B (4-bit, 2048 ctx, no truncated examples)
#   3. merge adapter, abliterate (scale 1.4, same as v3)
#   4. export Q4_K_M and Q5_K_M GGUF, create Ollama tags
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\15_v4.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\15_v4.ps1 -SkipData -SkipTrain

param(
    [string]$Student = "Qwen/Qwen3.5-9B",
    [string]$Name = "izumxai:9b-v4",
    [int]$VyberCap = 2200,
    [double]$Scale = 1.4,
    [switch]$SkipData,
    [switch]$SkipTrain,
    [switch]$SkipExport
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"
$env:HF_HOME = "$root\.cache\hf"
$env:PYTHONPATH = $root

# Native commands do not trigger $ErrorActionPreference, so check every step.
function Assert-Ok([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$step failed with exit code $LASTEXITCODE - stopping pipeline"
    }
}

$train = "data\distill\stage5_v4_train.jsonl"
$val = "data\distill\stage5_v4_val.jsonl"
$adapter = "runs\izumxai-9b-lora-v4"
$merged = "models\izumxai-9b-merged-v4"
$free = "models\izumxai-9b-free-v4"

if (-not $SkipData) {
    Write-Host "`n[1/6] Mix stage5 + external cyber SFT" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.mix_v4 --vyber-cap $VyberCap `
        --out-train $train --out-val $val
    Assert-Ok "mix_v4"
}

if (-not $SkipTrain) {
    Write-Host "`n[2/6] QLoRA on $Student" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.train_lora --model $Student --data $train --val $val `
        --out $adapter --max-len 2048 --vram-budget-gb 15.0 `
        --lora-r 32 --lora-alpha 64 --epochs 2 --grad-accum 8 --lr 2e-4 `
        --eval-every 100 --save-every 200
    Assert-Ok "train_lora"
}

if (-not $SkipExport) {
    Write-Host "`n[3/6] Merge adapter" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.merge_export --adapter "$adapter\final" `
        --base $Student --out $merged --name $Name
    Assert-Ok "merge_export"

    Write-Host "`n[4/6] Abliteration (scale $Scale)" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.abliterate --model $merged `
        --out $free --scale $Scale --device cpu
    Assert-Ok "abliterate"

    Write-Host "`n[5/6] GGUF + quantize" -ForegroundColor Cyan
    foreach ($q in @("Q4_K_M", "Q5_K_M")) {
        & $py -u -m izumxai.distill.merge_export --merged --base $free `
            --out "models\izumxai-9b-v4-$q" --gguf --quant $q `
            --stem "izumxai-9b-v4-$q" --name $Name --no-template
        Assert-Ok "export $q"
    }

    Write-Host "`n[6/6] Ollama tags" -ForegroundColor Cyan
    $q4 = (Resolve-Path "models\izumxai-9b-v4-Q4_K_M\Modelfile").Path
    & ollama create "izumxai:9b-v4" -f $q4
    Assert-Ok "ollama create 9b-v4"
    $q5 = (Resolve-Path "models\izumxai-9b-v4-Q5_K_M\Modelfile").Path
    & ollama create "izumxai:9b-v4-q5" -f $q5
    Assert-Ok "ollama create 9b-v4-q5"
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  $py -m izumxai.eval.bench --models $Name --n-cyber 24 --num-predict 1100 --level mix --seed 42"
Write-Host "  python ..\PurpleLlama\run_suite.py --model izumxai:9b-v4"
