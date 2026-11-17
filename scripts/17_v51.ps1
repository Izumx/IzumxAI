# Stage 7.1: izumxai-9b-v51 - v5 plus the verified stage5 part repeated twice.
#
# Why: v5 (full Vyber07 data) reached perfect code and zero refusals but lost
# instruction discipline (7/12 on the own bench vs 10/12 for v4): answers ran
# 27-30 words against a 25-word limit and word counting was guessed. Repeating
# the verified part of the mix restores that discipline without dropping the
# external cyber data.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\17_v51.ps1 -SkipData

param(
    [string]$Student = "Qwen/Qwen3.5-9B",
    [string]$Name = "izumxai:9b-v51",
    [int]$VyberCap = 8000,
    [int]$Stage5Repeat = 2,
    [double]$Scale = 1.4,
    [int]$Epochs = 1,
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
$env:OLLAMA_MODELS = "D:\ollama\models"

function Assert-Ok([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$step failed with exit code $LASTEXITCODE - stopping pipeline"
    }
}

$train = "data\distill\stage5_v51_train.jsonl"
$val = "data\distill\stage5_v51_val.jsonl"
$adapter = "runs\izumxai-9b-lora-v51"
$merged = "models\izumxai-9b-merged-v51"
$free = "models\izumxai-9b-free-v51"
$slug = "izumxai-9b-v51"

if (-not $SkipData) {
    Write-Host "`n[1/7] Mix stage5 (x$Stage5Repeat) + external cyber SFT" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.mix_v5 --vyber-cap $VyberCap --stage5-repeat $Stage5Repeat `
        --out-train $train --out-val $val
    Assert-Ok "mix_v5"
}

if (-not $SkipTrain) {
    Write-Host "`n[2/7] QLoRA on $Student" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.train_lora --model $Student --data $train --val $val `
        --out $adapter --max-len 2048 --vram-budget-gb 15.0 `
        --lora-r 32 --lora-alpha 64 --epochs $Epochs --grad-accum 8 --lr 2e-4 `
        --eval-every 200 --save-every 500
    Assert-Ok "train_lora"
}

if (-not $SkipExport) {
    Write-Host "`n[3/7] Merge adapter" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.merge_export --adapter "$adapter\final" `
        --base $Student --out $merged --name $Name
    Assert-Ok "merge_export"

    Write-Host "`n[4/7] Abliteration (scale $Scale)" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.abliterate --model $merged `
        --out $free --scale $Scale --device cpu
    Assert-Ok "abliterate"

    Write-Host "`n[5/7] GGUF + quantize" -ForegroundColor Cyan
    foreach ($q in @("Q4_K_M", "Q5_K_M")) {
        & $py -u -m izumxai.distill.merge_export --merged --base $free `
            --out "models\$slug-$q" --gguf --quant $q `
            --stem "$slug-$q" --name $Name --no-template
        Assert-Ok "export $q"
    }

    Write-Host "`n[6/7] Ollama tags" -ForegroundColor Cyan
    $q4 = (Resolve-Path "models\$slug-Q4_K_M\Modelfile").Path
    & ollama create $Name -f $q4
    Assert-Ok "ollama create $Name"
    $q5 = (Resolve-Path "models\$slug-Q5_K_M\Modelfile").Path
    & ollama create "$Name-q5" -f $q5
    Assert-Ok "ollama create $Name-q5"
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  $py -m izumxai.eval.bench --models $Name --n-cyber 24 --num-predict 1100 --level mix --seed 42"
Write-Host "  python ..\PurpleLlama\run_suite.py --model $Name --tag v51"
