# Stage 7: izumxai-9b-v5 - cyber-SFT on the *full* Vyber07 dataset.
#
# Difference from 15_v4.ps1: external data is the streamed selection from
# prepare_vyber_v5.py (per-domain hash sampling, deduped against v4), and the
# adapter is trained for a single epoch because the dataset is ~4x larger.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\16_v5.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\16_v5.ps1 -SkipData

param(
    [string]$Student = "Qwen/Qwen3.5-9B",
    [string]$Name = "izumxai:9b-v5",
    [int]$VyberCap = 12000,
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

$train = "data\distill\stage5_v5_train.jsonl"
$val = "data\distill\stage5_v5_val.jsonl"
$adapter = "runs\izumxai-9b-lora-v5"
$merged = "models\izumxai-9b-merged-v5"
$free = "models\izumxai-9b-free-v5"

if (-not $SkipData) {
    Write-Host "`n[1/7] Select full Vyber07 data (v5)" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.prepare_vyber_v5 --raw data\external\vyber\_raw_v5 `
        --out data\distill\vyber_v5.jsonl
    Assert-Ok "prepare_vyber_v5"

    Write-Host "`n[2/7] Mix stage5 + external cyber SFT" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.mix_v5 --vyber-cap $VyberCap `
        --out-train $train --out-val $val
    Assert-Ok "mix_v5"
}

if (-not $SkipTrain) {
    Write-Host "`n[3/7] QLoRA on $Student" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.train_lora --model $Student --data $train --val $val `
        --out $adapter --max-len 2048 --vram-budget-gb 15.0 `
        --lora-r 32 --lora-alpha 64 --epochs $Epochs --grad-accum 8 --lr 2e-4 `
        --eval-every 200 --save-every 500
    Assert-Ok "train_lora"
}

if (-not $SkipExport) {
    Write-Host "`n[4/7] Merge adapter" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.merge_export --adapter "$adapter\final" `
        --base $Student --out $merged --name $Name
    Assert-Ok "merge_export"

    Write-Host "`n[5/7] Abliteration (scale $Scale)" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.abliterate --model $merged `
        --out $free --scale $Scale --device cpu
    Assert-Ok "abliterate"

    Write-Host "`n[6/7] GGUF + quantize" -ForegroundColor Cyan
    foreach ($q in @("Q4_K_M", "Q5_K_M")) {
        & $py -u -m izumxai.distill.merge_export --merged --base $free `
            --out "models\izumxai-9b-v5-$q" --gguf --quant $q `
            --stem "izumxai-9b-v5-$q" --name $Name --no-template
        Assert-Ok "export $q"
    }

    Write-Host "`n[7/7] Ollama tags" -ForegroundColor Cyan
    $q4 = (Resolve-Path "models\izumxai-9b-v5-Q4_K_M\Modelfile").Path
    & ollama create "izumxai:9b-v5" -f $q4
    Assert-Ok "ollama create 9b-v5"
    $q5 = (Resolve-Path "models\izumxai-9b-v5-Q5_K_M\Modelfile").Path
    & ollama create "izumxai:9b-v5-q5" -f $q5
    Assert-Ok "ollama create 9b-v5-q5"
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  $py -m izumxai.eval.bench --models $Name --n-cyber 24 --num-predict 1100 --level mix --seed 42"
Write-Host "  python ..\PurpleLlama\run_suite.py --model $Name --tag v5"
