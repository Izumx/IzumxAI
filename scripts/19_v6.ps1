# Stage 8: izumxai-9b-v6 - v4 recipe plus execution-verified agentic SFT.
#
# Why: the agentic harness measured three gaps with the model alone
# (--escalate-after 0): eval()->RCE 0/8, no change of construction after
# "NOT SOLVED", search tools never called.  data/distill/agentic_sft.jsonl is
# built by izumxai.agentic.synth, which runs every gold action against a real
# service and keeps only rows the oracle confirms.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\19_v6.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\19_v6.ps1 -SkipData -SkipTrain

param(
    [string]$Student = "Qwen/Qwen3.5-9B",
    [string]$Name = "izumxai:9b-v6",
    [int]$VyberCap = 2200,
    [int]$AgenticRepeat = 3,
    [double]$Scale = 1.4,
    [int]$Epochs = 2,
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
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

function Assert-Ok([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$step failed with exit code $LASTEXITCODE - stopping pipeline"
    }
}

$train = "data\distill\stage5_v6_train.jsonl"
$val = "data\distill\stage5_v6_val.jsonl"
$adapter = "runs\izumxai-9b-lora-v6"
$merged = "models\izumxai-9b-merged-v6"
$free = "models\izumxai-9b-free-v6"
$slug = "izumxai-9b-v6"

if (-not $SkipData) {
    Write-Host "`n[1/6] Mix stage5 + Vyber07 + verified agentic SFT" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.mix_v4 --vyber-cap $VyberCap `
        --agentic data\distill\agentic_sft.jsonl --agentic-repeat $AgenticRepeat `
        --out-train $train --out-val $val
    Assert-Ok "mix_v4"
}

if (-not $SkipTrain) {
    Write-Host "`n[2/6] QLoRA on $Student" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.train_lora --model $Student --data $train --val $val `
        --out $adapter --max-len 2048 --vram-budget-gb 15.0 `
        --lora-r 32 --lora-alpha 64 --epochs $Epochs --grad-accum 8 --lr 2e-4 `
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
            --out "models\$slug-$q" --gguf --quant $q `
            --stem "$slug-$q" --name $Name --no-template
        Assert-Ok "export $q"
    }

    Write-Host "`n[6/6] Ollama tags" -ForegroundColor Cyan
    $q4 = (Resolve-Path "models\$slug-Q4_K_M\Modelfile").Path
    & ollama create $Name -f $q4
    Assert-Ok "ollama create $Name"
    $q5 = (Resolve-Path "models\$slug-Q5_K_M\Modelfile").Path
    & ollama create "$Name-q5" -f $q5
    Assert-Ok "ollama create $Name-q5"
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  $py -m izumxai.agentic.harness --model $Name --escalate-after 0 --samples 3"
Write-Host "  $py -m izumxai.eval.bench --models $Name --n-cyber 24 --num-predict 1100 --level mix --seed 42"
