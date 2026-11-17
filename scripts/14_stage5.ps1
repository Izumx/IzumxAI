# Stage 5: IzumxAI-9B — дистилляция от Qwen3.8-27B (abliterated) + abliteration.
#
#   1. учитель решает задачи, ответы проверяются программно (математика, логика,
#      код, кибербез) — в обучение идёт только верное;
#   2. QLoRA в Qwen3.5-9B (влезает в 6-8 ГБ VRAM после квантования);
#   3. слияние адаптера и удаление «направления отказа»;
#   4. экспорт GGUF Q4_K_M / Q5_K_M и регистрация в Ollama;
#   5. честный замер против учителя и базовой модели.
#
# Пример:
#   powershell -ExecutionPolicy Bypass -File scripts\14_stage5.ps1 -N 600
param(
    [int]$N = 600,
    [int]$NVal = 60,
    [string]$Teacher = "hf.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:UD-Q3_K_XL",
    [string]$Student = "Qwen/Qwen3.5-9B",
    [string]$Mix = "math:20,logic:10,code:30,cyber:40",
    [string]$Name = "izumxai:9b",
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

$train = "data\distill\stage5_train_split.jsonl"
$val = "data\distill\stage5_val.jsonl"

if (-not $SkipData) {
    Write-Host "`n[1/6] генерация обучающих данных от учителя" -ForegroundColor Cyan
    & $py -u -m izumxai.distill.gen_data --model $Teacher --n $N --out $train `
        --mix $Mix --level mix --seed 5 --num-ctx 8192 --num-predict 700
    Write-Host "`n[2/6] деление на train/val (стратифицированно, 6%)" -ForegroundColor Cyan
    & $py -m izumxai.distill.split_data --data $train --train "data\distill\stage5_train_split.jsonl" --val $val
}

if (-not $SkipTrain) {
    Write-Host "`n[3/6] QLoRA в $Student" -ForegroundColor Cyan
    & $py -m izumxai.distill.train_lora --model $Student --data $train --val $val `
        --out runs\izumxai-9b-lora --max-len 2048 --vram-budget-gb 15.0 `
        --lora-r 32 --lora-alpha 64 --epochs 2 --grad-accum 8 --lr 2e-4 `
        --eval-every 100 --save-every 200
}

if (-not $SkipExport) {
    Write-Host "`n[4/6] слияние адаптера" -ForegroundColor Cyan
    & $py -m izumxai.distill.merge_export --adapter runs\izumxai-9b-lora\final `
        --base $Student --out models\izumxai-9b-merged --name $Name

    Write-Host "`n[5/6] abliteration" -ForegroundColor Cyan
    # scale 1.4: отказы 85% -> ~15%, качество на 88 задачах не падает (90%).
    # 1.45 даёт ~10% отказов ценой 4 лишних провалов, 1.6 — 2% отказов и 74%.
    # --device cpu: bf16-модель (19 ГБ) целиком в RAM, иначе часть слоёв
    # остаётся meta-тензорами и правка весов падает.
    & $py -m izumxai.distill.abliterate --model models\izumxai-9b-merged `
        --out models\izumxai-9b-free --scale 1.4 --device cpu

    Write-Host "`n[6/6] GGUF + Ollama" -ForegroundColor Cyan
    foreach ($q in @("Q4_K_M", "Q5_K_M")) {
        & $py -m izumxai.distill.merge_export --merged --base models\izumxai-9b-free `
            --out "models\izumxai-9b-$q" --gguf --quant $q --stem "izumxai-9b-$q" --name $Name --no-template
    }
}

Write-Host "`nГотово. Замер:" -ForegroundColor Green
Write-Host "  $py -m izumxai.eval.bench --models $Name,$Teacher,$Student ``"
Write-Host "     --n-cyber 24 --num-predict 1100 --level mix --seed 42"
Write-Host "  $py -m izumxai.eval.refusal --models $Name"
