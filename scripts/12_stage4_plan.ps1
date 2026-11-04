# Stage 4: решения с проверкой крайних случаев -> QLoRA -> GGUF -> ollama (izumxai:2b-v4)
param([int]$N = 300, [string]$Tag = "izumxai:2b-v4",
      [string]$Adapter = "runs/izumxai-2b-plan-lora",
      [int]$Epochs = 2, [int]$MaxLen = 1024)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

# 1) генерация: учитель решает, затем проверяет крайние случаи (можно прерывать)
& $py -u -m izumxai.distill.gen_plan --n $N --out data/distill/stage4-plan/train.jsonl
if ($LASTEXITCODE -ne 0) { throw "gen_plan failed" }

# 2) сборка train/val: примеры с проверкой повторяются, обычные ответы сохраняются
& $py -m izumxai.distill.assemble_stage4 `
    --think data/distill/stage4-plan/train.jsonl `
    --plain data/distill/stage3-smel/train.jsonl `
    --plain-val data/distill/stage3-smel/val.jsonl `
    --out data/distill/stage4
if ($LASTEXITCODE -ne 0) { throw "assemble_stage4 failed" }

# 3) QLoRA от базовой Qwen3-1.7B
& $py -u -m izumxai.distill.train_lora --data data/distill/stage4/train.jsonl `
    --val data/distill/stage4/val.jsonl --out $Adapter --epochs $Epochs `
    --batch-size 1 --grad-accum 8 --max-len $MaxLen --lr 2e-4 `
    --lora-r 32 --lora-alpha 64 --vram-budget-gb 4.5 --eval-every 40
if ($LASTEXITCODE -ne 0) { throw "train_lora failed" }

# 4) merge -> GGUF f16 -> Q4_K_M
$name = ($Tag -split ":")[0]
& $py -m izumxai.distill.merge_export --adapter "$Adapter/final" `
    --base Qwen/Qwen3-1.7B --out "models/$name" --gguf --quant Q4_K_M
if ($LASTEXITCODE -ne 0) { throw "merge_export failed" }

# 5) импорт в Ollama
& ollama create $Tag -f "models/$name/Modelfile"
if ($LASTEXITCODE -ne 0) { throw "ollama create failed" }
