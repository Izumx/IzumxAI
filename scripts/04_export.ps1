# Слияние LoRA -> GGUF -> ollama (izumxai:2b)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = "$root\.cache\hf"
$py = ".\.venv\Scripts\python.exe"

# 1) конвертер llama.cpp (нужен только один раз)
if (-not (Test-Path "tools\llama_src\conversion")) {
    git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp tools\llama_src
    Push-Location tools\llama_src
    git sparse-checkout set --skip-checks conversion convert_hf_to_gguf.py gguf-py
    Pop-Location
}
# gguf-py из репозитория (в PyPI отстаёт от конвертера)
& $py -m pip install -q -e "tools\llama_src\gguf-py"
& $py -m pip install -q sentencepiece

# 2) llama-quantize (CPU-сборка достаточна для конвертации)
if (-not (Test-Path "tools\llama_bin\llama-quantize.exe")) {
    $tag = "b12069"
    New-Item -ItemType Directory -Force tools\llama_bin | Out-Null
    Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cpu-x64.zip" -OutFile tools\llama_bin.zip -UseBasicParsing
    Expand-Archive tools\llama_bin.zip -DestinationPath tools\llama_bin -Force
}

# 3) merge + конвертация + квантование Q4_K_M
& $py -m izumxai.distill.merge_export --adapter runs/izumxai-2b-lora/final `
    --base Qwen/Qwen3-1.7B --out models/izumxai-2b --gguf --quant Q4_K_M

# 4) импорт в Ollama
if (Test-Path "models/izumxai-2b/Modelfile") {
    ollama create izumxai:2b -f "models/izumxai-2b/Modelfile"
    ollama run izumxai:2b "Привет! Кто ты?"
}
