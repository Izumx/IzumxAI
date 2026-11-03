# Слияние LoRA -> GGUF -> ollama
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"

& $py -m izumxai.distill.merge_export --adapter runs/izumxai-2b-lora/final `
    --base Qwen/Qwen3-1.7B --out models/izumxai-2b --gguf

if (Test-Path "models/izumxai-2b/Modelfile") {
    ollama create izumxai:2b -f "models/izumxai-2b/Modelfile"
    ollama run izumxai:2b "Привет! Кто ты?"
}
