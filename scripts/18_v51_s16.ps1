# Re-abliterate the finished v5.1 merge at a different scale and rebuild Q4.
# Purpose: v5.1 at 1.4 refused 3/41 prompts while winning on every capability
# metric; at 1.6 refusals hit 0/41 but the own bench lost three tasks. The
# strength is the only knob involved, so intermediate points are swept here
# without touching training.
#
#   powershell -ExecutionPolicy Bypass -File scripts\18_v51_s16.ps1 -Scale 1.5
param(
    [double]$Scale = 1.6,
    [string]$Tag = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = $root
$env:HF_HOME = "$root\.cache\hf"
$env:OLLAMA_MODELS = "D:\ollama\models"

function Assert-Ok([string]$step) {
    if ($LASTEXITCODE -ne 0) { throw "$step failed with exit code $LASTEXITCODE" }
}

$suffix = ($Scale.ToString("0.0", [System.Globalization.CultureInfo]::InvariantCulture)).Replace(".", "")
$slug = "izumxai-9b-v51s$suffix"
if ($Tag) { $slug = $Tag }
$free = "models\izumxai-9b-free-v51-s$suffix"

Write-Host "`n[1/3] Abliteration (scale $Scale -> $slug)" -ForegroundColor Cyan
& $py -u -m izumxai.distill.abliterate --model "models\izumxai-9b-merged-v51" `
    --out $free --scale $Scale --device cpu
Assert-Ok "abliterate"

Write-Host "`n[2/3] GGUF Q4_K_M" -ForegroundColor Cyan
& $py -u -m izumxai.distill.merge_export --merged --base $free `
    --out "models\$slug-Q4_K_M" --gguf --quant Q4_K_M `
    --stem "$slug-Q4_K_M" --name "izumxai:$($slug.Replace('izumxai-9b-','9b-'))" --no-template
Assert-Ok "gguf"

Write-Host "`n[3/3] Ollama tag" -ForegroundColor Cyan
& ollama create ($slug.Replace("izumxai-9b-", "izumxai:9b-")) -f (Resolve-Path "models\$slug-Q4_K_M\Modelfile").Path
Assert-Ok "ollama create"

Write-Host "`nDone: $($slug.Replace('izumxai-9b-', 'izumxai:9b-'))" -ForegroundColor Green
