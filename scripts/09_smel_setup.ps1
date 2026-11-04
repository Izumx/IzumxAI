# Pinned upstream source; imports only smel.loss/quant, no full MoE dependencies.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$revision = "74a4635d5f7733baae9915d291573fae94a0499b"
$patch = Join-Path $root "patches/smel-ft-74a4635-loss-validation.patch"
if (-not (Test-Path "tools/smel-ft/.git")) {
    git clone --filter=blob:none --no-checkout https://github.com/artikulus1301/smel-ft.git tools/smel-ft
    if ($LASTEXITCODE -ne 0) { throw "SMEL FT clone failed" }
    git -C tools/smel-ft checkout $revision
    if ($LASTEXITCODE -ne 0) { throw "Pinned SMEL FT checkout failed" }
}
$current = git -C tools/smel-ft rev-parse HEAD
if ($current.Trim() -ne $revision) { throw "SMEL FT revision differs; preserve local changes and use the pinned revision" }
git -C tools/smel-ft apply --check $patch 2>$null
if ($LASTEXITCODE -eq 0) {
    git -C tools/smel-ft apply $patch
    if ($LASTEXITCODE -ne 0) { throw "SMEL FT patch failed" }
} else {
    git -C tools/smel-ft apply --reverse --check $patch 2>$null
    if ($LASTEXITCODE -ne 0) { throw "SMEL FT working tree differs from the reviewed patch" }
}
Write-Output "SMEL FT $revision + reviewed loss validation patch ready"
