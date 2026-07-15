# Build the distributable kovadapt folder (Windows).
#     powershell -ExecutionPolicy Bypass -File packaging/build.ps1
# Output: dist/kovadapt/kovadapt.exe (+ support files); zip the folder to ship.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

py -m pip install ".[gui]" pyinstaller --quiet
py -m PyInstaller packaging/kovadapt.spec --noconfirm --clean

$exe = "dist/kovadapt/kovadapt.exe"
if (Test-Path $exe) {
    $size = (Get-ChildItem dist/kovadapt -Recurse | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("OK: {0} ({1:N0} MB total)" -f $exe, $size)
} else {
    Write-Error "build failed: $exe not found"
}
