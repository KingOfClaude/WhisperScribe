# Downloads the Parakeet model + Silero VAD into .\models (one-time, ~0.7 GB)
#
#   powershell -ExecutionPolicy Bypass -File .\download-parakeet.ps1
#   powershell -ExecutionPolicy Bypass -File .\download-parakeet.ps1 -Model v2    (English-only model)
#
# v3 = English + 24 European languages (auto-detects the language)   [default]
# v2 = English only

param([ValidateSet("v3", "v2")][string]$Model = "v3")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Fail($msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { Fail "curl.exe not found (needs Windows 10 or newer)." }
if (-not (Get-Command tar.exe  -ErrorAction SilentlyContinue)) { Fail "tar.exe not found (needs Windows 10 or newer)." }

$base    = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
$name    = "sherpa-onnx-nemo-parakeet-tdt-0.6b-$Model-int8"
$models  = Join-Path $PSScriptRoot "models"
$modelDir = Join-Path $models $name
$vadFile  = Join-Path $models "silero_vad.onnx"

New-Item -ItemType Directory -Force -Path $models | Out-Null

# --- Silero VAD (small)
if (Test-Path $vadFile) {
    Write-Host "VAD model already present." -ForegroundColor DarkGray
} else {
    Write-Host "Downloading Silero VAD..." -ForegroundColor Cyan
    & curl.exe -L --fail -o $vadFile "$base/silero_vad.onnx"
    if ($LASTEXITCODE -ne 0) { Remove-Item $vadFile -ErrorAction SilentlyContinue; Fail "VAD download failed." }
}

# --- Parakeet model
if (Test-Path (Join-Path $modelDir "tokens.txt")) {
    Write-Host "Parakeet $Model already present." -ForegroundColor DarkGray
} else {
    $archive = Join-Path $models "$name.tar.bz2"
    Write-Host "Downloading Parakeet $Model (int8, several hundred MB)..." -ForegroundColor Cyan
    & curl.exe -L --fail -o $archive "$base/$name.tar.bz2"
    if ($LASTEXITCODE -ne 0) {
        Remove-Item $archive -ErrorAction SilentlyContinue
        Fail "Model download failed. Check your connection, or that '$name.tar.bz2' exists on the sherpa-onnx 'asr-models' release page."
    }
    Write-Host "Extracting..." -ForegroundColor Cyan
    & tar.exe -xjf $archive -C $models
    if ($LASTEXITCODE -ne 0) { Fail "Extraction failed." }
    Remove-Item $archive
}

if (-not (Test-Path (Join-Path $modelDir "tokens.txt"))) { Fail "Model folder looks incomplete: $modelDir" }

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
Write-Host "  Model: $modelDir"
Write-Host "  VAD:   $vadFile"
Write-Host "WhisperScribe will find these automatically if it runs from a build folder inside this project."
