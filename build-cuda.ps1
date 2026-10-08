# NVIDIA GPU build (CUDA) -> build-cuda\Release\whisper_scribe.exe
#
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File .\build-cuda.ps1
#   powershell -ExecutionPolicy Bypass -File .\build-cuda.ps1 -Clean    (wipe and rebuild)
#   powershell -ExecutionPolicy Bypass -File .\build-cuda.ps1 -Arch 86  (force an architecture)

param(
    [switch]$Clean,
    [string]$Arch = ""      # e.g. 75, 86, 89, 120  (default: detected from your GPU)
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. "$PSScriptRoot\common.ps1"

Require-Command cmake "Install with: winget install Kitware.CMake"
Require-Command nvcc  "Install the CUDA Toolkit with: winget install Nvidia.CUDA (install Visual Studio C++ tools first)"

# ---- pick the CUDA architecture. Kernels built for the wrong one crash at runtime with
#      "CUDA error: no kernel image is available for execution on the device".
$archValue = $Arch.Trim()
if (-not $archValue) {
    $caps = @()
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $rows = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null
        foreach ($r in $rows) {
            $parts = ("$r" -split ",") | ForEach-Object { $_.Trim() }
            if ($parts.Count -ge 2 -and $parts[1] -match '^\d+\.\d+$') {
                Write-Host ("GPU detected: {0}  (compute capability {1})" -f $parts[0], $parts[1]) -ForegroundColor Cyan
                $caps += ($parts[1] -replace '\.', '')
            }
        }
    }
    $caps = @($caps | Select-Object -Unique)
    if ($caps.Count -gt 0) {
        $archValue = ($caps -join ";")
    } else {
        Write-Host "Could not read the GPU's compute capability from nvidia-smi - using 'native'." -ForegroundColor Yellow
        if ((Get-CMakeVersion) -ge [version]"3.24") {
            $archValue = "native"
        } else {
            Fail "Could not detect your GPU and CMake is older than 3.24. Re-run with -Arch <number>, e.g. -Arch 86 (RTX 30), 89 (RTX 40), 75 (RTX 20 / GTX 16), 120 (RTX 50). Find yours with: nvidia-smi --query-gpu=name,compute_cap --format=csv"
        }
    }
}

# CUDA 13 dropped support for architectures older than 7.5 (Maxwell/Pascal/Volta)
foreach ($a in ($archValue -split ";")) {
    if ($a -match '^\d+$' -and [int]$a -lt 75) {
        Fail "CUDA architecture $a (GTX 9xx/10xx or Titan V) is not supported by CUDA 13. Use build-vulkan.ps1 instead, or install CUDA 12.x."
    }
}
Write-Host "Building for CUDA architecture(s): $archValue" -ForegroundColor Green

$extra = @("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=$archValue")

Invoke-Build "build-cuda" $extra $Clean.IsPresent
Write-Host "Note: CUDA builds are slow the first time (10-20 min is normal)." -ForegroundColor DarkGray
