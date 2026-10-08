# Shared helpers for build-cuda.ps1 / build-vulkan.ps1 (dot-sourced, not run directly)

function Fail($msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

function Require-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Fail "'$name' was not found on PATH. $hint (then open a NEW PowerShell window)."
    }
}

# Finds the newest Visual Studio that has the C++ tools and returns the matching
# CMake generator name, or $null if it can't tell (CMake will then auto-pick).
function Get-VsGenerator {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Fail "Visual Studio C++ tools not found. Install them with: winget install Microsoft.VisualStudio.2022.BuildTools --override ""--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait"""
    }
    $ver = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationVersion | Select-Object -First 1
    if (-not $ver) {
        Fail "Visual Studio is installed but the 'Desktop development with C++' workload is missing. Add it in the Visual Studio Installer (Modify)."
    }
    switch ([int]($ver.Split('.')[0])) {
        16 { return "Visual Studio 16 2019" }
        17 { return "Visual Studio 17 2022" }
        18 { return "Visual Studio 18 2026" }
        default { return $null }
    }
}

function Get-CMakeVersion {
    $line = (cmake --version | Select-Object -First 1)
    return [version]([regex]::Match($line, '\d+\.\d+(\.\d+)?').Value)
}

function Invoke-Build([string]$BuildDir, [string[]]$ExtraArgs, [bool]$Clean) {
    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Host "Cleaning $BuildDir ..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $BuildDir
    }

    $gen = Get-VsGenerator
    $cfg = @("-B", $BuildDir, "-A", "x64")
    if ($gen) { $cfg += @("-G", $gen) }
    $cfg += $ExtraArgs

    Write-Host "== Configuring ($BuildDir) ==" -ForegroundColor Cyan
    Write-Host "cmake $($cfg -join ' ')"
    & cmake @cfg
    if ($LASTEXITCODE -ne 0) { Fail "CMake configure failed (see output above)." }

    Write-Host "== Building (Release) ==" -ForegroundColor Cyan
    & cmake --build $BuildDir --config Release --parallel
    if ($LASTEXITCODE -ne 0) { Fail "Build failed (see output above)." }

    $exe = Join-Path $BuildDir "Release\whisper_scribe.exe"
    Write-Host ""
    Write-Host "SUCCESS. Run it with:  .\$exe" -ForegroundColor Green
}
