# Vulkan GPU build (AMD / Intel / NVIDIA) -> build-vulkan\Release\whisper_scribe.exe
#
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File .\build-vulkan.ps1
#   powershell -ExecutionPolicy Bypass -File .\build-vulkan.ps1 -Clean    (wipe and rebuild)

param([switch]$Clean)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. "$PSScriptRoot\common.ps1"

Require-Command cmake "Install with: winget install Kitware.CMake"

if (-not $env:VULKAN_SDK) {
    Fail "Vulkan SDK not found (VULKAN_SDK is not set). Install with: winget install KhronosGroup.VulkanSDK (then open a NEW PowerShell window)."
}
if (-not (Test-Path (Join-Path $env:VULKAN_SDK "Bin\glslc.exe"))) {
    Fail "glslc.exe is missing from the Vulkan SDK at $env:VULKAN_SDK. Reinstall the SDK."
}

Invoke-Build "build-vulkan" @("-DGGML_VULKAN=ON") $Clean.IsPresent
