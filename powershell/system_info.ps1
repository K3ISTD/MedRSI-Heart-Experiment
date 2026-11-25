$ErrorActionPreference = "Continue"
Write-Host "========== SYSTEM CONFIGURATION ==========" -ForegroundColor Cyan
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$ram = Get-CimInstance Win32_ComputerSystem
$gpu = Get-CimInstance Win32_VideoController | Sort-Object AdapterRAM -Descending | Select-Object -First 1
[PSCustomObject]@{
    CPU = $cpu.Name
    RAM_GB = [math]::Round($ram.TotalPhysicalMemory / 1GB, 1)
    GPU = $gpu.Name
    GPU_Memory_GB = [math]::Round($gpu.AdapterRAM / 1GB, 1)
} | Format-List
Write-Host "========== OLLAMA MODELS ==========" -ForegroundColor Cyan
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    ollama list
} else {
    Write-Warning "ollama command not found in PATH."
}
