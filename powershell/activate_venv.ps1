# Dot-source this script from the MedRSI project root:
#   . .\powershell\activate_venv.ps1
$venvActivate = Join-Path (Get-Location) ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    throw "Cannot find $venvActivate. Run this from C:\AI\MedRSI and create the venv first if needed."
}
. $venvActivate
Write-Host "Virtual environment activated in this PowerShell session." -ForegroundColor Green
