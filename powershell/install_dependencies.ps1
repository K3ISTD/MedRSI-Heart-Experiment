$ErrorActionPreference = "Stop"
$requirements = Join-Path $PSScriptRoot "..\requirements-tool-assisted.txt"
Write-Host "Installing pandas, scikit-learn and joblib into the currently active Python environment..." -ForegroundColor Cyan
python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
Write-Host "Dependencies installed." -ForegroundColor Green
