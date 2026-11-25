$ErrorActionPreference = "Stop"
$target = ".\medrsi\agent.py"
$backup = ".\medrsi\agent.py.backup"
if (-not (Test-Path $backup)) {
    throw "Backup not found: $backup. No file was changed."
}
Write-Warning "This will replace medrsi\agent.py with the pre-patch backup."
$answer = Read-Host "Type RESTORE to continue"
if ($answer -cne "RESTORE") { Write-Host "Cancelled; no file was changed."; exit 0 }
Copy-Item $backup $target -Force
python -m py_compile $target
if ($LASTEXITCODE -ne 0) { throw "Restored agent.py failed the syntax check. Restore the patched copy manually if needed." }
Write-Host "agent.py restored from agent.py.backup." -ForegroundColor Green
