$ErrorActionPreference = "Stop"
python -m py_compile .\medrsi\agent.py
if ($LASTEXITCODE -ne 0) { throw "agent.py syntax check failed." }
Write-Host "agent.py syntax is valid." -ForegroundColor Green
