$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "..\python\inspect_dataset.py")
if ($LASTEXITCODE -ne 0) { throw "Dataset inspection returned an error." }
