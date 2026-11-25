$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "..\python\compare_test_predictions.py")
if ($LASTEXITCODE -ne 0) { throw "Patient-level comparison failed." }
