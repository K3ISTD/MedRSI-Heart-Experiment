$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "..\python\slide33_confusion_case.py")
if ($LASTEXITCODE -ne 0) { throw "Slide 33 evidence generation failed." }
