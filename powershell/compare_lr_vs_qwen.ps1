$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "..\python\compare_lr_vs_qwen.py")
if ($LASTEXITCODE -ne 0) { throw "LR/Qwen comparison failed." }
