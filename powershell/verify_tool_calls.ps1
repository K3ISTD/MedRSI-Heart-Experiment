$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "..\python\verify_tool_calls.py")
if ($LASTEXITCODE -ne 0) { throw "Tool-call verification did not pass." }
