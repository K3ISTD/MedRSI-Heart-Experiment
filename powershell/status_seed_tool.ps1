$ErrorActionPreference = "Stop"
medrsi status --workdir ".\runs\uci_heart_tool_assisted"
if ($LASTEXITCODE -ne 0) { throw "medrsi status failed." }
