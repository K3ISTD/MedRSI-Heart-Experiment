$ErrorActionPreference = "Stop"
$experiments = @(
    @{ Label = "BASELINE"; Workdir = ".\runs\uci_heart_baseline" },
    @{ Label = "FULL MEDRSI"; Workdir = ".\runs\uci_heart_full_1" },
    @{ Label = "TOOL-ASSISTED"; Workdir = ".\runs\uci_heart_tool_assisted" }
)
foreach ($experiment in $experiments) {
    Write-Host "`n========== $($experiment.Label) / TEST ==========" -ForegroundColor Cyan
    medrsi eval --workdir $experiment.Workdir --split test
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($experiment.Label)." }
}
