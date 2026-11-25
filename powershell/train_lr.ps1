$ErrorActionPreference = "Stop"
Write-Warning "This script re-trains and overwrites tools\heart_disease_predictor\artifacts\model.joblib using only the 72 TRAIN rows. Do not run it unless you intend to reproduce training."
$answer = Read-Host "Type TRAIN to continue"
if ($answer -cne "TRAIN") { Write-Host "Cancelled; existing model artifact was not changed."; exit 0 }
python (Join-Path $PSScriptRoot "..\python\train_model.py")
if ($LASTEXITCODE -ne 0) { throw "Model training failed." }
