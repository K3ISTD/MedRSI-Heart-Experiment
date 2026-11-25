UCI HEART DISEASE (CLEVELAND) — MEDRSI DATA PREPARATION KIT
===========================================================

Dataset source
--------------
UCI Machine Learning Repository, Heart Disease, dataset ID 45.
Official page: https://archive.ics.uci.edu/dataset/45/heart+disease
DOI: 10.24432/C52P4X
License: CC BY 4.0

The Cleveland subset contains 303 records and 13 clinical features. The original
outcome is coded 0..4. This kit maps 0 -> no_heart_disease and 1..4 ->
heart_disease. Rows with missing values are dropped before sampling.

IMPORTANT
---------
This is historical retrospective data for an academic software experiment only.
The task's asymmetric error weights (false negative=2, false positive=1) are
illustrative benchmark penalties, not medically validated risk/severity scores.
Do not use outputs for diagnosis, triage, or patient-care decisions.

A. Install the small preparation dependencies
----------------------------------------------
Open PowerShell in your MedRSI repository folder, with its .venv activated:

    python -m pip install ucimlrepo pandas

B. Put this folder inside your MedRSI repository
------------------------------------------------
Recommended location:

    C:\AI\MedRSI\examples\uci_heart\

C. Download and prepare a 120-case real-data subset
---------------------------------------------------
From the MedRSI repository root:

    python .\examples\uci_heart\prepare_uci_heart.py --limit 120 --seed 42

This downloads dataset ID 45 from UCI and creates:
    task.json
    cases.jsonl
    uci_heart_cleveland_clean.csv
    dataset_info.json

The 120 cases are stratified by class and assigned to fixed MedRSI splits:
    train       60%
    discovery   15%
    trial       15%
    test        10%

The generated patient_id values are synthetic row keys because this public
Cleveland file does not provide a stable, usable patient identifier. Each row
is kept in exactly one split. The original data contain 303 rows; after dropping
missing values, the clean count is printed by the script.

D. To prepare all complete Cleveland cases instead
---------------------------------------------------
    python .\examples\uci_heart\prepare_uci_heart.py --limit 0 --seed 42

E. Run a one-round MedRSI test with Ollama/Qwen
-----------------------------------------------
Make sure Ollama is running and qwen3.5:4b is available. From the MedRSI root:

    medrsi run --task .\examples\uci_heart\task.json --data .\examples\uci_heart\cases.jsonl --llm "openai:qwen3.5:4b@http://localhost:11434/v1" --builder-llm "openai:qwen3.5:4b@http://localhost:11434/v1" --rounds 1 --workdir .\runs\uci_heart_qwen

Then inspect:

    medrsi status --workdir .\runs\uci_heart_qwen
    medrsi eval --workdir .\runs\uci_heart_qwen --split test

Start with one round. Do not start a 20-round run until you have confirmed
that diagnosis, tool generation, candidate checking, and evaluation work.

F. Dataset citation
-------------------
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
Heart Disease [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X

Please cite UCI and retain the CC BY 4.0 attribution when redistributing
or adapting the data.


Troubleshooting note (updated kit): the split routine creates a writable copy of
index arrays before shuffling, avoiding NumPy "array is read-only" errors.
