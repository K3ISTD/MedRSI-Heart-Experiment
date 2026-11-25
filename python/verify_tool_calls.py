"""Verify that the Logistic Regression Seed Tool ran for all saved Test diagnoses."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()
DATA_FILE = ROOT / "examples" / "uci_heart" / "cases.jsonl"
DIAG_FILE = ROOT / "runs" / "uci_heart_tool_assisted" / "diagnoses.jsonl"
TOOL_PREFIX = "heart_disease_predictor@"


def main() -> int:
    for path in (DATA_FILE, DIAG_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    test_ids = {
        row["id"]
        for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("split") == "test"
    }
    rows = [
        json.loads(line)
        for line in DIAG_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("case") in test_ids]

    used_ids: set[str] = set()
    failed_ids: set[str] = set()
    for row in rows:
        diagnosis = row.get("d") or {}
        calls = [call for call in diagnosis.get("tool_calls", []) if call.get("tool", "").startswith(TOOL_PREFIX)]
        if calls:
            used_ids.add(row.get("case"))
        if any("error" in (call.get("output") or {}) for call in calls):
            failed_ids.add(row.get("case"))

    missing_ids = sorted(test_ids - used_ids)
    print("========== TOOL EXECUTION VERIFICATION ==========")
    print(f"Total Test cases: {len(test_ids)}")
    print(f"Total saved Test diagnoses: {len(rows)}")
    print(f"Predictor called: {len(used_ids)}")
    print(f"Predictor errors: {len(failed_ids)}")
    print(f"Missing predictor: {missing_ids}")
    ok = len(test_ids) == 12 and used_ids == test_ids and not failed_ids
    print(f"Status: {'SUCCESS' if ok else 'CHECK REQUIRED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
