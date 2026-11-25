"""Print patient-level predictions for Baseline, Full MedRSI and Tool-assisted."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()
DATA_FILE = ROOT / "examples" / "uci_heart" / "cases.jsonl"
PATHS = {
    "Baseline": ROOT / "runs" / "uci_heart_baseline" / "diagnoses.jsonl",
    "Full": ROOT / "runs" / "uci_heart_full_1" / "diagnoses.jsonl",
    "Tool-assisted": ROOT / "runs" / "uci_heart_tool_assisted" / "diagnoses.jsonl",
}


def load_diagnoses(path: Path, test_ids: set[str]) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnoses file: {path}")
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        diagnosis = row.get("d", row)
        case_id = row.get("case") or diagnosis.get("case_id")
        if case_id in test_ids:
            result[case_id] = diagnosis
    return result


def main() -> int:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    cases = [json.loads(x) for x in DATA_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    actual = {r["id"]: r["label"] for r in cases if r.get("split") == "test"}
    test_ids = set(actual)
    predictions = {name: load_diagnoses(path, test_ids) for name, path in PATHS.items()}

    print("========== PATIENT-LEVEL TEST COMPARISON ==========")
    print(f"{'Patient':<17} {'Actual':<20} {'Baseline':<20} {'Full':<20} {'Tool-assisted':<20}")
    print("-" * 100)
    for case_id, label in actual.items():
        values = [predictions[name].get(case_id, {}).get("prediction", "MISSING") for name in PATHS]
        print(f"{case_id:<17} {label:<20} {values[0]:<20} {values[1]:<20} {values[2]:<20}")

    print("\n========== SUMMARY ==========")
    for name, result in predictions.items():
        correct = sum(result.get(case_id, {}).get("prediction") == label for case_id, label in actual.items())
        print(f"{name}: {len(result)}/{len(actual)} predictions available; {correct}/{len(actual)} correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
