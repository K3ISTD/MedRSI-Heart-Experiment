"""Inspect the prepared UCI Cleveland dataset used by the MedRSI experiment."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
DATA_FILE = ROOT / "examples" / "uci_heart" / "cases.jsonl"
EXPECTED_SPLITS = {"train": 72, "discovery": 18, "trial": 18, "test": 12}


def main() -> int:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. Run this script from the MedRSI project root."
        )

    rows = [
        json.loads(line)
        for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_counts = Counter(row.get("split") for row in rows)
    train_rows = [row for row in rows if row.get("split") == "train"]
    if not train_rows:
        raise ValueError("No training records found.")

    labels = Counter(row.get("label") for row in train_rows)
    features = list(train_rows[0].get("inputs", {}).keys())
    missing_count = sum(
        value is None
        for row in rows
        for value in (row.get("inputs") or {}).values()
    )

    print("========== DATASET INSPECTION ==========")
    print(f"Dataset file: {DATA_FILE}")
    print(f"TOTAL RECORDS: {len(rows)}")
    print(f"DATA SPLITS: {dict(split_counts)}")
    print(f"TRAIN LABELS: {dict(labels)}")
    print(f"FEATURE COUNT: {len(features)}")
    print(f"FEATURES: {', '.join(features)}")
    print(f"MISSING VALUES: {missing_count}")

    if len(rows) != 120 or dict(split_counts) != EXPECTED_SPLITS:
        print("WARNING: Dataset size or split counts differ from the expected experiment.")
        return 1
    print("CHECK: Dataset counts match the recorded experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
