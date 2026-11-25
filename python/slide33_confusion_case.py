"""Print the Tool-assisted confusion matrix and cleveland_0075 case study."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()
DATA_FILE = ROOT / "examples" / "uci_heart" / "cases.jsonl"
BASELINE_FILE = ROOT / "runs" / "uci_heart_baseline" / "diagnoses.jsonl"
FULL_FILE = ROOT / "runs" / "uci_heart_full_1" / "diagnoses.jsonl"
ASSISTED_FILE = ROOT / "runs" / "uci_heart_tool_assisted" / "diagnoses.jsonl"
TARGET_CASE = "cleveland_0075"
TOOL_PREFIX = "heart_disease_predictor@"


def load(path: Path, valid_ids: set[str]) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        diagnosis = row.get("d", row)
        case_id = row.get("case") or diagnosis.get("case_id")
        if case_id in valid_ids:
            result[case_id] = diagnosis
    return result


def main() -> int:
    cases = [json.loads(x) for x in DATA_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    actual = {r["id"]: r["label"] for r in cases if r.get("split") == "test"}
    valid_ids = set(actual)
    baseline = load(BASELINE_FILE, valid_ids)
    full = load(FULL_FILE, valid_ids)
    assisted = load(ASSISTED_FILE, valid_ids)

    pred = {k: assisted.get(k, {}).get("prediction") for k in actual}
    tp = sum(actual[k] == "heart_disease" and pred[k] == "heart_disease" for k in actual)
    fn = sum(actual[k] == "heart_disease" and pred[k] == "no_heart_disease" for k in actual)
    tn = sum(actual[k] == "no_heart_disease" and pred[k] == "no_heart_disease" for k in actual)
    fp = sum(actual[k] == "no_heart_disease" and pred[k] == "heart_disease" for k in actual)

    print("=" * 58)
    print("TOOL-ASSISTED: CONFUSION MATRIX")
    print("=" * 58)
    print("                         PREDICTED")
    print("                    Disease    No disease")
    print(f"Actual Disease       TP={tp:<5}    FN={fn}")
    print(f"Actual No disease    FP={fp:<5}    TN={tn}")

    if TARGET_CASE not in actual or TARGET_CASE not in assisted:
        raise ValueError(f"Target case {TARGET_CASE} is not present in saved Test results.")
    diagnosis = assisted[TARGET_CASE]
    tool_call = next(
        (call for call in diagnosis.get("tool_calls", []) if call.get("tool", "").startswith(TOOL_PREFIX)),
        None,
    )
    if not tool_call:
        raise ValueError(f"No {TOOL_PREFIX} tool call found for {TARGET_CASE}.")
    tool_output = tool_call.get("output") or {}

    print("\n" + "=" * 58)
    print(f"CASE STUDY: {TARGET_CASE.upper()}")
    print("=" * 58)
    print("Actual label        :", actual[TARGET_CASE])
    print("Baseline prediction :", baseline.get(TARGET_CASE, {}).get("prediction"))
    print("Full MedRSI         :", full.get(TARGET_CASE, {}).get("prediction"))
    print("Logistic Regression :", tool_output.get("prediction"))
    print("LR estimated prob.  :", tool_output.get("confidence"))
    print("Qwen final          :", diagnosis.get("prediction"))
    print("Qwen confidence     :", diagnosis.get("confidence"))
    print("=" * 58)
    print("Interpretation: this case changed from incorrect in Baseline/Full to correct in Tool-assisted.")
    print("Note: Qwen's output matched the tool in the recorded run; this does not establish independent combined reasoning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
