"""Compare the standalone Logistic Regression prediction with Qwen's final output."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()
DATA_FILE = ROOT / "examples" / "uci_heart" / "cases.jsonl"
DIAG_FILE = ROOT / "runs" / "uci_heart_tool_assisted" / "diagnoses.jsonl"
TOOL_PREFIX = "heart_disease_predictor@"


def metrics(name: str, actual: dict[str, str], predictions: dict[str, str]) -> None:
    tp = sum(actual[k] == "heart_disease" and predictions.get(k) == "heart_disease" for k in actual)
    fn = sum(actual[k] == "heart_disease" and predictions.get(k) == "no_heart_disease" for k in actual)
    tn = sum(actual[k] == "no_heart_disease" and predictions.get(k) == "no_heart_disease" for k in actual)
    fp = sum(actual[k] == "no_heart_disease" and predictions.get(k) == "heart_disease" for k in actual)
    n = len(actual)
    accuracy = (tp + tn) / n if n else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    print(f"\n{name}")
    print(f"TP={tp}, FN={fn}, TN={tn}, FP={fp}")
    print(f"Accuracy={accuracy:.4f}, Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}, F1={f1:.4f}")


def main() -> int:
    for path in (DATA_FILE, DIAG_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    cases = [json.loads(x) for x in DATA_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    actual = {r["id"]: r["label"] for r in cases if r.get("split") == "test"}
    rows = [json.loads(x) for x in DIAG_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("case") in actual]

    lr: dict[str, str] = {}
    qwen: dict[str, str] = {}
    qwen_conf: dict[str, float | None] = {}
    lr_conf: dict[str, float | None] = {}
    for row in rows:
        case_id = row["case"]
        diagnosis = row.get("d") or {}
        qwen[case_id] = diagnosis.get("prediction")
        qwen_conf[case_id] = diagnosis.get("confidence")
        for call in diagnosis.get("tool_calls", []):
            if call.get("tool", "").startswith(TOOL_PREFIX):
                output = call.get("output") or {}
                lr[case_id] = output.get("prediction")
                lr_conf[case_id] = output.get("confidence")

    print("========== LOGISTIC REGRESSION VS QWEN ==========")
    print(f"Test cases: {len(actual)}")
    print(f"LR predictions: {len(lr)}")
    print(f"Qwen predictions: {len(qwen)}")
    print(f"LR and Qwen agree: {sum(lr.get(k) == qwen.get(k) for k in actual)}")
    confidence_equal = sum(lr_conf.get(k) == qwen_conf.get(k) for k in actual)
    print(f"Confidence identical: {confidence_equal}/{len(actual)}")
    metrics("Logistic Regression alone", actual, lr)
    metrics("Qwen with Logistic Regression tool", actual, qwen)

    print("\n========== PATIENT-LEVEL DIFFERENCES ==========")
    differences = 0
    for case_id in actual:
        if lr.get(case_id) != qwen.get(case_id):
            differences += 1
            print(f"{case_id} | Actual: {actual[case_id]} | LR: {lr.get(case_id)} | Qwen: {qwen.get(case_id)}")
    if differences == 0:
        print("None: Qwen's final labels match the tool labels for all Test cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
