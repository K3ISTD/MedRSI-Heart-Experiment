#!/usr/bin/env python3
"""Prepare a small, reproducible MedRSI task from the UCI Cleveland Heart Disease data.

Downloads the real UCI dataset via ucimlrepo, removes rows with missing values,
converts target num=0 to no_heart_disease and num=1..4 to heart_disease,
then creates a stratified subset (120 cases by default) and explicit MedRSI splits.

The generated clinical-cost weights are illustrative research weights only, not
validated medical severity scores and not suitable for clinical decision-making.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

DESCRIPTIONS = {
    "age": "Age in years.",
    "sex": "Sex code as recorded in the UCI dataset (categorical code).",
    "cp": "Chest-pain type code as recorded in the UCI dataset (categorical code).",
    "trestbps": "Resting blood pressure in mm Hg.",
    "chol": "Serum cholesterol in mg/dl.",
    "fbs": "Fasting blood sugar indicator as recorded in the UCI dataset (categorical code).",
    "restecg": "Resting electrocardiographic result code.",
    "thalach": "Maximum heart rate achieved.",
    "exang": "Exercise-induced angina indicator (categorical code).",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of the peak exercise ST segment (categorical code).",
    "ca": "Number of major vessels colored by fluoroscopy (0–3; may be missing in source).",
    "thal": "Thallium stress-test result code (categorical code; may be missing in source).",
}


def stratified_subset(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if limit < 40:
        raise ValueError("For four MedRSI splits, --limit should be at least 40; recommend 120.")

    # Allocate the requested sample proportionally across the two labels.
    counts = (df["label"].value_counts(normalize=True) * limit).round().astype(int)
    # Correct rounding so the total equals limit.
    while int(counts.sum()) < limit:
        fractions = df["label"].value_counts(normalize=True) * limit - counts
        counts.loc[fractions.idxmax()] += 1
    while int(counts.sum()) > limit:
        fractions = df["label"].value_counts(normalize=True) * limit - counts
        counts.loc[fractions.idxmin()] -= 1

    chunks = []
    for label, n in counts.items():
        group = df[df["label"] == label]
        if n > len(group):
            raise ValueError(f"Requested {n} rows from class {label}, but only {len(group)} exist.")
        chunks.append(group.sample(n=int(n), random_state=seed + int(label == "heart_disease")))
    return pd.concat(chunks).sample(frac=1, random_state=seed).reset_index(drop=True)


def assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Stratified 60/15/15/10 split, separately within each class."""
    rng = __import__("numpy").random.default_rng(seed)
    out = df.copy()
    out["split"] = ""
    for label in ["no_heart_disease", "heart_disease"]:
        idx = out.index[out["label"] == label].to_numpy(dtype="int64", copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * 0.60))
        n_discovery = int(round(n * 0.15))
        n_trial = int(round(n * 0.15))
        # Remaining records form the test set.
        parts = [
            (idx[:n_train], "train"),
            (idx[n_train:n_train + n_discovery], "discovery"),
            (idx[n_train + n_discovery:n_train + n_discovery + n_trial], "trial"),
            (idx[n_train + n_discovery + n_trial:], "test"),
        ]
        for rows, split in parts:
            out.loc[rows, "split"] = split
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=120,
                        help="Number of real, complete cases to prepare; default 120. Use 0 for all complete cases.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible sampling/split seed.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                        help="Output directory (default: this script's directory).")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("Missing dependency. Run: python -m pip install ucimlrepo pandas", file=sys.stderr)
        return 2

    print("Downloading UCI Heart Disease dataset (dataset id=45)...")
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if y.shape[1] != 1:
        raise RuntimeError(f"Expected one target column, received: {list(y.columns)}")

    # Normalize names and validate that the expected Cleveland variables are present.
    X.columns = [str(c).strip().lower() for c in X.columns]
    target_name = str(y.columns[0]).strip().lower()
    y.columns = [target_name]
    missing_features = [c for c in FEATURES if c not in X.columns]
    if missing_features:
        raise RuntimeError(
            f"UCI API returned an unexpected feature schema. Missing: {missing_features}; "
            f"available: {list(X.columns)}"
        )

    frame = X[FEATURES].copy()
    frame["target_raw"] = pd.to_numeric(y.iloc[:, 0], errors="coerce").to_numpy()
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=FEATURES + ["target_raw"]).copy()
    frame["target_raw"] = frame["target_raw"].astype(int)
    frame = frame[frame["target_raw"].between(0, 4)].copy()
    if frame.empty:
        raise RuntimeError("No complete rows remained after cleaning the UCI data.")

    frame["label"] = frame["target_raw"].map(
        lambda v: "no_heart_disease" if int(v) == 0 else "heart_disease"
    )
    frame["source_row"] = [int(i) for i in frame.index]

    # Preserve the cleaned full data file (including original 0..4 outcome) for auditability.
    clean_csv = args.out / "uci_heart_cleveland_clean.csv"
    frame[FEATURES + ["target_raw", "label", "source_row"]].to_csv(clean_csv, index=False)

    sampled = stratified_subset(frame, args.limit, args.seed)
    sampled = assign_splits(sampled, args.seed)

    task = {
        "name": "uci_cleveland_heart_disease",
        "description": (
            "Academic benchmark: classify the presence or absence of angiographic heart disease "
            "from 13 clinical attributes in the UCI Cleveland Heart Disease dataset. "
            "This is retrospective historical data and is not a clinical diagnostic system."
        ),
        "labels": ["no_heart_disease", "heart_disease"],
        "positive_label": "heart_disease",
        "inputs": {k: DESCRIPTIONS[k] for k in FEATURES},
        "consequence": {
            "heart_disease->no_heart_disease": 2,
            "no_heart_disease->heart_disease": 1,
        },
        "rubric": (
            "ACADEMIC PROXY ONLY: for this software experiment, a false negative is assigned "
            "weight 2 and a false positive weight 1 to illustrate asymmetric error costs. "
            "These weights are not clinically validated severity estimates and must not be used "
            "for patient care or medical decisions."
        ),
    }
    (args.out / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    cases_path = args.out / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as f:
        for n, (_, row) in enumerate(sampled.iterrows(), start=1):
            inputs = {}
            for col in FEATURES:
                value = float(row[col])
                inputs[col] = int(value) if value.is_integer() else value
            case = {
                "id": f"cleveland_{n:04d}",
                # A generated record key used only to keep each row isolated during splitting;
                # it is not a source patient identifier.
                "patient_id": f"cleveland_record_{int(row['source_row']):04d}",
                "inputs": inputs,
                "label": str(row["label"]),
                "split": str(row["split"]),
            }
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    counts = sampled["split"].value_counts().to_dict()
    labels = sampled["label"].value_counts().to_dict()
    info = {
        "source": "UCI Machine Learning Repository, Heart Disease, dataset id 45; Cleveland processed subset",
        "doi": "10.24432/C52P4X",
        "license": "CC BY 4.0",
        "source_rows_returned": int(before),
        "complete_rows_after_dropping_missing": int(len(frame)),
        "prepared_cases": int(len(sampled)),
        "requested_limit": int(args.limit),
        "seed": int(args.seed),
        "split_counts": {k: int(v) for k, v in counts.items()},
        "label_counts": {k: int(v) for k, v in labels.items()},
        "target_mapping": "original num=0 -> no_heart_disease; num=1,2,3,4 -> heart_disease",
        "missing_data_handling": "Rows with any missing feature or target were removed.",
        "warning": "Historical benchmark data; not for clinical diagnosis or patient care.",
    }
    (args.out / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPreparation complete.")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\nFiles written to: {args.out.resolve()}")
    print(" - task.json")
    print(" - cases.jsonl")
    print(" - uci_heart_cleveland_clean.csv")
    print(" - dataset_info.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
