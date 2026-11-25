import json
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "examples" / "uci_heart" / "cases.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parent / "artifacts" / "model.joblib"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
VALID_LABELS = {"no_heart_disease", "heart_disease"}

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    rows = [
        json.loads(line)
        for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    train_rows = [row for row in rows if row.get("split") == "train"]

    if len(train_rows) != 72:
        raise ValueError(f"Expected 72 training records, found {len(train_rows)}")

    labels = [row["label"] for row in train_rows]
    if set(labels) != VALID_LABELS:
        raise ValueError(f"Unexpected labels: {set(labels)}")

    X = pd.DataFrame(
        [{feature: row["inputs"][feature] for feature in FEATURES}
         for row in train_rows],
        columns=FEATURES,
    )
    y = labels

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    model.fit(X, y)

    artifact = {
        "model": model,
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "classes": list(model.classes_),
        "trained_on_split": "train",
        "train_n": len(train_rows),
        "train_label_counts": dict(Counter(y)),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT_FILE)

    print("Training completed.")
    print(f"Training records: {len(train_rows)}")
    print(f"Training labels: {dict(Counter(y))}")
    print(f"Features: {len(FEATURES)}")
    print(f"Artifact: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
