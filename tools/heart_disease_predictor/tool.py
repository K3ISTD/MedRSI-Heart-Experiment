from pathlib import Path

import joblib
import pandas as pd

NAME = "heart_disease_predictor"

DESCRIPTION = (
    "Predicts heart disease risk using a Logistic Regression model "
    "trained on the UCI Cleveland training split. Returns the predicted "
    "class, calibrated-by-model probability estimates, and confidence. "
    "For educational research use only."
)


def run(inputs, ctx):
    artifact_dir = getattr(ctx, "artifact_dir", None)

    if artifact_dir is None:
        raise RuntimeError("Tool context does not provide artifact_dir.")

    model_path = Path(artifact_dir) / "model.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]

    missing = [name for name in features if name not in inputs]
    if missing:
        raise ValueError(f"Missing patient features: {missing}")

    patient = {name: inputs[name] for name in features}
    X = pd.DataFrame([patient], columns=features)

    probabilities = model.predict_proba(X)[0]
    classes = model.classes_
    prediction = str(model.predict(X)[0])

    probability_map = {
        str(label): round(float(probability), 6)
        for label, probability in zip(classes, probabilities)
    }

    confidence = probability_map[prediction]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_map,
        "model": "LogisticRegression",
        "training_split": artifact["trained_on_split"],
        "training_records": artifact["train_n"],
    }
