from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "feature_selector.pkl"
METRICS_PATH = BASE_DIR / "results" / "janhavi_model_metrics.csv"


def build_target(df: pd.DataFrame):
    if "production" not in df.columns:
        raise ValueError("Column 'production' required for genomic target generation")
    return pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)


def pick_features(df: pd.DataFrame):
    genomic_cols = [c for c in df.columns if c.startswith("genomic_")]
    context_cols = [c for c in ["country", "year", "temperature_celsius", "precip_mm", "humidity"] if c in df.columns]
    cols = genomic_cols + context_cols
    if not cols:
        cols = [c for c in df.columns if c not in ["production", "Production_Category", "disease_risk_target", "disease_risk_score"]]
    return cols


def make_numeric(df: pd.DataFrame):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = pd.factorize(out[col].astype(str))[0]
    return out.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    y = build_target(df)
    cols = pick_features(df)
    X = make_numeric(df[cols].copy())

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = HistGradientBoostingClassifier(random_state=42)

    t0 = time.perf_counter()
    model.fit(xtr, ytr)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(xte)
    infer_s = time.perf_counter() - t1

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metrics = pd.DataFrame(
        [
            {
                "Model": "HistGB",
                "Training_Time_s": float(train_s),
                "Inference_Time_s": float(infer_s),
                "Accuracy": float(accuracy_score(yte, pred)),
                "Precision_macro": float(precision_score(yte, pred, average="macro", zero_division=0)),
                "Recall_macro": float(recall_score(yte, pred, average="macro", zero_division=0)),
                "F1_Score_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
            }
        ]
    )
    metrics.to_csv(METRICS_PATH, index=False)

    print(f"Saved genomic model -> {MODEL_PATH}")
    print(f"Saved genomic metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
