from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
OUT_PATH = BASE_DIR / "results" / "validation_audit.md"


def split_features(df: pd.DataFrame):
    drop_cols = [
        "Production_Category",
        "disease_risk_target",
        "disease_risk_score",
    ]
    keep_cols = [c for c in df.columns if c not in drop_cols]
    X = df[keep_cols].copy()

    if "production" not in df.columns:
        raise ValueError("Column 'production' required for audit target generation")

    y = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)
    return X, y


def build_pipeline(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    return Pipeline(steps=[("prep", prep), ("model", model)])


def temporal_split(df: pd.DataFrame):
    if "year" not in df.columns:
        return None, None
    ordered = df.sort_values("year").reset_index(drop=True)
    cut = int(len(ordered) * 0.8)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def classification_report_values(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    dup_ratio = float(df.duplicated().mean())

    X, y = split_features(df)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipe_random = build_pipeline(X)
    pipe_random.fit(x_train, y_train)
    pred_random = pipe_random.predict(x_test)
    random_scores = classification_report_values(y_test, pred_random)

    temp_train_df, temp_test_df = temporal_split(df)
    temporal_scores = None
    if temp_train_df is not None and temp_test_df is not None and len(temp_test_df) > 0:
        xtr, ytr = split_features(temp_train_df)
        xte, yte = split_features(temp_test_df)
        pipe_temporal = build_pipeline(xtr)
        pipe_temporal.fit(xtr, ytr)
        pred_temporal = pipe_temporal.predict(xte)
        temporal_scores = classification_report_values(yte, pred_temporal)

    lines = []
    lines.append("# Validation and Leakage Audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Dataset Integrity")
    lines.append(f"- Samples: {len(df)}")
    lines.append(f"- Duplicate-row ratio: {dup_ratio:.4f}")
    lines.append("")
    lines.append("## Robustness Checks")
    lines.append("### Random Stratified Split (80/20)")
    lines.append(f"- Accuracy: {random_scores['accuracy']:.4f}")
    lines.append(f"- F1 macro: {random_scores['f1_macro']:.4f}")
    lines.append("")

    if temporal_scores is not None:
        lines.append("### Temporal Split (first 80% years train, last 20% years test)")
        lines.append(f"- Accuracy: {temporal_scores['accuracy']:.4f}")
        lines.append(f"- F1 macro: {temporal_scores['f1_macro']:.4f}")
        lines.append("")
        drift = random_scores["f1_macro"] - temporal_scores["f1_macro"]
        lines.append("## Drift Signal")
        lines.append(f"- F1 gap (random - temporal): {drift:.4f}")
    else:
        lines.append("### Temporal Split")
        lines.append("- Not available because 'year' column is missing.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved validation audit to: {OUT_PATH}")


if __name__ == "__main__":
    main()
