from pathlib import Path
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "productivity_model.pkl"
METRICS_PATH = BASE_DIR / "results" / "productivity_metrics.csv"


def make_dataset(df: pd.DataFrame):
    if "production" not in df.columns:
        raise ValueError("Column 'production' required for productivity target generation")

    y = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)
    drop_cols = [
        "production",
        "Production_Category",
        "disease_risk_target",
        "disease_risk_score",
        "genomic_Disease_Risk_global_mode",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return X, y


def make_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def eval_model(name, model, xtr, xte, ytr, yte):
    pre = make_preprocessor(xtr)
    pipe = Pipeline(steps=[("prep", pre), ("model", model)])

    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()
    pipe.fit(xtr, ytr)
    train_time = time.perf_counter() - t0
    mem_after = proc.memory_info().rss / (1024 * 1024)
    peak_ram = max(0.0, mem_after - mem_before)

    t1 = time.perf_counter()
    pred = pipe.predict(xte)
    infer_time = (time.perf_counter() - t1) * 1000.0
    infer_ms_per_1000 = infer_time * (1000.0 / max(1, len(xte)))

    fd, tmp_path = tempfile.mkstemp(suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(pipe, tmp_path)
        model_size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(yte, pred)),
        "precision_macro": float(precision_score(yte, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(yte, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "train_seconds": float(train_time),
        "infer_ms_per_1000": float(infer_ms_per_1000),
        "model_size_mb": float(model_size_mb),
        "peak_train_ram_mb": float(peak_ram),
    }
    return pipe, metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X, y = make_dataset(df)

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=500, n_jobs=None)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=8, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
    ]

    rows = []
    best_pipe = None
    best_score = -np.inf

    for name, model in candidates:
        pipe, metrics = eval_model(name, model, xtr, xte, ytr, yte)
        rows.append(metrics)
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_pipe = pipe

    if best_pipe is None:
        raise RuntimeError("No productivity model could be trained")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    pd.DataFrame(rows).sort_values("f1_macro", ascending=False).to_csv(METRICS_PATH, index=False)

    print(f"Saved productivity model -> {MODEL_PATH}")
    print(f"Saved productivity metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
