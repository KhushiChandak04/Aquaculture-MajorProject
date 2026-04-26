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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "productivity_model.pkl"
METRICS_PATH = BASE_DIR / "results" / "productivity_metrics.csv"
DEPLOYMENT_ENSEMBLE_CANDIDATES = {"XGBoost", "ExtraTrees", "RandomForest"}
MIN_DEPLOY_ACCURACY = 0.75
MIN_DEPLOY_F1 = 0.75


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
    # Robustly split columns by semantic dtype. Relying on object-only checks can
    # misclassify pandas category/string extension dtypes as numeric in CI.
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Defensive fallback: if any supposed numeric column contains non-numeric text,
    # move it to categorical encoding to avoid StandardScaler conversion failures.
    safe_num_cols = []
    moved_to_cat = []
    for c in num_cols:
        coerced = pd.to_numeric(X[c], errors="coerce")
        if X[c].notna().any() and coerced.isna().any():
            moved_to_cat.append(c)
            cat_cols.append(c)
        else:
            safe_num_cols.append(c)

    num_cols = safe_num_cols
    if moved_to_cat:
        print(f"Moved non-numeric columns to categorical encoder: {moved_to_cat}")

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


def pick_best_latency_aware(metrics_df: pd.DataFrame) -> pd.Series:
    if metrics_df.empty:
        raise ValueError("No metrics available to select best model")
    return metrics_df.sort_values(
        ["recall_macro", "f1_macro", "accuracy", "infer_ms_per_1000", "model_size_mb", "train_seconds"],
        ascending=[False, False, False, True, True, True],
    ).iloc[0]


def pick_deployment_model(metrics_df: pd.DataFrame) -> pd.Series:
    if metrics_df.empty:
        raise ValueError("No metrics available to select best model")

    required_cols = {"model", "accuracy", "f1_macro", "infer_ms_per_1000"}
    if required_cols.issubset(set(metrics_df.columns)):
        candidates = metrics_df[metrics_df["model"].astype(str).isin(DEPLOYMENT_ENSEMBLE_CANDIDATES)].copy()
        if len(candidates) > 0:
            deployable = candidates[
                (pd.to_numeric(candidates["accuracy"], errors="coerce") >= MIN_DEPLOY_ACCURACY)
                & (pd.to_numeric(candidates["f1_macro"], errors="coerce") >= MIN_DEPLOY_F1)
            ].copy()

            if len(deployable) > 0:
                return deployable.sort_values(
                    ["infer_ms_per_1000", "recall_macro", "f1_macro", "accuracy", "model_size_mb", "train_seconds"],
                    ascending=[True, False, False, False, True, True],
                ).iloc[0]

    return pick_best_latency_aware(metrics_df)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X, y = make_dataset(df)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    xtr, xte, ytr, yte = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=500, n_jobs=None)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=8, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
    ]

    if HAS_XGBOOST:
        candidates.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=220,
                    max_depth=5,
                    learning_rate=0.06,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=42,
                ),
            )
        )

    rows = []
    trained = {}

    for name, model in candidates:
        pipe, metrics = eval_model(name, model, xtr, xte, ytr, yte)
        rows.append(metrics)
        trained[name] = pipe

    if not trained:
        raise RuntimeError("No productivity model could be trained")

    metrics_df = pd.DataFrame(rows)
    best_row = pick_deployment_model(metrics_df)
    best_model_name = str(best_row["model"])
    best_pipe = trained[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    metrics_df.sort_values(["recall_macro", "f1_macro", "accuracy"], ascending=[False, False, False]).to_csv(METRICS_PATH, index=False)

    print(f"Saved productivity model -> {MODEL_PATH}")
    print(f"Saved productivity metrics -> {METRICS_PATH}")
    print(
        "Selected productivity model (latency-aware deployable ensemble competition): "
        f"{best_model_name}"
    )


if __name__ == "__main__":
    main()
