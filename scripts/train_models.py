from pathlib import Path
import sys
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from preprocessing.run_all import run_all as run_preprocessing

    from scripts.build_sustainability_model import main as build_sustainability_model
    from scripts.generate_final_results_summary import main as generate_summary
    from scripts.generate_paper_results import main as generate_paper_results
    from scripts.generate_psg_explainability import main as generate_psg_explainability
    from scripts.generate_visual_reports import main as generate_visual_reports
    from scripts.recompute_genomic_importance import main as recompute_genomic_importance
    from scripts.train_genomic_model import main as train_genomic_model
    from scripts.train_productivity_model import main as train_productivity_model
    from scripts.validation_audit import main as run_validation_audit
except ImportError:
    from preprocessing.run_all import run_all as run_preprocessing

    from build_sustainability_model import main as build_sustainability_model
    from generate_final_results_summary import main as generate_summary
    from generate_paper_results import main as generate_paper_results
    from generate_psg_explainability import main as generate_psg_explainability
    from generate_visual_reports import main as generate_visual_reports
    from recompute_genomic_importance import main as recompute_genomic_importance
    from train_genomic_model import main as train_genomic_model
    from train_productivity_model import main as train_productivity_model
    from validation_audit import main as run_validation_audit

RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
PRODUCTIVITY_MODEL_PATH = BASE_DIR / "models" / "productivity_model.pkl"
PRODUCTIVITY_METRICS_PATH = RESULTS_DIR / "productivity_metrics.csv"


def make_augmented_productivity_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "production" not in df.columns:
        raise ValueError("Column 'production' required for productivity target generation")

    # log1p preserves ordering while reducing production's extreme right skew.
    log_target = np.log1p(pd.to_numeric(df["production"], errors="coerce").clip(lower=0))
    y = pd.qcut(log_target, q=3, labels=["Low", "Medium", "High"]).astype(str)

    X = df.drop(
        columns=[
            c
            for c in [
                "production",
                "Production_Category",
                "disease_risk_target",
                "disease_risk_score",
                "genomic_Disease_Risk_global_mode",
            ]
            if c in df.columns
        ],
        errors="ignore",
    ).copy()
    if "year" in X.columns:
        X["year"] = pd.to_numeric(X["year"], errors="coerce")
        X["decade"] = (X["year"] // 10) * 10
    return X, y


def augmented_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def run_augmented_productivity_benchmark() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = make_augmented_productivity_data(df)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("DecisionTree", DecisionTreeClassifier(max_depth=8, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
    ]
    if XGBClassifier is not None:
        models.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=42,
                ),
            )
        )
    if LGBMClassifier is not None:
        models.append(
            (
                "LightGBM",
                LGBMClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    verbosity=-1,
                ),
            )
        )

    rows = []
    trained = {}
    for name, estimator in models:
        pipe = Pipeline([("prep", augmented_preprocessor(x_train)), ("model", estimator)])
        process = psutil.Process()
        before = process.memory_info().rss / (1024 * 1024)
        started = time.perf_counter()
        pipe.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started
        after = process.memory_info().rss / (1024 * 1024)
        train_pred = pipe.predict(x_train)
        started = time.perf_counter()
        test_pred = pipe.predict(x_test)
        infer_ms_per_1000 = (time.perf_counter() - started) * 1000 * (1000 / len(x_test))

        fd, tmp_path = tempfile.mkstemp(suffix=".joblib")
        os.close(fd)
        try:
            joblib.dump(pipe, tmp_path)
            model_size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        rows.append(
            {
                "model": name,
                "train_accuracy": float(accuracy_score(y_train, train_pred)),
                "test_accuracy": float(accuracy_score(y_test, test_pred)),
                "accuracy": float(accuracy_score(y_test, test_pred)),
                "precision_macro": float(precision_score(y_test, test_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
                "train_seconds": float(train_seconds),
                "infer_ms_per_1000": float(infer_ms_per_1000),
                "model_size_mb": float(model_size_mb),
                "peak_train_ram_mb": float(max(0.0, after - before)),
            }
        )
        trained[name] = pipe

    metrics_df = pd.DataFrame(rows).sort_values(
        ["recall_macro", "f1_macro", "test_accuracy", "infer_ms_per_1000"],
        ascending=[False, False, False, True],
    )
    best_name = str(metrics_df.iloc[0]["model"])
    PRODUCTIVITY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained[best_name], PRODUCTIVITY_MODEL_PATH)
    metrics_df.to_csv(PRODUCTIVITY_METRICS_PATH, index=False)
    print("Augmented productivity benchmark (log1p target quantiles):")
    print(metrics_df[["model", "train_accuracy", "test_accuracy", "accuracy", "f1_macro"]].to_string(index=False))
    print(f"Selected productivity model: {best_name}")


def main() -> None:
    print("[1/8] Running preprocessing...")
    run_preprocessing()

    print("[2/8] Training augmented productivity models...")
    run_augmented_productivity_benchmark()

    print("[3/8] Training genomic model...")
    train_genomic_model()

    print("[4/8] Building sustainability model...")
    build_sustainability_model()

    print("[5/8] Recomputing genomic importance...")
    recompute_genomic_importance()

    print("[6/8] Running validation audit...")
    run_validation_audit()

    print("[7/8] Generating visual reports...")
    generate_visual_reports()

    print("[8/9] Generating paper-ready tables and PSG fusion metrics...")
    generate_paper_results()

    print("[9/10] Generating PSG weight and feature-SHAP charts...")
    generate_psg_explainability()

    print("[10/10] Regenerating final summary...")
    generate_summary()

    print("Training pipeline completed.")
    print(f"Artifacts available in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
