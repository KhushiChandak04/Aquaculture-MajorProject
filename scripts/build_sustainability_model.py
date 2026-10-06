from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def build_target(df: pd.DataFrame) -> pd.Series:
    """Create a 3-class sustainability target from production quantiles."""
    return pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"])


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Robust split: pandas string/category extension dtypes may appear in CI.
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Defensive fallback for mixed-type columns that look numeric but contain text.
    safe_numeric_cols = []
    moved_to_cat = []
    for c in numeric_cols:
        coerced = pd.to_numeric(X[c], errors="coerce")
        if X[c].notna().any() and coerced.isna().any():
            moved_to_cat.append(c)
            categorical_cols.append(c)
        else:
            safe_numeric_cols.append(c)

    if moved_to_cat:
        print(f"Moved non-numeric columns to categorical encoder: {moved_to_cat}")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), safe_numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    data_path = base / "data" / "processed" / "final_dataset.csv"
    model_path = base / "models" / "sustainability_model.pkl"
    metrics_path = base / "results" / "sustainability_metrics.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")

    if metrics_path.exists():
        existing_metrics = pd.read_csv(metrics_path)
        if "model" in existing_metrics.columns and existing_metrics["model"].astype(str).eq("MLP").any():
            current_columns = set(pd.read_csv(data_path, nrows=0).columns)
            bundle_is_compatible = False
            if model_path.exists():
                existing_bundle = joblib.load(model_path)
                stored_features = set(existing_bundle.get("feature_columns", [])) if isinstance(existing_bundle, dict) else set()
                bundle_is_compatible = stored_features.issubset(current_columns)
            if bundle_is_compatible:
                print(f"Preserving notebook-derived sustainability metrics at: {metrics_path}")
                print(f"Preserving compatible sustainability model bundle at: {model_path}")
                return
            print("Sustainability artifact schema is stale; rebuilding against current final_dataset.csv")

    df = pd.read_csv(data_path)
    target = build_target(df)

    feature_cols = [
        "country",
        "year",
        "temperature_celsius",
        "precip_mm",
        "humidity",
        "water_Salinity (ppt)",
        "water_pH",
        "water_SecchiDepth (m)",
        "water_WaterDepth (m)",
        "water_WaterTemp (C)",
        "water_AirTemp (C)",
        "genomic_GC_Content_global_mean",
        "genomic_AT_Content_global_mean",
        "genomic_Mutation_Flag_global_mean",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = target.astype(str)
    preprocessor = make_preprocessor(X)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )

    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics_df = pd.DataFrame(
        [
            {
                "model": "XGBoost_rebuild",
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            }
        ]
    )

    model_bundle = {
        "model": pipeline,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "feature_columns": feature_cols,
        "target_name": "sustainability_class",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved sustainability model bundle to: {model_path}")
    print(f"Saved sustainability metrics to: {metrics_path}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Features: {feature_cols}")


if __name__ == "__main__":
    main()
