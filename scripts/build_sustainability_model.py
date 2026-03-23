from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def build_target(df: pd.DataFrame) -> pd.Series:
    """Create a 3-class sustainability target from production quantiles."""
    return pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"])


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    data_path = base / "data" / "processed" / "final_dataset.csv"
    model_path = base / "models" / "sustainability_model.pkl"
    metrics_path = base / "results" / "sustainability_metrics.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")

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

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=400,
        random_state=42,
        learning_rate_init=0.001,
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
                "model": "MLP_rebuild",
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
