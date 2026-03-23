from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
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

    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipeline.fit(X, y_enc)

    model_bundle = {
        "model": pipeline,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "feature_columns": feature_cols,
        "target_name": "sustainability_class",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)

    print(f"Saved sustainability model bundle to: {model_path}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Features: {feature_cols}")


if __name__ == "__main__":
    main()
