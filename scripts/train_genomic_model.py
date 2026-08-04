from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "feature_selector.pkl"
METRICS_PATH = BASE_DIR / "results" / "janhavi_model_metrics.csv"


def build_target(df: pd.DataFrame):
    if "production" not in df.columns:
        raise ValueError("Column 'production' required for genomic target generation")
    return pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["production", "Production_Category"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    if X.empty:
        raise ValueError("No genomic features available after dropping target columns")
    return X


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    encoders: dict[str, LabelEncoder] = {}

    categorical_cols = train_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for col in categorical_cols:
        encoder = LabelEncoder()
        train_encoded[col] = encoder.fit_transform(train_df[col].astype(str))
        test_encoded[col] = encoder.transform(test_df[col].astype(str))
        encoders[col] = encoder

    return train_encoded, test_encoded, encoders


def scale_frame(train_df: pd.DataFrame, test_df: pd.DataFrame):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    train_scaled = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
    test_scaled = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)
    return train_scaled, test_scaled, scaler


def evaluate_model(name, model, xtr, xte, ytr, yte):
    t0 = time.perf_counter()
    model.fit(xtr, ytr)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(xte)
    infer_s = time.perf_counter() - t1

    return {
        "Model": name,
        "Training_Time_s": float(train_s),
        "Inference_Time_s": float(infer_s),
        "Accuracy": float(accuracy_score(yte, pred)),
        "Precision_macro": float(precision_score(yte, pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(yte, pred, average="macro", zero_division=0)),
        "F1_Score_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "_estimator": model,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    y = build_target(df)
    X = make_feature_frame(df)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_train_encoded, x_test_encoded, _ = encode_categoricals(x_train, x_test)

    from sklearn.feature_selection import mutual_info_classif

    mi_scores = mutual_info_classif(x_train_encoded, y_train, random_state=42)
    mi_df = (
        pd.DataFrame({"Feature": x_train_encoded.columns, "MI_Score": mi_scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )

    top_5_features = mi_df.head(5)["Feature"].tolist()

    x_train_reduced = x_train[top_5_features].copy()
    x_test_reduced = x_test[top_5_features].copy()

    x_train_reduced, x_test_reduced, _ = encode_categoricals(x_train_reduced, x_test_reduced)
    x_train_reduced_scaled, x_test_reduced_scaled, _ = scale_frame(x_train_reduced, x_test_reduced)

    models = {
        "NaiveBayes": GaussianNB(),
        "KNN_k3": KNeighborsClassifier(n_neighbors=3),
        "KNN_k5": KNeighborsClassifier(n_neighbors=5),
        "KNN_k7": KNeighborsClassifier(n_neighbors=7),
        "SVM_linear": SVC(kernel="linear"),
        "SVM_rbf": SVC(kernel="rbf"),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "HistGB": HistGradientBoostingClassifier(random_state=42),
    }

    results = []
    trained_models = {}
    for name, model in models.items():
        row = evaluate_model(name, model, x_train_reduced_scaled, x_test_reduced_scaled, y_train, y_test)
        trained_models[name] = row.pop("_estimator")
        results.append(row)

    results_df = pd.DataFrame(results)

    best_accuracy_idx = results_df["Accuracy"].idxmax()
    best_model_name = str(results_df.loc[best_accuracy_idx, "Model"])
    best_model = trained_models[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    results_df.to_csv(METRICS_PATH, index=False)

    print("Model Training & Evaluation Results:")
    print(results_df.to_string(index=False))
    print(f"\nResults DataFrame shape: {results_df.shape}")
    print(f"Columns: {results_df.columns.tolist()}")
    print(f"Results saved to: {METRICS_PATH}")
    print(f"File size: {METRICS_PATH.stat().st_size} bytes")
    print("\nSummary Statistics:")
    print(f"Best Model (Accuracy): {best_model_name} ({results_df.loc[best_accuracy_idx, 'Accuracy']:.4f})")
    fastest_training = results_df.loc[results_df['Training_Time_s'].idxmin()]
    fastest_inference = results_df.loc[results_df['Inference_Time_s'].idxmin()]
    print(f"Fastest Training: {fastest_training['Model']} ({fastest_training['Training_Time_s']:.4f}s)")
    print(f"Fastest Inference: {fastest_inference['Model']} ({fastest_inference['Inference_Time_s']:.4f}s)")


if __name__ == "__main__":
    main()
