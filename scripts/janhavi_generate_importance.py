from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    model_path = base / "models" / "feature_selector.pkl"
    data_path = base / "data" / "processed" / "final_dataset.csv"
    out_path = base / "results" / "genomic_feature_importance.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df["Production_Category"] = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"])

    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise RuntimeError("Model has no feature_names_in_. Re-run notebook training cells to rebuild model artifact.")

    X = df[feature_names].copy()
    y = df["Production_Category"]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.factorize(X[col])[0]

    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        scoring="f1_macro",
    )

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
