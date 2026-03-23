from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "feature_selector.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
OUT_PATH = BASE_DIR / "results" / "genomic_feature_importance.csv"


def make_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = pd.factorize(out[col].astype(str))[0]
    return out.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def perturbation_importance(model, X: pd.DataFrame, repeats: int = 20, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed=random_state)
    baseline_pred = model.predict(X)

    means = []
    stds = []

    for col in X.columns:
        impacts = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[col] = Xp[col].to_numpy()[perm]
            pred_p = model.predict(Xp)
            impacts.append(float(np.mean(pred_p != baseline_pred)))
        means.append(float(np.mean(impacts)))
        stds.append(float(np.std(impacts)))

    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": means,
            "importance_std": stds,
            "method": "prediction_sensitivity",
        }
    ).sort_values("importance_mean", ascending=False)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    feature_names = list(getattr(model, "feature_names_in_", []))
    if feature_names:
        X = df[feature_names].copy()
    else:
        drop_cols = [c for c in ["Production_Category", "disease_risk_target", "disease_risk_score"] if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore")

    Xn = make_numeric_frame(X)
    importance_df = perturbation_importance(model, Xn, repeats=20)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(OUT_PATH, index=False)

    print(f"Saved genomic importance to: {OUT_PATH}")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
