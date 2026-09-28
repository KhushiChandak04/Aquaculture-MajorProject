from pathlib import Path
from datetime import datetime
import json

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
TEMPORAL_OUT_PATH = BASE_DIR / "results" / "temporal_walk_forward.csv"
LEAKAGE_OUT_PATH = BASE_DIR / "results" / "leakage_audit.json"
WALK_FORWARD_START_YEAR = 2011


def split_features(df: pd.DataFrame):
    X = feature_frame(df)

    if "production" not in df.columns:
        raise ValueError("Column 'production' required for audit target generation")

    y = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)
    return X, y


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "Production_Category",
        "disease_risk_target",
        "disease_risk_score",
    ]
    keep_cols = [c for c in df.columns if c not in drop_cols]
    return df[keep_cols].copy()


def build_pipeline(X: pd.DataFrame):
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    safe_numeric_cols = []
    moved_to_cat = []
    for c in numeric_cols:
        coerced = pd.to_numeric(X[c], errors="coerce")
        if X[c].notna().any() and coerced.isna().any():
            moved_to_cat.append(c)
            cat_cols.append(c)
        else:
            safe_numeric_cols.append(c)

    num_cols = safe_numeric_cols
    if moved_to_cat:
        print(f"Moved non-numeric columns to categorical encoder: {moved_to_cat}")

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


def temporal_target(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Create production classes using thresholds learned from the training fold only."""
    train_production = pd.to_numeric(train_df["production"], errors="coerce")
    eval_production = pd.to_numeric(eval_df["production"], errors="coerce")
    q1, q2 = train_production.quantile([1 / 3, 2 / 3]).tolist()
    bins = [-np.inf, q1, q2, np.inf]
    labels = ["Low", "Medium", "High"]
    train_y = pd.cut(train_production, bins=bins, labels=labels, include_lowest=True).astype(str)
    eval_y = pd.cut(eval_production, bins=bins, labels=labels, include_lowest=True).astype(str)
    return train_y, eval_y


def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each year using only records from earlier years."""
    if "year" not in df.columns or "production" not in df.columns:
        return pd.DataFrame()

    rows = []
    years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique())
    for year in years:
        if year < WALK_FORWARD_START_YEAR:
            continue
        train_df = df[df["year"] < year].copy()
        test_df = df[df["year"] == year].copy()
        if train_df.empty or test_df.empty or train_df["production"].nunique() < 3:
            continue

        y_train, y_test = temporal_target(train_df, test_df)
        if y_train.nunique() < 3:
            continue

        x_train = feature_frame(train_df)
        x_test = feature_frame(test_df)
        pipe = build_pipeline(x_train)
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        scores = classification_report_values(y_test, pred)
        rows.append(
            {
                "test_year": year,
                "train_start_year": int(train_df["year"].min()),
                "train_end_year": int(train_df["year"].max()),
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                **scores,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    dup_ratio = float(df.duplicated().mean())
    duplicate_country_year = 0
    if {"country", "year"}.issubset(df.columns):
        duplicate_country_year = int(df.duplicated(subset=["country", "year"]).sum())
    constant_columns = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    leakage_audit = {
        "official_dataset_path": str(DATA_PATH.relative_to(BASE_DIR)),
        "full_dataset_rows": int(len(df)),
        "duplicate_row_count": int(df.duplicated().sum()),
        "duplicate_country_year_count": duplicate_country_year,
        "constant_columns": constant_columns,
        "train_only_transform_policy": "Pipeline preprocessing is fitted after each train split.",
        "target_threshold_policy": "Walk-forward thresholds are fitted from each historical training fold.",
        "synthetic_data_created": False,
    }

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

    walk_forward = run_walk_forward(df)
    if not walk_forward.empty:
        walk_forward.to_csv(TEMPORAL_OUT_PATH, index=False)
        lines.extend(
            [
                "",
                "## Walk-Forward Validation",
                "- Each test year is evaluated using only earlier years for training.",
                "- Scaling and categorical encoding are fitted within each training fold.",
                "- Production class thresholds are derived from the training fold and applied to that test year.",
                f"- Evaluated years: {int(walk_forward['test_year'].min())}-{int(walk_forward['test_year'].max())}.",
                f"- Mean walk-forward accuracy: {walk_forward['accuracy'].mean():.4f}.",
                f"- Mean walk-forward F1 macro: {walk_forward['f1_macro'].mean():.4f}.",
                f"- Detailed results: `{TEMPORAL_OUT_PATH.relative_to(BASE_DIR)}`.",
            ]
        )
    else:
        lines.extend(["", "## Walk-Forward Validation", "- Not available for the current dataset."])

    lines.extend(
        [
            "",
            "## Data Integrity Findings",
            f"- Duplicate country-year rows: {duplicate_country_year}.",
            f"- Constant final-dataset columns: {', '.join(constant_columns) if constant_columns else 'None'}.",
            "- No synthetic records are created by this audit.",
            "- Water geography and genomic sample assignments are not reconstructed without official source keys.",
        ]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LEAKAGE_OUT_PATH.write_text(json.dumps(leakage_audit, indent=2), encoding="utf-8")
    print(f"Saved validation audit to: {OUT_PATH}")


if __name__ == "__main__":
    main()
