from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"

OUTPUT_MD = RESULTS_DIR / "paper_results_tables.md"
OUTPUT_PSG_CSV = RESULTS_DIR / "psg_combined_metrics.csv"
OUTPUT_PSG_SAMPLES = RESULTS_DIR / "psg_combined_probabilities.csv"
PSG_WEIGHT_PRODUCTIVITY = 0.50
PSG_WEIGHT_SUSTAINABILITY = 0.35
PSG_WEIGHT_GENOMIC = 0.15


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return pd.read_csv(path)


def fmt_num(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def markdown_table(headers: list[str], rows: list[list[str]], right_align: list[bool] | None = None) -> list[str]:
    if right_align is None:
        right_align = [False] * len(headers)

    lines = ["| " + " | ".join(headers) + " |"]
    align_parts = ["---:" if right_align[i] else "---" for i in range(len(headers))]
    lines.append("|" + "|".join(align_parts) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def rows_from_df(df: pd.DataFrame, numeric_cols: set[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        rendered: list[str] = []
        for col in df.columns:
            val = row[col]
            if col in numeric_cols:
                rendered.append(fmt_num(val))
            else:
                rendered.append(str(val))
        rows.append(rendered)
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def encode_target(df: pd.DataFrame) -> tuple[pd.Series, np.ndarray, LabelEncoder]:
    if "production" not in df.columns:
        raise ValueError("Column 'production' required for target generation")
    y = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y, y_encoded, label_encoder


def productivity_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "production",
        "Production_Category",
        "disease_risk_target",
        "disease_risk_score",
        "genomic_Disease_Risk_global_mode",
    ]
    out = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out["decade"] = (out["year"] // 10) * 10
    return out


def build_productivity_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    safe_num_cols = []
    moved_to_cat = []
    for col in num_cols:
        coerced = pd.to_numeric(X[col], errors="coerce")
        if X[col].notna().any() and coerced.isna().any():
            moved_to_cat.append(col)
            cat_cols.append(col)
        else:
            safe_num_cols.append(col)

    if moved_to_cat:
        print(f"Moved non-numeric productivity columns to categorical encoding: {moved_to_cat}")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), safe_num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def sustainability_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return df[feature_columns].copy()


def genomic_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["production", "Production_Category"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    if X.empty:
        raise ValueError("No genomic features available after dropping target columns")
    return X


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    categorical_cols = train_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    for col in categorical_cols:
        encoder = LabelEncoder()
        train_encoded[col] = encoder.fit_transform(train_df[col].astype(str))
        test_encoded[col] = encoder.transform(test_df[col].astype(str))

    return train_encoded, test_encoded


def scale_frame(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)
    return train_scaled, test_scaled


def select_top_features(train_encoded: pd.DataFrame, y_train: pd.Series, top_n: int = 5) -> list[str]:
    scores = mutual_info_classif(train_encoded, y_train, random_state=42)
    mi_df = pd.DataFrame({"Feature": train_encoded.columns, "MI_Score": scores})
    return mi_df.sort_values("MI_Score", ascending=False).head(top_n)["Feature"].tolist()


def render_full_tables() -> str:
    productivity = read_csv(RESULTS_DIR / "productivity_metrics.csv")
    sustainability = read_csv(RESULTS_DIR / "sustainability_metrics.csv")
    genomic = read_csv(RESULTS_DIR / "janhavi_model_metrics.csv")

    lines: list[str] = []
    lines.append("# Paper Results Tables")
    lines.append("")
    lines.append("This file is generated from the current benchmark CSVs in results/.")
    lines.append("")
    lines.append("## Feature Cleanup Note")
    lines.append("")
    lines.append("The current benchmark uses the official 9-column dataset: country, year, production, and six time-bucket water-quality features. Climate columns were excluded because the climate extract has no production-year overlap, and genomic global aggregates were excluded because no production-compatible join key exists. Earlier benchmarks containing zero-variance climate/genomic columns are not valid evidence of predictive performance. The current PSG result is the honest post-cleanup baseline.")
    lines.append("")

    lines.append("## Table II - Productivity Models")
    prod_cols = ["model", "train_accuracy", "test_accuracy", "accuracy", "precision_macro", "recall_macro", "f1_macro", "train_seconds", "infer_ms_per_1000", "model_size_mb", "peak_train_ram_mb"]
    prod_cols = [c for c in prod_cols if c in productivity.columns]
    prod_df = productivity[prod_cols].copy()
    lines.extend(markdown_table(prod_cols, rows_from_df(prod_df, set(prod_cols[1:])), right_align=[False] + [True] * (len(prod_cols) - 1)))
    lines.append("")

    lines.append("## Table III - Sustainability Models")
    sus_cols = ["model", "accuracy", "precision", "recall", "f1"]
    sus_df = sustainability[sus_cols].copy()
    lines.extend(markdown_table(sus_cols, rows_from_df(sus_df, set(sus_cols[1:])), right_align=[False] + [True] * (len(sus_cols) - 1)))
    lines.append("")

    lines.append("## Table IV - Genomic Models")
    gen_cols = ["Model", "Training_Time_s", "Inference_Time_s", "Accuracy", "Precision_macro", "Recall_macro", "F1_Score_macro"]
    gen_df = genomic[gen_cols].copy()
    lines.extend(markdown_table(gen_cols, rows_from_df(gen_df, set(gen_cols[1:])), right_align=[False] + [True] * (len(gen_cols) - 1)))
    lines.append("")

    return "\n".join(lines) + "\n"


def productivity_best_model(common_indices: list[int], y_labels: pd.Series, y_encoded: np.ndarray) -> pd.Series:
    del y_labels, y_encoded
    df = pd.read_csv(DATA_PATH)
    bundle = joblib.load(MODELS_DIR / "productivity_model.pkl")
    X = productivity_features(df)
    X_common = X.loc[common_indices]
    proba = bundle.predict_proba(X_common)
    high_idx = 0
    return pd.Series(proba[:, high_idx], index=common_indices)


def sustainability_best_model(common_indices: list[int], y_true: pd.Series) -> pd.Series:
    df = pd.read_csv(DATA_PATH)
    bundle = joblib.load(MODELS_DIR / "sustainability_model.pkl")
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])

    x_train_idx, x_test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y_true)
    _ = x_train_idx
    x_test = sustainability_features(df, feature_columns).loc[x_test_idx]

    proba = model.predict_proba(x_test.loc[common_indices])
    high_idx = int(bundle["label_encoder"].transform(["High"])[0])
    return pd.Series(proba[:, high_idx], index=common_indices)


def genomic_best_model(common_indices: list[int], y_true: pd.Series) -> pd.Series:
    df = pd.read_csv(DATA_PATH)
    X = genomic_feature_frame(df)

    x_train_idx, x_test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    x_train = X.loc[x_train_idx]
    x_test = X.loc[x_test_idx]
    y_train = y_true.loc[x_train_idx]

    x_train_encoded, x_test_encoded = encode_categoricals(x_train, x_test)
    top_features = select_top_features(x_train_encoded, y_train)
    x_train_reduced = x_train[top_features].copy()
    x_test_reduced = x_test[top_features].copy()
    x_train_reduced, x_test_reduced = encode_categoricals(x_train_reduced, x_test_reduced)
    x_train_scaled, x_test_scaled = scale_frame(x_train_reduced, x_test_reduced)

    model = joblib.load(MODELS_DIR / "feature_selector.pkl")

    proba = model.predict_proba(x_test_scaled.loc[common_indices])
    classes = list(model.classes_)
    high_idx = classes.index("High")
    return pd.Series(proba[:, high_idx], index=common_indices)


def compute_psg_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    y_labels, y_encoded, _ = encode_target(df)

    prod_train_idx, prod_test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y_encoded)
    gen_train_idx, gen_test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

    common_indices = sorted(set(prod_test_idx).intersection(set(gen_test_idx)))
    if not common_indices:
        raise RuntimeError("No overlapping held-out samples were found for PSG fusion")

    y_common = y_labels.loc[common_indices]
    y_common_enc = LabelEncoder().fit_transform(y_common)

    prod_prob = productivity_best_model(common_indices, y_labels, y_encoded)
    sus_prob = sustainability_best_model(common_indices, y_labels)
    gen_prob = genomic_best_model(common_indices, y_labels)

    fused_r = PSG_WEIGHT_PRODUCTIVITY * prod_prob + PSG_WEIGHT_SUSTAINABILITY * sus_prob + PSG_WEIGHT_GENOMIC * gen_prob
    # Use data-driven tertiles of R, matching the discretization approach used for production target
    fused_pred = pd.qcut(
        fused_r,
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates='drop'  # Handle ties if present
    ).astype(str)

    metrics = pd.DataFrame(
        [
            {
                "model": "PSG Model (Proposed)",
                "sample_count": len(common_indices),
                "accuracy": float(accuracy_score(y_common, fused_pred)),
                "precision_macro": float(precision_score(y_common, fused_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_common, fused_pred, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_common, fused_pred, average="macro", zero_division=0)),
                "fused_r_mean": float(fused_r.mean()),
                "fused_r_std": float(fused_r.std(ddof=0)),
            }
        ]
    )

    samples = pd.DataFrame(
        {
            "index": common_indices,
            "true_label": y_common.values,
            "P_r": prod_prob.values,
            "S_r": sus_prob.values,
            "G_r": gen_prob.values,
            "R": fused_r.values,
            "predicted_label": fused_pred.values,
        }
    )

    return metrics, samples


def main() -> None:
    table_md = render_full_tables()
    write_text(OUTPUT_MD, table_md)

    psg_metrics, psg_samples = compute_psg_metrics()
    psg_metrics.to_csv(OUTPUT_PSG_CSV, index=False)
    psg_samples.to_csv(OUTPUT_PSG_SAMPLES, index=False)

    print(f"Wrote paper tables -> {OUTPUT_MD}")
    print(f"Wrote PSG metrics -> {OUTPUT_PSG_CSV}")
    print(f"Wrote PSG samples -> {OUTPUT_PSG_SAMPLES}")
    print(psg_metrics.to_string(index=False))


if __name__ == "__main__":
    main()