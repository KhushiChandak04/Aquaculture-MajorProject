"""Generate PSG weight and productivity-feature SHAP figures from current artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "productivity_model.pkl"
RESULTS_DIR = BASE_DIR / "results"
WEIGHT_PIE_PATH = RESULTS_DIR / "psg_track_weight_split.png"
SHAP_BAR_PATH = RESULTS_DIR / "productivity_feature_shap_importance.png"
SHAP_CSV_PATH = RESULTS_DIR / "productivity_feature_shap_importance.csv"
CONTRIBUTION_CSV_PATH = RESULTS_DIR / "psg_track_contributions.csv"

PSG_WEIGHTS = {
    "Productivity": 0.50,
    "Sustainability": 0.35,
    "Genomic": 0.15,
}
FEATURE_COLUMNS = [
    "country",
    "year",
    "water_Salinity (ppt)",
    "water_pH",
    "water_SecchiDepth (m)",
    "water_WaterDepth (m)",
    "water_WaterTemp (C)",
    "water_AirTemp (C)",
]


def make_productivity_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    production_log = np.log1p(pd.to_numeric(df["production"], errors="coerce").clip(lower=0))
    labels = pd.qcut(production_log, q=3, labels=["Low", "Medium", "High"]).astype(str)
    X = df[FEATURE_COLUMNS].copy()
    X["year"] = pd.to_numeric(X["year"], errors="coerce")
    X["decade"] = (X["year"] // 10) * 10
    y = pd.factorize(labels, sort=True)[0]
    return X, y


def mean_absolute_shap_by_feature() -> pd.DataFrame:
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("Current dataset and productivity model artifacts are required")

    df = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURE_COLUMNS + ["production"] if column not in df.columns]
    if missing:
        raise ValueError(f"Current final dataset is missing required features: {missing}")

    pipeline = joblib.load(MODEL_PATH)
    expected_columns = list(pipeline.named_steps["prep"].feature_names_in_)
    y_source = pd.to_numeric(df["production"], errors="coerce").clip(lower=0)
    if "decade" in expected_columns:
        y_source = np.log1p(y_source)
    labels = pd.qcut(y_source, q=3, labels=["Low", "Medium", "High"]).astype(str)
    X = df[[column for column in expected_columns if column != "decade"]].copy()
    if "year" in X.columns:
        X["year"] = pd.to_numeric(X["year"], errors="coerce")
        if "decade" in expected_columns:
            X["decade"] = (X["year"] // 10) * 10
    y = pd.factorize(labels, sort=True)[0]
    X = X[expected_columns]
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = pipeline.named_steps["prep"]
    estimator = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X_test)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

    explanation = shap.TreeExplainer(estimator).shap_values(X_transformed)
    if isinstance(explanation, list):
        values = np.stack([np.asarray(value) for value in explanation], axis=-1)
    else:
        values = np.asarray(explanation)

    if values.ndim == 2:
        mean_abs = np.mean(np.abs(values), axis=0)
    elif values.ndim == 3 and values.shape[0] == len(X_test) and values.shape[1] == len(feature_names):
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
    elif values.ndim == 3 and values.shape[1] == len(X_test) and values.shape[2] == len(feature_names):
        mean_abs = np.mean(np.abs(values), axis=(0, 1))
    else:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    records = []
    for name, value in zip(feature_names, mean_abs):
        feature = name.split("__", 1)[-1]
        if feature == "country":
            feature = "country (encoded)"
        records.append({"feature": feature, "mean_abs_shap": float(value)})

    result = pd.DataFrame(records).groupby("feature", as_index=False).sum()
    total = result["mean_abs_shap"].sum()
    result["relative_importance_pct"] = (
        result["mean_abs_shap"] / total * 100.0 if total > 0 else 0.0
    )
    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def write_realized_contributions() -> pd.DataFrame:
    sample_path = RESULTS_DIR / "psg_combined_probabilities.csv"
    samples = pd.read_csv(sample_path)
    track_columns = {"Productivity": ("P_r", PSG_WEIGHTS["Productivity"]),
                     "Sustainability": ("S_r", PSG_WEIGHTS["Sustainability"]),
                     "Genomic": ("G_r", PSG_WEIGHTS["Genomic"])}
    rows = []
    for track, (column, weight) in track_columns.items():
        weighted = weight * pd.to_numeric(samples[column], errors="coerce")
        rows.append(
            {
                "track": track,
                "configured_weight_pct": 100.0 * weight,
                "mean_track_probability": float(samples[column].mean()),
                "mean_weighted_contribution": float(weighted.mean()),
            }
        )
    result = pd.DataFrame(rows)
    contribution_total = result["mean_weighted_contribution"].sum()
    result["realized_mean_contribution_pct"] = (
        100.0 * result["mean_weighted_contribution"] / contribution_total
        if contribution_total
        else 0.0
    )
    result.to_csv(CONTRIBUTION_CSV_PATH, index=False)
    return result


def plot_weight_split() -> None:
    labels = list(PSG_WEIGHTS)
    values = [PSG_WEIGHTS[label] * 100 for label in labels]
    colors = ["#247BA0", "#F2A541", "#5B8E7D"]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    wedges, _, _ = ax.pie(
        values,
        labels=None,
        autopct="%1.0f%%",
        startangle=90,
        counterclock=False,
        colors=colors,
        pctdistance=0.68,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 13, "weight": "bold"},
    )
    ax.legend(
        wedges,
        [f"{name}: {weight:.0%}" for name, weight in PSG_WEIGHTS.items()],
        title="Configured PSG weights",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    ax.set_title("PSG Fusion: Fixed Track-Weight Allocation", fontsize=16, pad=18)
    fig.text(
        0.5,
        0.02,
        "Weights are configured constants, not learned from SHAP or optimized from validation data.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(WEIGHT_PIE_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_feature_shap(importance: pd.DataFrame, model_name: str, target_description: str) -> None:
    plotted = importance.sort_values("relative_importance_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    bars = ax.barh(
        plotted["feature"],
        plotted["relative_importance_pct"],
        color="#247BA0",
        edgecolor="white",
    )
    max_value = max(float(plotted["relative_importance_pct"].max()), 1.0)
    ax.set_xlim(0, max_value * 1.3)
    for bar, (_, row) in zip(bars, plotted.iterrows()):
        ax.text(
            bar.get_width() + max_value * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{row['relative_importance_pct']:.1f}%  (mean |SHAP| {row['mean_abs_shap']:.4f})",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Share of total mean absolute SHAP value (%)")
    ax.set_title(f"Productivity {model_name}: Feature-Level SHAP Importance", fontsize=15, pad=14)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.015,
        f"Held-out split; {target_description}. Feature SHAP describes this model only and does not determine PSG track weights.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(SHAP_BAR_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    importance = mean_absolute_shap_by_feature()
    importance.to_csv(SHAP_CSV_PATH, index=False)
    model = joblib.load(MODEL_PATH)
    estimator = model.named_steps.get("model", model)
    model_name = {
        "LGBMClassifier": "LightGBM",
        "XGBClassifier": "XGBoost",
        "ExtraTreesClassifier": "ExtraTrees",
        "RandomForestClassifier": "RandomForest",
        "HistGradientBoostingClassifier": "HistGB",
    }.get(estimator.__class__.__name__, estimator.__class__.__name__)
    feature_names = list(model.named_steps["prep"].feature_names_in_)
    target_description = "log1p-production tertiles" if "decade" in feature_names else "production tertiles"
    plot_weight_split()
    plot_feature_shap(importance, model_name, target_description)
    print(f"Wrote PSG weight chart -> {WEIGHT_PIE_PATH}")
    print(f"Wrote feature SHAP chart -> {SHAP_BAR_PATH}")
    print(f"Wrote SHAP values -> {SHAP_CSV_PATH}")
    contributions = write_realized_contributions()
    print(f"Wrote realized PSG contributions -> {CONTRIBUTION_CSV_PATH}")
    print(contributions.to_string(index=False))
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()