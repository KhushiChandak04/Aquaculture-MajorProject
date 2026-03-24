from pathlib import Path
from datetime import datetime

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
OUT_PATH = RESULTS_DIR / "final_project_results_summary.md"


def read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def pick_best_row(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return None
    score_cols = [c for c in ["f1_macro", "f1", "F1_Score_macro", "accuracy", "Accuracy"] if c in df.columns]
    if not score_cols:
        return df.iloc[0]
    return df.sort_values(score_cols[0], ascending=False).iloc[0]


def to_pairs(row):
    if row is None:
        return []
    pairs = []
    for col in row.index:
        val = row[col]
        if isinstance(val, float):
            pairs.append((str(col), f"{val:.4f}"))
        else:
            pairs.append((str(col), str(val)))
    return pairs


def sort_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    score_cols = [c for c in ["f1_macro", "f1", "F1_Score_macro", "accuracy", "Accuracy"] if c in df.columns]
    if score_cols:
        return df.sort_values(score_cols[0], ascending=False)
    return df


def df_to_markdown_table(df: pd.DataFrame) -> list[str]:
    if df is None or len(df) == 0:
        return ["- No rows available"]

    show_df = df.copy()
    for col in show_df.columns:
        if pd.api.types.is_float_dtype(show_df[col]):
            show_df[col] = show_df[col].map(lambda x: f"{x:.4f}")

    headers = [str(c) for c in show_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in show_df.iterrows():
        vals = [str(row[c]) for c in show_df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def build_summary_text() -> str:
    productivity = read_csv(RESULTS_DIR / "productivity_metrics.csv")
    sustainability = read_csv(RESULTS_DIR / "sustainability_metrics.csv")
    genomic_metrics = read_csv(RESULTS_DIR / "janhavi_model_metrics.csv")
    genomic_importance = read_csv(RESULTS_DIR / "genomic_feature_importance.csv")

    best_prod = pick_best_row(productivity)
    best_sus = pick_best_row(sustainability)
    best_gen = pick_best_row(genomic_metrics)

    if sustainability is not None and "Unnamed: 0" in sustainability.columns and "model" not in sustainability.columns:
        sustainability = sustainability.rename(columns={"Unnamed: 0": "model"})
        best_sus = pick_best_row(sustainability)

    model_status = {
        "productivity_model.pkl": (MODELS_DIR / "productivity_model.pkl").exists(),
        "sustainability_model.pkl": (MODELS_DIR / "sustainability_model.pkl").exists(),
        "feature_selector.pkl": (MODELS_DIR / "feature_selector.pkl").exists(),
    }

    lines = []
    lines.append("# Final Project Results Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## Artifact Status")
    for name, ok in model_status.items():
        lines.append(f"- {name}: {'Available' if ok else 'Missing'}")
    lines.append("")

    lines.append("## Best Productivity Model Snapshot")
    if best_prod is None:
        lines.append("- productivity_metrics.csv not found or unreadable")
    else:
        for k, v in to_pairs(best_prod):
            lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## All Productivity Models")
    if productivity is None:
        lines.append("- productivity_metrics.csv not found or unreadable")
    else:
        lines.extend(df_to_markdown_table(sort_for_display(productivity)))
    lines.append("")

    lines.append("## Best Sustainability Model Snapshot")
    if best_sus is None:
        lines.append("- sustainability_metrics.csv not found or unreadable")
    else:
        for k, v in to_pairs(best_sus):
            lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## All Sustainability Models")
    if sustainability is None:
        lines.append("- sustainability_metrics.csv not found or unreadable")
    else:
        lines.extend(df_to_markdown_table(sort_for_display(sustainability)))
    lines.append("")

    lines.append("## Best Genomic Model Snapshot")
    if best_gen is None:
        lines.append("- janhavi_model_metrics.csv not found or unreadable")
    else:
        for k, v in to_pairs(best_gen):
            lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## All Genomic Models")
    if genomic_metrics is None:
        lines.append("- janhavi_model_metrics.csv not found or unreadable")
    else:
        lines.extend(df_to_markdown_table(sort_for_display(genomic_metrics)))
    lines.append("")

    lines.append("## Top Genomic Feature Drivers")
    if genomic_importance is None or len(genomic_importance) == 0:
        lines.append("- genomic_feature_importance.csv not found or empty")
    else:
        top = genomic_importance.sort_values("importance_mean", ascending=False).head(5)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['feature']}: mean_importance={float(row['importance_mean']):.6f}, std={float(row['importance_std']):.6f}"
            )
    lines.append("")

    lines.append("## Inference and Explainability Status")
    lines.append("- Real-time multi-model predictions: Active in Streamlit")
    lines.append("- Local SHAP explanations: Integrated with safe fallback")
    lines.append("- Forecasting: ARIMA with trend fallback")
    lines.append("- Scenario simulation: Baseline vs perturbation analysis")
    lines.append("- Policy recommendations: Hybrid model-assisted scoring plus rule layer")

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_summary_text(), encoding="utf-8")
    print(f"Wrote consolidated summary to: {OUT_PATH}")


if __name__ == "__main__":
    main()
