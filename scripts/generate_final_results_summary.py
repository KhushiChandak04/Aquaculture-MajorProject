from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
OUT_PATH = RESULTS_DIR / "final_project_results_summary.md"
DEPLOYMENT_ENSEMBLE_CANDIDATES = {"XGBoost", "ExtraTrees", "RandomForest"}
MIN_DEPLOY_ACCURACY = 0.75
MIN_DEPLOY_F1 = 0.75


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_sustainability(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    if "model" not in out.columns:
        unnamed_cols = [c for c in out.columns if str(c).lower().startswith("unnamed")]
        if unnamed_cols:
            out = out.rename(columns={unnamed_cols[0]: "model"})
    return out


def pick_best_row(
    df: pd.DataFrame | None,
    score_cols: list[str],
    model_col: str = "model",
) -> pd.Series | None:
    if df is None or len(df) == 0:
        return None

    for col in score_cols:
        if col in df.columns:
            scores = pd.to_numeric(df[col], errors="coerce")
            if scores.notna().any():
                return df.loc[scores.idxmax()]

    return df.iloc[0]


def pick_productivity_deployment_row(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or len(df) == 0:
        return None

    if {"train_accuracy", "test_accuracy", "recall_macro", "f1_macro"}.issubset(df.columns):
        return df.sort_values(
            ["recall_macro", "f1_macro", "test_accuracy", "infer_ms_per_1000"],
            ascending=[False, False, False, True],
        ).iloc[0]

    required_cols = {"model", "accuracy", "f1_macro", "infer_ms_per_1000"}
    if required_cols.issubset(set(df.columns)):
        candidates = df[df["model"].astype(str).isin(DEPLOYMENT_ENSEMBLE_CANDIDATES)].copy()
        if len(candidates) > 0:
            deployable = candidates[
                (pd.to_numeric(candidates["accuracy"], errors="coerce") >= MIN_DEPLOY_ACCURACY)
                & (pd.to_numeric(candidates["f1_macro"], errors="coerce") >= MIN_DEPLOY_F1)
            ].copy()

            if len(deployable) > 0:
                return deployable.sort_values(
                    ["infer_ms_per_1000", "recall_macro", "f1_macro", "accuracy", "model_size_mb", "train_seconds"],
                    ascending=[True, False, False, False, True, True],
                ).iloc[0]

    return pick_best_row(df, score_cols=["f1_macro", "recall_macro", "accuracy"], model_col="model")


def fmt_num(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def status_label(path: Path) -> str:
    return "Available" if path.exists() else "Missing"


def markdown_table(headers: list[str], rows: list[list[str]], right_align: list[bool] | None = None) -> list[str]:
    if right_align is None:
        right_align = [False] * len(headers)

    lines = ["| " + " | ".join(headers) + " |"]
    align_parts = ["---:" if right_align[i] else "---" for i in range(len(headers))]
    lines.append("|" + "|".join(align_parts) + "|")

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return lines


def rows_from_df(df: pd.DataFrame, columns: list[str], numeric_cols: set[str], digits: int = 4) -> list[list[str]]:
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        out_row: list[str] = []
        for col in columns:
            val = row[col]
            if col in numeric_cols:
                out_row.append(fmt_num(val, digits=digits))
            else:
                out_row.append(str(val))
        rows.append(out_row)
    return rows


def build_summary_text() -> str:
    productivity = read_csv(RESULTS_DIR / "productivity_metrics.csv")
    sustainability = normalize_sustainability(read_csv(RESULTS_DIR / "sustainability_metrics.csv"))
    genomic_metrics = read_csv(RESULTS_DIR / "janhavi_model_metrics.csv")
    genomic_importance = read_csv(RESULTS_DIR / "genomic_feature_importance.csv")
    psg_metrics = read_csv(RESULTS_DIR / "psg_combined_metrics.csv")
    psg_contributions = read_csv(RESULTS_DIR / "psg_track_contributions.csv")
    productivity_shap = read_csv(RESULTS_DIR / "productivity_feature_shap_importance.csv")

    best_prod = pick_productivity_deployment_row(productivity)
    best_sus = pick_best_row(
        sustainability,
        score_cols=["f1", "recall", "accuracy"],
        model_col="model",
    )
    best_gen = pick_best_row(
        genomic_metrics,
        score_cols=["F1_Score_macro", "Recall_macro", "Accuracy"],
        model_col="Model",
    )

    lines: list[str] = []
    lines.append("# Final Project Results Summary")
    lines.append("")
    lines.append('<p align="center"><strong>Consolidated Evidence Report for Productivity, Sustainability, and Genomic Tracks</strong></p>')
    lines.append("")
    lines.append('<p align="center">')
    lines.append('\t<img src="https://img.shields.io/badge/Report_Type-Consolidated_Results-1D3557" alt="Report Type" />')
    lines.append('\t<img src="https://img.shields.io/badge/Scope-Tri_Track_Modeling-2A9D8F" alt="Scope" />')
    lines.append('\t<img src="https://img.shields.io/badge/Explainability-SHAP_Integrated-0C7BDC" alt="Explainability" />')
    lines.append("</p>")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("## Feature Cleanup Interpretation")
    lines.append("")
    psg_accuracy = fmt_num(psg_metrics.iloc[0].get("accuracy", "N/A"), digits=4) if psg_metrics is not None and len(psg_metrics) else "N/A"
    psg_f1 = fmt_num(psg_metrics.iloc[0].get("f1_macro", "N/A"), digits=4) if psg_metrics is not None and len(psg_metrics) else "N/A"
    lines.append(f"The current results use the rebuilt official 9-column dataset: country, year, production, and six time-bucket water-quality features. Climate columns were excluded because the source covers 2024-2026 while production covers 1960-2018. Genomic global aggregates were excluded because no production-compatible join key exists. Country is label-encoded, year is numeric, decade is derived from year, and the production target uses log1p quantile construction. Earlier optimistic metrics that used zero-variance climate/genomic columns are not valid predictive evidence. The current PSG accuracy is {psg_accuracy} and macro-F1 is {psg_f1}; these are the honest augmented-feature results.")
    lines.append("")

    lines.append("## PSG Weighting and Feature Attribution")
    lines.append("")
    lines.append("The implemented fusion rule uses fixed coefficients: Productivity 50%, Sustainability 35%, and Genomic 15%. Productivity has the largest configured coefficient because that is how the current rule is coded; the repository contains no documented learned-weight procedure or validation study establishing that 50/35/15 is optimal. The pie chart shows configured coefficients, not model-learned importance.")
    lines.append("")
    lines.append("| Track | Configured weight | Realized mean contribution share on PSG held-out samples |")
    lines.append("|---|---:|---:|")
    if psg_contributions is not None and len(psg_contributions):
        for _, row in psg_contributions.iterrows():
            lines.append(f"| {row['track']} | {float(row['configured_weight_pct']):.1f}% | {float(row['realized_mean_contribution_pct']):.1f}% |")
    lines.append("")
    lines.append("Configured weights reflect the current rule, not a learned optimum: Productivity is highest at 50%, Sustainability is 35%, and Genomic is 15%. The repository does not document a tuning or validation study that established this ratio; productivity's larger coefficient is a design choice, not a SHAP-derived conclusion.")
    lines.append("")
    lines.append("The feature SHAP chart is computed for the selected productivity LightGBM model on its held-out split. It ranks input-feature influence within that model; it does not justify or estimate cross-track PSG weights. Current results rank country encoding and year highest. Country uses ordinal/label-style codes, which impose an artificial ordering, and year/decade are correlated; interpret those attributions cautiously. The six water inputs are time-bucket measurements, not country-specific observations.")
    lines.append("")
    lines.append("![Configured PSG track weights](psg_track_weight_split.png)")
    lines.append("")
    lines.append("![Productivity feature SHAP importance](productivity_feature_shap_importance.png)")
    lines.append("")
    lines.append("### Productivity Feature SHAP Values")
    lines.append("")
    if productivity_shap is not None and len(productivity_shap):
        shap_rows = rows_from_df(
            productivity_shap,
            ["feature", "mean_abs_shap", "relative_importance_pct"],
            numeric_cols={"mean_abs_shap", "relative_importance_pct"},
            digits=4,
        )
        lines.extend(markdown_table(["Feature", "Mean |SHAP|", "Share of total (%)"], shap_rows))
    else:
        lines.append("- Feature SHAP values are unavailable.")
    lines.append("")
    lines.append("")

    lines.append("## 1. Artifact Readiness Matrix")
    artifact_rows = [
        ["Productivity model", "models/productivity_model.pkl", status_label(MODELS_DIR / "productivity_model.pkl")],
        ["Sustainability model", "models/sustainability_model.pkl", status_label(MODELS_DIR / "sustainability_model.pkl")],
        ["Genomic model", "models/feature_selector.pkl", status_label(MODELS_DIR / "feature_selector.pkl")],
        ["Productivity metrics", "results/productivity_metrics.csv", status_label(RESULTS_DIR / "productivity_metrics.csv")],
        ["Sustainability metrics", "results/sustainability_metrics.csv", status_label(RESULTS_DIR / "sustainability_metrics.csv")],
        ["Genomic metrics", "results/janhavi_model_metrics.csv", status_label(RESULTS_DIR / "janhavi_model_metrics.csv")],
        ["Genomic feature importance", "results/genomic_feature_importance.csv", status_label(RESULTS_DIR / "genomic_feature_importance.csv")],
        ["Validation audit", "results/validation_audit.md", status_label(RESULTS_DIR / "validation_audit.md")],
    ]
    lines.extend(markdown_table(["Artifact", "Path", "Status"], artifact_rows))
    lines.append("")

    lines.append("## 2. Best Model Snapshots")
    lines.append("")

    prod_name = str(best_prod.get("model", "N/A")) if best_prod is not None else "N/A"
    lines.append(f"### 2.1 Productivity Track (Best: {prod_name})")
    prod_snapshot_rows = [["Accuracy", "N/A"], ["Precision (macro)", "N/A"], ["Recall (macro)", "N/A"], ["F1 (macro)", "N/A"], ["Training Time (s)", "N/A"], ["Inference per 1000 rows (ms)", "N/A"], ["Model Size (MB)", "N/A"], ["Peak Train RAM (MB)", "N/A"]]
    if best_prod is not None:
        prod_snapshot_rows = [
            ["Accuracy", fmt_num(best_prod.get("accuracy", "N/A"))],
            ["Precision (macro)", fmt_num(best_prod.get("precision_macro", "N/A"))],
            ["Recall (macro)", fmt_num(best_prod.get("recall_macro", "N/A"))],
            ["F1 (macro)", fmt_num(best_prod.get("f1_macro", "N/A"))],
            ["Training Time (s)", fmt_num(best_prod.get("train_seconds", "N/A"))],
            ["Inference per 1000 rows (ms)", fmt_num(best_prod.get("infer_ms_per_1000", "N/A"))],
            ["Model Size (MB)", fmt_num(best_prod.get("model_size_mb", "N/A"))],
            ["Peak Train RAM (MB)", fmt_num(best_prod.get("peak_train_ram_mb", "N/A"))],
        ]
    lines.extend(markdown_table(["Metric", "Value"], prod_snapshot_rows))
    lines.append("")

    sus_name = str(best_sus.get("model", "N/A")) if best_sus is not None else "N/A"
    lines.append(f"### 2.2 Sustainability Track (Best: {sus_name})")
    sus_snapshot_rows = [["Accuracy", "N/A"], ["Precision (macro)", "N/A"], ["Recall (macro)", "N/A"], ["F1 (macro)", "N/A"]]
    if best_sus is not None:
        sus_snapshot_rows = [
            ["Accuracy", fmt_num(best_sus.get("accuracy", "N/A"))],
            ["Precision (macro)", fmt_num(best_sus.get("precision", "N/A"))],
            ["Recall (macro)", fmt_num(best_sus.get("recall", "N/A"))],
            ["F1 (macro)", fmt_num(best_sus.get("f1", "N/A"))],
        ]
    lines.extend(markdown_table(["Metric", "Value"], sus_snapshot_rows))
    lines.append("")

    gen_name = str(best_gen.get("Model", "N/A")) if best_gen is not None else "N/A"
    lines.append(f"### 2.3 Genomic Track (Best: {gen_name})")
    gen_snapshot_rows = [["Accuracy", "N/A"], ["Precision (macro)", "N/A"], ["Recall (macro)", "N/A"], ["F1 (macro)", "N/A"], ["Training Time (s)", "N/A"], ["Inference Time (s)", "N/A"]]
    if best_gen is not None:
        gen_snapshot_rows = [
            ["Accuracy", fmt_num(best_gen.get("Accuracy", "N/A"))],
            ["Precision (macro)", fmt_num(best_gen.get("Precision_macro", "N/A"))],
            ["Recall (macro)", fmt_num(best_gen.get("Recall_macro", "N/A"))],
            ["F1 (macro)", fmt_num(best_gen.get("F1_Score_macro", "N/A"))],
            ["Training Time (s)", fmt_num(best_gen.get("Training_Time_s", "N/A"))],
            ["Inference Time (s)", fmt_num(best_gen.get("Inference_Time_s", "N/A"))],
        ]
    lines.extend(markdown_table(["Metric", "Value"], gen_snapshot_rows))
    lines.append("")

    lines.append("## 3. Full Benchmark Tables")
    lines.append("")

    lines.append("### 3.1 Productivity Models")
    if productivity is None or len(productivity) == 0:
        lines.append("- productivity_metrics.csv not found or unreadable")
    else:
        prod_cols = [
            "model",
            "train_accuracy",
            "test_accuracy",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "train_seconds",
            "infer_ms_per_1000",
            "model_size_mb",
            "peak_train_ram_mb",
        ]
        prod_cols = [c for c in prod_cols if c in productivity.columns]
        prod_rows = rows_from_df(productivity[prod_cols], prod_cols, numeric_cols=set(prod_cols[1:]), digits=4)
        lines.extend(markdown_table(prod_cols, prod_rows, right_align=[False] + [True] * (len(prod_cols) - 1)))
    lines.append("")

    lines.append("### 3.2 Sustainability Models")
    if sustainability is None or len(sustainability) == 0:
        lines.append("- sustainability_metrics.csv not found or unreadable")
    else:
        sus_cols = ["model", "accuracy", "precision", "recall", "f1"]
        sus_cols = [c for c in sus_cols if c in sustainability.columns]
        sus_rows = rows_from_df(sustainability[sus_cols], sus_cols, numeric_cols=set(sus_cols[1:]), digits=4)
        lines.extend(markdown_table(sus_cols, sus_rows, right_align=[False] + [True] * (len(sus_cols) - 1)))
    lines.append("")

    lines.append("### 3.3 Genomic Models")
    if genomic_metrics is None or len(genomic_metrics) == 0:
        lines.append("- janhavi_model_metrics.csv not found or unreadable")
    else:
        gen_cols = ["Model", "Training_Time_s", "Inference_Time_s", "Accuracy", "Precision_macro", "Recall_macro", "F1_Score_macro"]
        gen_cols = [c for c in gen_cols if c in genomic_metrics.columns]
        gen_rows = rows_from_df(genomic_metrics[gen_cols], gen_cols, numeric_cols=set(gen_cols[1:]), digits=4)
        lines.extend(markdown_table(gen_cols, gen_rows, right_align=[False] + [True] * (len(gen_cols) - 1)))
    lines.append("")

    lines.append("## 4. Top Genomic Feature Drivers")
    if genomic_importance is None or len(genomic_importance) == 0:
        lines.append("- genomic_feature_importance.csv not found or unreadable")
    else:
        gi = genomic_importance.copy()
        if {"importance_mean", "importance_std"}.issubset(set(gi.columns)):
            gi["importance_mean"] = pd.to_numeric(gi["importance_mean"], errors="coerce")
            gi["importance_std"] = pd.to_numeric(gi["importance_std"], errors="coerce")
            gi = gi.sort_values("importance_mean", ascending=False).head(5)
            feat_cols = ["feature", "importance_mean", "importance_std"]
            feat_rows = rows_from_df(gi[feat_cols], feat_cols, numeric_cols={"importance_mean", "importance_std"}, digits=6)
            lines.extend(markdown_table(["Feature", "Mean Importance", "Std"], feat_rows, right_align=[False, True, True]))
        else:
            lines.append("- genomic_feature_importance.csv missing required columns")
    lines.append("")

    lines.append("## 5. Integrated Inference Capability Status")
    lines.append("1. Real-time synchronized multi-model predictions: Active.")
    lines.append("2. Track-wise SHAP explainability with fallback paths: Active.")
    lines.append("3. Forecasting with ARIMA/trend fallback: Active.")
    lines.append("4. Scenario simulation with confidence deltas: Active.")
    lines.append("5. Policy recommendation layer with model-conditioned logic: Active.")
    lines.append("")

    lines.append("## 6. Governance Notes")
    lines.append("1. This summary is a consolidated operational-research view and should be read alongside:")
    lines.append("\t - results/validation_audit.md")
    lines.append("\t - results/xai_evidence_report.md")
    lines.append("2. Model outputs are predictive evidence and should be interpreted with domain and ecological validation.")

    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_summary_text(), encoding="utf-8")
    print(f"Wrote consolidated summary to: {OUT_PATH}")


if __name__ == "__main__":
    main()
