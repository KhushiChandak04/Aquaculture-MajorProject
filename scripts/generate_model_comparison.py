"""Generate comparative diagrams for the best model from each track using real results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_CSV = RESULTS_DIR / "model_comparison_summary.csv"
OUTPUT_PNG = RESULTS_DIR / "model_comparison_visualization.png"
OUTPUT_MD = RESULTS_DIR / "model_comparison_summary.md"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def fmt_num(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def pick_best_row(df: pd.DataFrame, score_cols: list[str]) -> pd.Series:
    for col in score_cols:
        if col in df.columns:
            scores = pd.to_numeric(df[col], errors="coerce")
            if scores.notna().any():
                return df.loc[scores.idxmax()]
    return df.iloc[0]


def normalize_track_frame(df: pd.DataFrame | None, track: str) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None

    d = df.copy()
    if track == "productivity":
        required = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
        if not set(required).issubset(d.columns):
            return None
        best = pick_best_row(d, ["f1_macro", "recall_macro", "accuracy"])
        return pd.DataFrame(
            [
                {
                    "Track": "Productivity",
                    "Model": str(best.get("model", "N/A")),
                    "Accuracy": float(best["accuracy"]),
                    "Precision": float(best["precision_macro"]),
                    "Recall": float(best["recall_macro"]),
                    "F1-Score": float(best["f1_macro"]),
                    "Training Time (s)": float(best.get("train_seconds", np.nan)),
                    "Inference Time (ms/1000)": float(best.get("infer_ms_per_1000", np.nan)),
                }
            ]
        )

    if track == "sustainability":
        required = ["model", "accuracy", "precision", "recall", "f1"]
        if not set(required).issubset(d.columns):
            return None
        best = pick_best_row(d, ["f1", "recall", "accuracy"])
        return pd.DataFrame(
            [
                {
                    "Track": "Sustainability",
                    "Model": str(best.get("model", "N/A")),
                    "Accuracy": float(best["accuracy"]),
                    "Precision": float(best["precision"]),
                    "Recall": float(best["recall"]),
                    "F1-Score": float(best["f1"]),
                    "Training Time (s)": np.nan,
                    "Inference Time (ms/1000)": np.nan,
                }
            ]
        )

    required = ["Model", "Accuracy", "Precision_macro", "Recall_macro", "F1_Score_macro"]
    if not set(required).issubset(d.columns):
        return None

    best = pick_best_row(d, ["F1_Score_macro", "Recall_macro", "Accuracy"])
    return pd.DataFrame(
        [
            {
                "Track": "Genomic",
                "Model": str(best.get("Model", "N/A")),
                "Accuracy": float(best["Accuracy"]),
                "Precision": float(best["Precision_macro"]),
                "Recall": float(best["Recall_macro"]),
                "F1-Score": float(best["F1_Score_macro"]),
                "Training Time (s)": float(best.get("Training_Time_s", np.nan)),
                "Inference Time (ms/1000)": float(best.get("Inference_Time_s", np.nan)) * 1000.0,
            }
        ]
    )


def build_comparison_frame() -> pd.DataFrame:
    productivity = normalize_track_frame(safe_read_csv(RESULTS_DIR / "productivity_metrics.csv"), "productivity")
    sustainability = normalize_track_frame(safe_read_csv(RESULTS_DIR / "sustainability_metrics.csv"), "sustainability")
    genomic = normalize_track_frame(safe_read_csv(RESULTS_DIR / "janhavi_model_metrics.csv"), "genomic")

    frames = [df for df in [productivity, sustainability, genomic] if df is not None]
    if not frames:
        raise FileNotFoundError("No track metrics CSVs were found in results/")

    return pd.concat(frames, ignore_index=True)


def save_visualization(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    models = df["Model"].astype(str).values
    x = np.arange(len(models))
    width = 0.2

    bars1 = ax.bar(x - 1.5 * width, df["Accuracy"], width, label="Accuracy", color="#2E86AB")
    bars2 = ax.bar(x - 0.5 * width, df["Precision"], width, label="Precision", color="#A23B72")
    bars3 = ax.bar(x + 0.5 * width, df["Recall"], width, label="Recall", color="#F18F01")
    bars4 = ax.bar(x + 1.5 * width, df["F1-Score"], width, label="F1-Score", color="#C73E1D")

    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Best Models Across Tracks - Derived from Real Benchmark Results", fontsize=15, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{track}\n{model}" for track, model in zip(df["Track"], models)], fontsize=11, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary_markdown(df: pd.DataFrame) -> str:
    best_by_f1 = df.sort_values(["F1-Score", "Accuracy"], ascending=[False, False]).iloc[0]

    lines = [
        "# Model Performance Comparison - Final Summary",
        "",
        "## Best Models Across Tracks",
        "",
        "This report is derived directly from the current benchmark CSVs in `results/`.",
        "",
        "### Model Overview",
        "",
        "| Track | Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms/1000) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Track"]),
                    str(row["Model"]),
                    fmt_num(row["Accuracy"]),
                    fmt_num(row["Precision"]),
                    fmt_num(row["Recall"]),
                    fmt_num(row["F1-Score"]),
                    fmt_num(row["Training Time (s)"], digits=4) if pd.notna(row["Training Time (s)"]) else "N/A",
                    fmt_num(row["Inference Time (ms/1000)"], digits=4) if pd.notna(row["Inference Time (ms/1000)"]) else "N/A",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### Key Performance Insights",
            "",
            f"**Best F1-Score:** {best_by_f1['Track']} - {best_by_f1['Model']} ({fmt_num(best_by_f1['F1-Score'])})",
            f"**Best Accuracy:** {df.sort_values(['Accuracy', 'F1-Score'], ascending=[False, False]).iloc[0]['Track']} - {df.sort_values(['Accuracy', 'F1-Score'], ascending=[False, False]).iloc[0]['Model']} ({fmt_num(df.sort_values(['Accuracy', 'F1-Score'], ascending=[False, False]).iloc[0]['Accuracy'])})",
            "",
            "### Visualization",
            "",
            "A bar chart has been generated from the current benchmark outputs.",
            "",
            f"See: `results/{OUTPUT_PNG.name}`",
            "",
            "---",
            "",
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )

    return "\n".join(lines) + "\n"


def create_model_comparison() -> None:
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    df_models = build_comparison_frame()
    df_models.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved model comparison CSV to: {OUTPUT_CSV}")

    save_visualization(df_models)
    print(f"[OK] Saved visualization to: {OUTPUT_PNG}")

    summary_md = build_summary_markdown(df_models)
    OUTPUT_MD.write_text(summary_md, encoding="utf-8")
    print(f"[OK] Saved summary markdown to: {OUTPUT_MD}")

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON - FINAL SUMMARY")
    print("=" * 80)
    print("\n" + df_models.to_string(index=False))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    create_model_comparison()