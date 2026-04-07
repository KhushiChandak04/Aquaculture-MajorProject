from __future__ import annotations

from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def normalize_track_metrics(df: pd.DataFrame | None, track: str) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None

    d = df.copy()
    if track == "productivity":
        model_col = first_existing(d, ["model", "Model"])
        acc_col = first_existing(d, ["accuracy", "Accuracy"])
        pre_col = first_existing(d, ["precision_macro", "precision", "Precision_macro"])
        rec_col = first_existing(d, ["recall_macro", "recall", "Recall_macro"])
        f1_col = first_existing(d, ["f1_macro", "f1", "F1_Score_macro"])
    elif track == "sustainability":
        model_col = first_existing(d, ["model", "Model"])
        acc_col = first_existing(d, ["accuracy", "Accuracy"])
        pre_col = first_existing(d, ["precision", "precision_macro", "Precision_macro"])
        rec_col = first_existing(d, ["recall", "recall_macro", "Recall_macro"])
        f1_col = first_existing(d, ["f1", "f1_macro", "F1_Score_macro"])
    else:  # genomic
        model_col = first_existing(d, ["Model", "model"])
        acc_col = first_existing(d, ["Accuracy", "accuracy"])
        pre_col = first_existing(d, ["Precision_macro", "precision_macro", "precision"])
        rec_col = first_existing(d, ["Recall_macro", "recall_macro", "recall"])
        f1_col = first_existing(d, ["F1_Score_macro", "f1_macro", "f1"])

    if not acc_col or not pre_col or not rec_col or not f1_col:
        return None

    out = pd.DataFrame(
        {
            "model": d[model_col].astype(str) if model_col else f"{track.title()}Model",
            "accuracy": pd.to_numeric(d[acc_col], errors="coerce"),
            "precision": pd.to_numeric(d[pre_col], errors="coerce"),
            "recall": pd.to_numeric(d[rec_col], errors="coerce"),
            "f1": pd.to_numeric(d[f1_col], errors="coerce"),
        }
    ).dropna(subset=["accuracy", "precision", "recall", "f1"]) 

    if len(out) == 0:
        return None

    return out.sort_values(["f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)


def plot_spiral(df: pd.DataFrame, title: str, out_path: Path) -> None:
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    metric_cols = ["accuracy", "precision", "recall", "f1"]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(df))))

    for idx, (_, row) in enumerate(df.iterrows()):
        vals = [float(row[c]) for c in metric_cols]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.0, label=str(row["model"]), color=colors[idx])
        ax.fill(angles, vals, alpha=0.08, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_title(title, pad=18)
    ax.grid(alpha=0.25)

    if len(df) <= 10:
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_train_test_accuracy(track: str, train_acc: float, test_acc: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Train", "Test"]
    vals = [float(train_acc), float(test_acc)]
    bars = ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e"], width=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{track.title()} Train vs Test Accuracy")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_productivity_train_test() -> tuple[float, float] | None:
    if not DATA_PATH.exists() or not (MODELS_DIR / "productivity_model.pkl").exists():
        return None

    df = pd.read_csv(DATA_PATH)
    if "production" not in df.columns:
        return None

    y = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)
    drop_cols = [
        "production",
        "Production_Category",
        "disease_risk_target",
        "disease_risk_score",
        "genomic_Disease_Risk_global_mode",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    y_enc = LabelEncoder().fit_transform(y)
    xtr, xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    model = joblib.load(MODELS_DIR / "productivity_model.pkl")
    tr_pred = model.predict(xtr)
    te_pred = model.predict(xte)
    return float(accuracy_score(ytr, tr_pred)), float(accuracy_score(yte, te_pred))


def compute_sustainability_train_test() -> tuple[float, float] | None:
    model_path = MODELS_DIR / "sustainability_model.pkl"
    if not DATA_PATH.exists() or not model_path.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    if "production" not in df.columns:
        return None

    bundle = joblib.load(model_path)
    model = bundle.get("model") if isinstance(bundle, dict) else bundle
    feat_cols = bundle.get("feature_columns") if isinstance(bundle, dict) else None
    le = bundle.get("label_encoder") if isinstance(bundle, dict) else None
    if model is None or feat_cols is None or le is None:
        return None

    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].copy()
    y = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)
    y_enc = le.transform(y)

    xtr, xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    tr_pred = model.predict(xtr)
    te_pred = model.predict(xte)
    return float(accuracy_score(ytr, tr_pred)), float(accuracy_score(yte, te_pred))


def compute_genomic_train_test() -> tuple[float, float] | None:
    model_path = MODELS_DIR / "feature_selector.pkl"
    if not DATA_PATH.exists() or not model_path.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    if "production" not in df.columns:
        return None

    y = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"]).astype(str)
    genomic_cols = [c for c in df.columns if c.startswith("genomic_")]
    context_cols = [c for c in ["country", "year", "temperature_celsius", "precip_mm", "humidity"] if c in df.columns]
    cols = genomic_cols + context_cols
    if not cols:
        cols = [c for c in df.columns if c not in ["production", "Production_Category", "disease_risk_target", "disease_risk_score"]]

    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.factorize(X[c].astype(str))[0]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = joblib.load(model_path)
    tr_pred = model.predict(xtr)
    te_pred = model.predict(xte)
    return float(accuracy_score(ytr, tr_pred)), float(accuracy_score(yte, te_pred))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prod_metrics = normalize_track_metrics(safe_read_csv(RESULTS_DIR / "productivity_metrics.csv"), "productivity")
    sus_metrics = normalize_track_metrics(safe_read_csv(RESULTS_DIR / "sustainability_metrics.csv"), "sustainability")
    gen_metrics = normalize_track_metrics(safe_read_csv(RESULTS_DIR / "janhavi_model_metrics.csv"), "genomic")

    if prod_metrics is not None:
        plot_spiral(prod_metrics, "Productivity Spiral Net (All Models)", RESULTS_DIR / "productivity_spiral_all_models.png")
        for _, row in prod_metrics.iterrows():
            plot_spiral(pd.DataFrame([row]), f"Productivity Spiral Net - {row['model']}", RESULTS_DIR / f"productivity_spiral_{slugify(row['model'])}.png")

    if sus_metrics is not None:
        plot_spiral(sus_metrics, "Sustainability Spiral Net", RESULTS_DIR / "sustainability_spiral_all_models.png")
        for _, row in sus_metrics.iterrows():
            plot_spiral(pd.DataFrame([row]), f"Sustainability Spiral Net - {row['model']}", RESULTS_DIR / f"sustainability_spiral_{slugify(row['model'])}.png")

    if gen_metrics is not None:
        plot_spiral(gen_metrics, "Genomic Spiral Net", RESULTS_DIR / "genomic_spiral_all_models.png")
        for _, row in gen_metrics.iterrows():
            plot_spiral(pd.DataFrame([row]), f"Genomic Spiral Net - {row['model']}", RESULTS_DIR / f"genomic_spiral_{slugify(row['model'])}.png")

    prod_tt = compute_productivity_train_test()
    if prod_tt is not None:
        plot_train_test_accuracy("productivity", prod_tt[0], prod_tt[1], RESULTS_DIR / "productivity_train_vs_test_accuracy.png")

    sus_tt = compute_sustainability_train_test()
    if sus_tt is not None:
        plot_train_test_accuracy("sustainability", sus_tt[0], sus_tt[1], RESULTS_DIR / "sustainability_train_vs_test_accuracy.png")

    gen_tt = compute_genomic_train_test()
    if gen_tt is not None:
        plot_train_test_accuracy("genomic", gen_tt[0], gen_tt[1], RESULTS_DIR / "genomic_train_vs_test_accuracy.png")

    # Combined train-vs-test overview chart for all tracks.
    tt_rows = []
    if prod_tt is not None:
        tt_rows.append(("Productivity", prod_tt[0], prod_tt[1]))
    if sus_tt is not None:
        tt_rows.append(("Sustainability", sus_tt[0], sus_tt[1]))
    if gen_tt is not None:
        tt_rows.append(("Genomic", gen_tt[0], gen_tt[1]))

    if tt_rows:
        labels = [r[0] for r in tt_rows]
        train_vals = [r[1] for r in tt_rows]
        test_vals = [r[2] for r in tt_rows]

        x = np.arange(len(labels), dtype=float)
        w = 0.34
        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        b1 = ax.bar(x - w / 2, train_vals, width=w, label="Train", color="#1f77b4")
        b2 = ax.bar(x + w / 2, test_vals, width=w, label="Test", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("Train vs Test Accuracy Across Tracks")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)

        for bars in [b1, b2]:
            for b in bars:
                v = float(b.get_height())
                ax.text(b.get_x() + b.get_width() / 2.0, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "all_tracks_train_vs_test_accuracy.png", dpi=220)
        plt.close(fig)

    print(f"Saved visual reports in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
