from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data" / "processed"


def check_exists(path: Path, missing: list):
    if not path.exists():
        missing.append(str(path))


def check_csv_columns(path: Path, required_cols: list, failures: list):
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        failures.append(f"unreadable_csv:{path}:{exc}")
        return

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        failures.append(f"missing_columns:{path}:{missing_cols}")


def main() -> int:
    missing = []
    failures = []

    required_files = [
        MODELS_DIR / "productivity_model.pkl",
        MODELS_DIR / "sustainability_model.pkl",
        MODELS_DIR / "feature_selector.pkl",
        DATA_DIR / "final_dataset.csv",
        RESULTS_DIR / "productivity_metrics.csv",
        RESULTS_DIR / "sustainability_metrics.csv",
        RESULTS_DIR / "genomic_feature_importance.csv",
        RESULTS_DIR / "paper_results_tables.md",
        RESULTS_DIR / "psg_combined_metrics.csv",
        RESULTS_DIR / "psg_combined_probabilities.csv",
        RESULTS_DIR / "final_project_results_summary.md",
        RESULTS_DIR / "novelty_evidence.md",
    ]

    for p in required_files:
        check_exists(p, missing)

    check_csv_columns(RESULTS_DIR / "productivity_metrics.csv", ["model", "accuracy", "f1_macro"], failures)
    check_csv_columns(RESULTS_DIR / "sustainability_metrics.csv", ["accuracy", "f1"], failures)
    check_csv_columns(RESULTS_DIR / "genomic_feature_importance.csv", ["feature", "importance_mean", "importance_std"], failures)
    check_csv_columns(RESULTS_DIR / "psg_combined_metrics.csv", ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"], failures)

    if missing or failures:
        print("QUALITY GATE: FAILED")
        if missing:
            print("Missing artifacts:")
            for m in missing:
                print(f"- {m}")
        if failures:
            print("Validation failures:")
            for f in failures:
                print(f"- {f}")
        return 1

    print("QUALITY GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
