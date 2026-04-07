from pathlib import Path

from preprocessing.run_all import run_all as run_preprocessing

from scripts.build_sustainability_model import main as build_sustainability_model
from scripts.generate_final_results_summary import main as generate_summary
from scripts.generate_visual_reports import main as generate_visual_reports
from scripts.quality_gate import main as run_quality_gate
from scripts.recompute_genomic_importance import main as recompute_genomic_importance
from scripts.train_genomic_model import main as train_genomic_model
from scripts.train_productivity_model import main as train_productivity_model
from scripts.validation_audit import main as run_validation_audit


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"


def main() -> None:
    print("[1/9] Running shared preprocessing pipeline...")
    run_preprocessing()

    print("[2/9] Training productivity model and benchmarks...")
    train_productivity_model()

    print("[3/9] Training genomic model and benchmarks...")
    train_genomic_model()

    print("[4/9] Building sustainability model bundle...")
    build_sustainability_model()

    print("[5/9] Recomputing genomic importance (prediction sensitivity)...")
    recompute_genomic_importance()

    print("[6/9] Running validation and leakage audit...")
    run_validation_audit()

    print("[7/9] Generating visual reports...")
    generate_visual_reports()

    print("[8/9] Regenerating consolidated project summary...")
    generate_summary()

    print("[9/9] Running quality gate checks...")
    gate_code = run_quality_gate()
    if gate_code != 0:
        raise SystemExit(gate_code)

    print("Full rebuild pipeline complete.")
    print(f"Artifacts available in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
