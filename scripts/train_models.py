from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from preprocessing.run_all import run_all as run_preprocessing

    from scripts.build_sustainability_model import main as build_sustainability_model
    from scripts.generate_final_results_summary import main as generate_summary
    from scripts.recompute_genomic_importance import main as recompute_genomic_importance
    from scripts.train_genomic_model import main as train_genomic_model
    from scripts.train_productivity_model import main as train_productivity_model
    from scripts.validation_audit import main as run_validation_audit
except ImportError:
    from preprocessing.run_all import run_all as run_preprocessing

    from build_sustainability_model import main as build_sustainability_model
    from generate_final_results_summary import main as generate_summary
    from recompute_genomic_importance import main as recompute_genomic_importance
    from train_genomic_model import main as train_genomic_model
    from train_productivity_model import main as train_productivity_model
    from validation_audit import main as run_validation_audit

RESULTS_DIR = BASE_DIR / "results"


def main() -> None:
    print("[1/7] Running preprocessing...")
    run_preprocessing()

    print("[2/7] Training productivity model...")
    train_productivity_model()

    print("[3/7] Training genomic model...")
    train_genomic_model()

    print("[4/7] Building sustainability model...")
    build_sustainability_model()

    print("[5/7] Recomputing genomic importance...")
    recompute_genomic_importance()

    print("[6/7] Running validation audit...")
    run_validation_audit()

    print("[7/7] Regenerating final summary...")
    generate_summary()

    print("Training pipeline completed.")
    print(f"Artifacts available in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
