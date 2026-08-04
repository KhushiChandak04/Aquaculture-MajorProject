from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from preprocessing.run_all import run_all as run_preprocessing

    from scripts.build_sustainability_model import main as build_sustainability_model
    from scripts.generate_final_results_summary import main as generate_summary
    from scripts.generate_paper_results import main as generate_paper_results
    from scripts.generate_visual_reports import main as generate_visual_reports
    from scripts.recompute_genomic_importance import main as recompute_genomic_importance
    from scripts.train_genomic_model import main as train_genomic_model
    from scripts.train_productivity_model import main as train_productivity_model
    from scripts.validation_audit import main as run_validation_audit
except ImportError:
    from preprocessing.run_all import run_all as run_preprocessing

    from build_sustainability_model import main as build_sustainability_model
    from generate_final_results_summary import main as generate_summary
    from generate_paper_results import main as generate_paper_results
    from generate_visual_reports import main as generate_visual_reports
    from recompute_genomic_importance import main as recompute_genomic_importance
    from train_genomic_model import main as train_genomic_model
    from train_productivity_model import main as train_productivity_model
    from validation_audit import main as run_validation_audit

RESULTS_DIR = BASE_DIR / "results"


def main() -> None:
    print("[1/8] Running preprocessing...")
    run_preprocessing()

    print("[2/8] Training productivity model...")
    train_productivity_model()

    print("[3/8] Training genomic model...")
    train_genomic_model()

    print("[4/8] Building sustainability model...")
    build_sustainability_model()

    print("[5/8] Recomputing genomic importance...")
    recompute_genomic_importance()

    print("[6/8] Running validation audit...")
    run_validation_audit()

    print("[7/8] Generating visual reports...")
    generate_visual_reports()

    print("[8/9] Generating paper-ready tables and PSG fusion metrics...")
    generate_paper_results()

    print("[9/9] Regenerating final summary...")
    generate_summary()

    print("Training pipeline completed.")
    print(f"Artifacts available in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
