from pathlib import Path

from preprocessing.run_all import run_all as run_preprocessing
from preprocessing.check_reconstruction_readiness import main as check_reconstruction_readiness
from preprocessing.genomic_sample_clustering import main as cluster_genomic_samples

from scripts.build_sustainability_model import main as build_sustainability_model
from scripts.generate_final_results_summary import main as generate_summary
from scripts.generate_dataset_docs import main as generate_dataset_docs
from scripts.generate_model_comparison import create_model_comparison
from scripts.generate_paper_results import main as generate_paper_results
from scripts.generate_psg_explainability import main as generate_psg_explainability
from scripts.generate_visual_reports import main as generate_visual_reports
from scripts.quality_gate import main as run_quality_gate
from scripts.recompute_genomic_importance import main as recompute_genomic_importance
from scripts.train_genomic_model import main as train_genomic_model
from scripts.train_models import run_augmented_productivity_benchmark
from scripts.validation_audit import main as run_validation_audit


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"


def main() -> None:
    print("[1/14] Running shared preprocessing pipeline...")
    run_preprocessing()

    print("[2/13] Checking official-source reconstruction readiness...")
    check_reconstruction_readiness()

    print("[3/13] Clustering official genomic samples without production integration...")
    cluster_genomic_samples()

    print("[4/14] Training augmented productivity model and benchmarks...")
    run_augmented_productivity_benchmark()

    print("[5/13] Training genomic model and benchmarks...")
    train_genomic_model()

    print("[6/13] Building sustainability model bundle...")
    build_sustainability_model()

    print("[7/13] Recomputing genomic importance (prediction sensitivity)...")
    recompute_genomic_importance()

    print("[8/13] Running validation and leakage audit...")
    run_validation_audit()

    print("[9/15] Generating visual reports...")
    generate_visual_reports()

    print("[10/15] Regenerating cross-track model comparison...")
    create_model_comparison()

    print("[11/15] Generating paper-ready tables and PSG fusion metrics...")
    generate_paper_results()

    print("[12/15] Generating PSG weight and feature-SHAP charts...")
    generate_psg_explainability()

    print("[13/15] Regenerating consolidated project summary...")
    generate_summary()

    print("[14/15] Generating dataset methodology audit...")
    generate_dataset_docs()

    print("[15/15] Running quality gate checks...")
    gate_code = run_quality_gate()
    if gate_code != 0:
        raise SystemExit(gate_code)

    print("Full rebuild pipeline complete.")
    print(f"Artifacts available in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
