from pathlib import Path

try:
    from .climate_preprocess import preprocess_climate
    from .genomic_preprocess import preprocess_genomic
    from .production_preprocess import preprocess_production
    from .water_preprocess import preprocess_water
except ImportError:
    from climate_preprocess import preprocess_climate
    from genomic_preprocess import preprocess_genomic
    from production_preprocess import preprocess_production
    from water_preprocess import preprocess_water


def run_all():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"

    preprocess_production(
        raw_dir / "production.csv",
        processed_dir / "production_clean.csv",
    )

    preprocess_water(
        raw_dir / "water_quality.csv",
        processed_dir / "water_quality_clean.csv",
    )

    preprocess_climate(
        raw_dir / "climate.csv",
        processed_dir / "climate_clean.csv",
    )

    preprocess_genomic(
        raw_dir / "genomic.csv",
        processed_dir / "genomic_clean.csv",
    )

    print("All preprocessing complete.")


if __name__ == "__main__":
    run_all()
