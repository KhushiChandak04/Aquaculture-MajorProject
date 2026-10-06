"""Check whether official sources support geographic feature reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_PATH = BASE_DIR / "results" / "reconstruction_readiness.json"


def main() -> None:
    production = pd.read_csv(RAW_DIR / "production.csv")
    climate = pd.read_csv(RAW_DIR / "climate.csv")
    water = pd.read_csv(RAW_DIR / "water_quality.csv")

    production_years = set(pd.to_numeric(production["Year"], errors="coerce").dropna().astype(int))
    climate_years = pd.to_datetime(climate["last_updated"], errors="coerce").dt.year.dropna().astype(int)
    water_years = pd.to_datetime(water["Date"], errors="coerce").dt.year.dropna().astype(int)
    report = {
        "official_sources_only": True,
        "synthetic_values_created": False,
        "water_country_key_available": "country" in water.columns,
        "climate_production_year_overlap": sorted(production_years & set(climate_years)),
        "water_production_year_overlap": sorted(production_years & set(water_years)),
        "water_geographic_reconstruction_ready": False,
        "reason_water_not_ready": [
            "water_quality.csv has no country key",
            "climate source has no overlap with production years",
        ],
        "genomic_production_mapping_key_available": False,
        "reason_genomic_not_ready": [
            "genomic.csv has Sample_ID but no production-compatible sample, farm, species, or country key",
        ],
        "approved_action": "Keep final_dataset.csv unchanged until official join keys and overlapping observations are supplied.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote reconstruction readiness report -> {OUTPUT_PATH}")
    print("Geographic water reconstruction: BLOCKED by official source coverage/key limitations")
    print("Genomic production integration: BLOCKED by missing official mapping key")


if __name__ == "__main__":
    main()