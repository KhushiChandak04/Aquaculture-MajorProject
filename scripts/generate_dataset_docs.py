"""Generate reproducible dataset-quality documentation for the project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs" / "dataset_methodology"
OUTPUT_PATH = DOCS_DIR / "PROFESSOR_SUMMARY.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def pct(value: int, total: int) -> str:
    return f"{100.0 * value / total:.2f}%" if total else "0.00%"


def dataset_summary(name: str, raw_path: Path, processed_path: Path) -> dict[str, object]:
    raw = read_csv(raw_path)
    processed = read_csv(processed_path)
    raw_missing = int(raw.isna().sum().sum())
    return {
        "name": name,
        "raw_rows": len(raw),
        "processed_rows": len(processed),
        "raw_missing": raw_missing,
        "raw_missing_pct": pct(raw_missing, raw.size),
        "raw_duplicates": int(raw.duplicated().sum()),
        "processed_duplicates": int(processed.duplicated().sum()),
    }


def main() -> None:
    summaries = [
        dataset_summary("Production", RAW_DIR / "production.csv", PROCESSED_DIR / "production_clean.csv"),
        dataset_summary("Climate", RAW_DIR / "climate.csv", PROCESSED_DIR / "climate_clean.csv"),
        dataset_summary("Water quality", RAW_DIR / "water_quality.csv", PROCESSED_DIR / "water_quality_clean.csv"),
        dataset_summary("Genomic", RAW_DIR / "genomic.csv", PROCESSED_DIR / "genomic_clean.csv"),
    ]

    final_df = read_csv(PROCESSED_DIR / "final_dataset.csv")
    water_cols = [c for c in final_df.columns if c.lower().startswith("water_")]
    genomic_cols = [c for c in final_df.columns if c.lower().startswith("genomic_")]
    constant_cols = [c for c in final_df.columns if final_df[c].nunique(dropna=False) == 1]
    low_variance_cols = [
        (c, int(final_df[c].nunique(dropna=False)))
        for c in final_df.columns
        if final_df[c].nunique(dropna=False) <= 3
    ]
    water_by_year = {
        col: (
            int(final_df.groupby("year")[col].nunique(dropna=False).min()),
            int(final_df.groupby("year")[col].nunique(dropna=False).max()),
        )
        for col in water_cols
    }

    production = read_csv(PROCESSED_DIR / "production_clean.csv")
    climate = read_csv(PROCESSED_DIR / "climate_clean.csv")
    water = read_csv(PROCESSED_DIR / "water_quality_clean.csv")
    production_years = set(pd.to_numeric(production["year"], errors="coerce").dropna().astype(int))
    climate_years = set(pd.to_numeric(climate["year"], errors="coerce").dropna().astype(int))
    water_years = set(pd.to_numeric(water["year"], errors="coerce").dropna().astype(int))

    lines = [
        "# Dataset Summary for Professor Review",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This summary is generated from the official raw, processed, and final datasets. No synthetic data is created. It describes the data actually used by the pipeline and does not infer geographic or sample-level variation that is absent from the files.",
        "",
        "## Source and Processed Dataset Summary",
        "",
        "| Dataset | Raw rows | Processed rows | Raw missing cells | Raw missing % | Raw duplicates | Processed duplicates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['name']} | {item['raw_rows']} | {item['processed_rows']} | {item['raw_missing']} | {item['raw_missing_pct']} | {item['raw_duplicates']} | {item['processed_duplicates']} |"
        )

    lines.extend(
        [
            "",
            "## Join Coverage",
            "",
            f"- Production base: {len(production):,} rows, years {min(production_years)}-{max(production_years)}, {production['country'].nunique()} countries.",
            f"- Climate processed years: {min(climate_years)}-{max(climate_years)}; overlap with production years: {len(production_years & climate_years)} years.",
            f"- Water processed years: {min(water_years)}-{max(water_years)}; overlap with production years: {len(production_years & water_years)} years.",
            f"- Final dataset rows: {len(final_df):,}; rows dropped from the production base during final construction: {len(production) - len(final_df)}.",
            "",
            "## Final-Dataset Variation Findings",
            "",
            f"- Constant columns: {', '.join(constant_cols) if constant_cols else 'None'}.",
            f"- Genomic columns with one unique value: {sum(final_df[c].nunique(dropna=False) == 1 for c in genomic_cols)} of {len(genomic_cols)}.",
            "- Water unique-value counts within each year (minimum, maximum):",
        ]
    )
    for col, bounds in water_by_year.items():
        lines.append(f"  - `{col}`: {bounds[0]}, {bounds[1]}")

    lines.extend(["", "## Low-Variance Columns", "", "| Column | Unique values in final dataset |", "|---|---:|"])
    for col, count in low_variance_cols:
        lines.append(f"| `{col}` | {count} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Water is joined by year in `preprocessing/final_dataset_builder.py`; therefore every country receives the same water value for a given year.",
            "- The current climate data has no production-year overlap, so the climate features in the final dataset are constant after the left merge and subsequent imputation.",
            "- Genomic features are added as global means or a global mode, so they do not provide row-level genomic variation in the final dataset.",
            "- Missing values are resolved in the final builder, but imputation provenance is not currently retained as row-level flags.",
            "",
            "## Recommended Follow-up",
            "",
            "1. Replace the climate source or align its time coverage with the production years before claiming country-year climate effects.",
            "2. Add geographic water data or a documented geographic interpolation strategy if country-level water inference is required.",
            "3. Join genomic records through a defensible sample, species, or strain key; otherwise keep genomic outputs explicitly exploratory.",
            "4. Add missingness indicators and report observed-versus-imputed coverage in each model evaluation.",
            "",
        ]
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote professor summary -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()