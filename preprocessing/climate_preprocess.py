from pathlib import Path

import pandas as pd


def _extract_year(df: pd.DataFrame) -> pd.Series:
    # Try common date/time columns present across climate exports.
    candidate_cols = ["date", "Date", "last_updated", "last_updated_date", "timestamp"]
    for col in candidate_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed.dt.year

    if "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
        if year.notna().any():
            return year

    raise ValueError(
        "Could not derive year. Provide one of: date/Date/last_updated/timestamp/year."
    )


def preprocess_climate(input_path, output_path):
    df = pd.read_csv(input_path)
    df["year"] = _extract_year(df)

    # Standardize column names.
    df.columns = [c.lower() for c in df.columns]

    climate_df = pd.DataFrame()
    climate_df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "country" in df.columns:
        climate_df["country"] = df["country"].astype(str)

    # Use canonical climate feature names while accepting common aliases.
    temp_col = next((c for c in ["temperature_celsius", "temperature", "temp"] if c in df.columns), None)
    precip_col = next((c for c in ["precip_mm", "precipitation", "rainfall"] if c in df.columns), None)
    humid_col = next((c for c in ["humidity"] if c in df.columns), None)

    if temp_col is not None:
        climate_df["temperature_celsius"] = pd.to_numeric(df[temp_col], errors="coerce")
    if precip_col is not None:
        climate_df["precip_mm"] = pd.to_numeric(df[precip_col], errors="coerce")
    if humid_col is not None:
        climate_df["humidity"] = pd.to_numeric(df[humid_col], errors="coerce")

    feature_cols = [c for c in climate_df.columns if c not in ["year", "country"]]
    if not feature_cols:
        raise ValueError("No supported climate feature columns found after cleaning.")

    climate_df = climate_df.dropna(subset=["year"])
    climate_df["year"] = climate_df["year"].astype("Int64")

    # Keep row-level records to preserve full dataset size for modeling.
    climate_df = climate_df.dropna(subset=feature_cols, how="all")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    climate_df.to_csv(output_path, index=False)
    print(f"Climate dataset cleaned -> {output_path}")
