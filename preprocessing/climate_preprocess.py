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

    # Standardize column names and keep key climate signals if available.
    df.columns = [c.lower() for c in df.columns]
    possible_cols = {
        "temperature",
        "temp",
        "temperature_celsius",
        "precipitation",
        "rainfall",
        "precip_mm",
        "humidity",
        "year",
    }
    selected_cols = [c for c in df.columns if c in possible_cols]

    if "year" not in selected_cols:
        selected_cols.append("year")

    # Ensure yearly aggregation has at least one feature besides year.
    feature_cols = [c for c in selected_cols if c != "year"]
    if not feature_cols:
        raise ValueError("No supported climate feature columns found for aggregation.")

    climate_df = df[selected_cols].copy()
    for col in feature_cols:
        climate_df[col] = pd.to_numeric(climate_df[col], errors="coerce")

    climate_df = climate_df.dropna(subset=["year"]) 
    climate_df = climate_df.groupby("year", as_index=False).mean(numeric_only=True)
    climate_df = climate_df.dropna()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    climate_df.to_csv(output_path, index=False)
    print(f"Climate dataset cleaned -> {output_path}")
