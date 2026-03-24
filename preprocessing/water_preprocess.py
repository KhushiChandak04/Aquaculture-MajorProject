from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_water(input_path, output_path):
    df = pd.read_csv(input_path)

    year_series = None
    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        if parsed.notna().any():
            year_series = parsed.dt.year
    elif "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        if parsed.notna().any():
            year_series = parsed.dt.year

    # Keep only numeric columns for modeling.
    numeric_df = df.select_dtypes(include=["float64", "int64", "float32", "int32"])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in water quality dataset.")

    # Drop columns with too many missing values.
    threshold = int(len(numeric_df) * 0.7)
    numeric_df = numeric_df.dropna(axis=1, thresh=threshold)

    # Fill remaining missing values with column means.
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))

    # Drop any columns that still cannot be imputed (all-NaN columns).
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.empty:
        raise ValueError("All water quality numeric columns became empty after cleaning.")

    if year_series is not None:
        numeric_df = numeric_df.copy()
        numeric_df["year"] = pd.to_numeric(year_series, errors="coerce")
        numeric_df = numeric_df.dropna(subset=["year"])
        numeric_df["year"] = numeric_df["year"].astype("Int64")

    feature_cols = [c for c in numeric_df.columns if c != "year"]
    if not feature_cols:
        raise ValueError("No water quality feature columns available for scaling.")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric_df[feature_cols])
    df_scaled = pd.DataFrame(scaled_values, columns=feature_cols)
    if "year" in numeric_df.columns:
        df_scaled.insert(0, "year", numeric_df["year"].reset_index(drop=True))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"Water quality dataset cleaned -> {output_path}")
