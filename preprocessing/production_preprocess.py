from pathlib import Path

import pandas as pd


def preprocess_production(input_path, output_path):
    df = pd.read_csv(input_path)

    # Rename columns to standardized names.
    df = df.rename(
        columns={
            "Entity": "country",
            "Year": "year",
            "Aquaculture production (metric tons)": "production",
        }
    )

    required_cols = ["country", "year", "production"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required production columns: {missing_cols}")

    df = df[required_cols]

    # Safe numeric coercion before filtering.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["production"] = pd.to_numeric(df["production"], errors="coerce")

    # Filter recent data and remove nulls.
    df = df[df["year"] >= 2000]
    df = df.dropna(subset=required_cols)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Production dataset cleaned -> {output_path}")
