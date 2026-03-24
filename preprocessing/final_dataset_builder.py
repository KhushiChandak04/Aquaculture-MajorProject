from pathlib import Path

import pandas as pd


def _add_constant_columns(base_df: pd.DataFrame, source_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    numeric_cols = source_df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        means = source_df[numeric_cols].mean(numeric_only=True)
        for col, val in means.items():
            base_df[f"{prefix}_{col}_global_mean"] = val

    non_numeric_cols = source_df.select_dtypes(exclude=["number"]).columns.tolist()
    for col in non_numeric_cols:
        if source_df[col].dropna().empty:
            continue
        mode_value = source_df[col].mode(dropna=True)
        if not mode_value.empty:
            base_df[f"{prefix}_{col}_global_mode"] = mode_value.iloc[0]

    return base_df


def _aggregate_for_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in keys]
    object_cols = [c for c in df.select_dtypes(exclude=["number"]).columns if c not in keys]

    agg_map: dict[str, str] = {}
    for col in numeric_cols:
        agg_map[col] = "mean"
    for col in object_cols:
        agg_map[col] = "first"

    if not agg_map:
        return df[keys].drop_duplicates().reset_index(drop=True)

    return df.groupby(keys, as_index=False).agg(agg_map)


def build_final_dataset(processed_dir):
    processed_dir = Path(processed_dir)

    production_path = processed_dir / "production_clean.csv"
    climate_path = processed_dir / "climate_clean.csv"
    water_path = processed_dir / "water_quality_clean.csv"
    genomic_path = processed_dir / "genomic_clean.csv"
    final_path = processed_dir / "final_dataset.csv"

    for required in [production_path, climate_path, water_path, genomic_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required processed file not found: {required}")

    final_df = pd.read_csv(production_path)
    if "year" not in final_df.columns:
        raise ValueError("production_clean.csv must contain year column.")

    climate_df = pd.read_csv(climate_path)
    climate_df.columns = [c.lower() for c in climate_df.columns]
    if "year" in climate_df.columns and "country" in climate_df.columns:
        climate_df["country"] = climate_df["country"].astype(str)
        climate_df = _aggregate_for_keys(climate_df, ["country", "year"])
        final_df = final_df.merge(climate_df, on=["country", "year"], how="left")
    elif "year" in climate_df.columns:
        climate_df = _aggregate_for_keys(climate_df, ["year"])
        final_df = final_df.merge(climate_df, on="year", how="left")
    else:
        final_df = _add_constant_columns(final_df, climate_df, prefix="climate")

    water_df = pd.read_csv(water_path)
    if "year" in water_df.columns:
        water_df = _aggregate_for_keys(water_df, ["year"])
        water_renamed = water_df.rename(
            columns={c: f"water_{c}" for c in water_df.columns if c != "year"}
        )
        final_df = final_df.merge(water_renamed, on="year", how="left")
    else:
        final_df = _add_constant_columns(final_df, water_df, prefix="water")

    genomic_df = pd.read_csv(genomic_path)
    final_df = _add_constant_columns(final_df, genomic_df, prefix="genomic")

    final_df = final_df.dropna(subset=["country", "year", "production"])

    numeric_cols = final_df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        final_df[numeric_cols] = final_df[numeric_cols].fillna(
            final_df[numeric_cols].mean(numeric_only=True)
        )
        final_df[numeric_cols] = final_df[numeric_cols].fillna(0)

    object_cols = final_df.select_dtypes(exclude=["number"]).columns.tolist()
    for col in object_cols:
        if final_df[col].isna().any() and not final_df[col].dropna().empty:
            final_df[col] = final_df[col].fillna(final_df[col].mode(dropna=True).iloc[0])
        elif final_df[col].isna().any():
            final_df[col] = final_df[col].fillna("Unknown")

    final_df["year"] = pd.to_numeric(final_df["year"], errors="coerce").astype("Int64")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(final_path, index=False)
    print(f"Final dataset created -> {final_path}")
