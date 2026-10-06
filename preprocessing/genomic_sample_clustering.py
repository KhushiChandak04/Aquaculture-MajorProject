"""Cluster official genomic samples without inventing farm or strain mappings.

The output is a standalone sample-level analysis. It is intentionally not merged
into final_dataset.csv because the official data has no production-compatible
sample, farm, species, or country key.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "genomic.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "genomic_sample_clusters.csv"
MODEL_PATH = BASE_DIR / "models" / "genomic_sample_clusterer.pkl"


def main(n_clusters: int = 5) -> None:
    df = pd.read_csv(RAW_PATH)
    if "Sample_ID" not in df.columns:
        raise ValueError("Official genomic source requires Sample_ID for standalone clustering")

    excluded = {"Sample_ID", "Sequence", "Class_Label", "Disease_Risk"}
    feature_columns = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(feature_columns) < 2:
        raise ValueError("At least two official numeric genomic features are required")

    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.mean()).fillna(0.0)
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=42, n_init=20)),
        ]
    )
    cluster_ids = pipeline.fit_predict(features)

    output = df[["Sample_ID"]].copy()
    output["cluster_id"] = cluster_ids.astype(int)
    output["cluster_assignment_scope"] = "official genomic sample only"
    output["production_join_key_available"] = False
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_columns}, MODEL_PATH)

    print(f"Wrote official genomic sample clusters -> {OUTPUT_PATH}")
    print(f"Saved clusterer -> {MODEL_PATH}")
    print(f"Cluster counts: {output['cluster_id'].value_counts().sort_index().to_dict()}")
    print("No cluster assignments were merged into final_dataset.csv.")


if __name__ == "__main__":
    main()