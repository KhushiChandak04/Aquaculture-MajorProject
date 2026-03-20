from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_genomic(input_path, output_path):
    df = pd.read_csv(input_path)
    if df.shape[1] < 2:
        raise ValueError("Genomic dataset requires at least one feature column and one target column.")

    # Assume last column is target label.
    target_col = df.columns[-1]

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Keep numeric features only; sequence/string identifiers are excluded.
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    X_numeric = X_numeric.dropna(axis=1, how="all")
    if X_numeric.empty:
        raise ValueError("No numeric genomic features available after cleaning.")

    X_numeric = X_numeric.fillna(X_numeric.mean(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)

    # Preserve label column for supervised modeling.
    df_clean = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Genomic dataset cleaned -> {output_path}")
