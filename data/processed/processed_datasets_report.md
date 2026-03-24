# Processed Datasets Report

Generated from current contents of `data/raw` and `data/processed`.

## 1) Processed Dataset Metrics

| File | Rows | Columns | Missing Cells | Duplicate Rows | Numeric Cols | Categorical Cols | Year Range | Unique Countries |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `production_clean.csv` | 11657 | 3 | 0 | 0 | 2 | 1 | 1960-2018 | 242 |
| `water_quality_clean.csv` | 2366 | 7 | 0 | 23 | 7 | 0 | 1989-2019 | - |
| `climate_clean.csv` | 130588 | 5 | 0 | 12959 | 4 | 1 | 2024-2026 | 211 |
| `genomic_clean.csv` | 3000 | 10 | 0 | 0 | 9 | 1 | - | - |
| `final_dataset.csv` | 11657 | 22 | 0 | 0 | 20 | 2 | 1960-2018 | 242 |

Additional production stats in final dataset:
- production min: 0.0
- production max: 106451316.491233
- production mean: 1702216.4944174765

## 2) Raw vs Processed Shape Changes

| Dataset | Raw Shape | Processed Shape | Row Delta | Column Delta |
|---|---|---|---:|---:|
| production | 11657 x 4 | 11657 x 3 | 0 | -1 |
| water_quality | 2371 x 8 | 2366 x 7 | -5 | -1 |
| climate | 130588 x 41 | 130588 x 5 | 0 | -36 |
| genomic | 3000 x 13 | 3000 x 10 | 0 | -3 |

## 3) What Preprocessing Did (Per Script)

### production_preprocess.py
- Renamed columns:
  - `Entity` -> `country`
  - `Year` -> `year`
  - `Aquaculture production (metric tons)` -> `production`
- Kept only required columns: `country`, `year`, `production`.
- Coerced `year` and `production` to numeric with invalid values as NaN.
- Removed rows where any of required columns were missing (`dropna` on required columns).

Removed in practice:
- 1 column removed (from 4 to 3).
- 0 rows removed in current data snapshot.

### water_preprocess.py
- Tried deriving `year` from `Date` or `date`.
- Kept only numeric columns for modeling (`select_dtypes` numeric).
- Dropped columns with more than 30% missing values (`thresh = 70% non-null`).
- Coerced numeric data and filled remaining missing values with column means.
- Dropped all-NaN columns after imputation attempt.
- Added `year` if derivable and removed rows where `year` is missing.
- Standard-scaled feature columns using `StandardScaler`.

Removed in practice:
- 1 column removed (from 8 to 7).
- 5 rows removed (from 2371 to 2366), mainly from invalid or missing date/year extraction.

### climate_preprocess.py
- Derived `year` from one of: `date`, `Date`, `last_updated`, `last_updated_date`, `timestamp`, or fallback `year`.
- Lowercased all column names.
- Built a canonical climate frame with:
  - `year`
  - optional `country`
  - `temperature_celsius` from aliases (`temperature_celsius`, `temperature`, `temp`)
  - `precip_mm` from aliases (`precip_mm`, `precipitation`, `rainfall`)
  - `humidity`
- Dropped rows with missing `year`.
- Dropped rows where all climate feature columns are missing.

Removed in practice:
- 36 columns removed (from 41 to 5).
- 0 rows removed in current data snapshot.

### genomic_preprocess.py
- Assumed last column is target label.
- Split features/target and kept numeric features only (non-numeric sequence/id-like columns excluded).
- Dropped all-NaN feature columns.
- Mean-imputed remaining numeric missing values.
- Standard-scaled numeric features with `StandardScaler`.
- Reattached target column.

Removed in practice:
- 3 columns removed (from 13 to 10).
- 0 rows removed in current data snapshot.

### final_dataset_builder.py
- Started from `production_clean.csv` as base table.
- Merged climate:
  - by (`country`, `year`) if both present,
  - else by `year`,
  - else added global mean/mode constant columns.
- Merged water:
  - by `year` when present (prefixed as `water_*`),
  - else added global mean/mode constants.
- Added genomic global aggregate columns (`genomic_*_global_mean` or `_global_mode`).
- Removed rows missing core keys/target: `country`, `year`, `production`.
- Imputed numeric NaN with column mean, then fallback 0.
- Imputed object NaN with mode, else `Unknown`.

Removed in practice:
- No row drop from production base in current snapshot (final rows = 11657).
- Missing values in final merged table fully imputed (0 missing cells remain).

## 4) Notes
- Duplicate rows still exist in:
  - `climate_clean.csv` (12959)
  - `water_quality_clean.csv` (23)
- This is not currently blocking because final merge + aggregation and imputations still produce a complete `final_dataset.csv` with no missing cells.
