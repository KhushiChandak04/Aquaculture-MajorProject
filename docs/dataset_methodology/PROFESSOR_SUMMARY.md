# Dataset Summary for Professor Review

Generated: 2026-09-28 14:34:51

This summary is generated from the official raw, processed, and final datasets. No synthetic data is created. It describes the data actually used by the pipeline and does not infer geographic or sample-level variation that is absent from the files.

## Source and Processed Dataset Summary

| Dataset | Raw rows | Processed rows | Raw missing cells | Raw missing % | Raw duplicates | Processed duplicates |
|---|---:|---:|---:|---:|---:|---:|
| Production | 11657 | 11657 | 2773 | 5.95% | 0 | 0 |
| Climate | 130588 | 130588 | 0 | 0.00% | 0 | 12959 |
| Water quality | 2371 | 2366 | 1346 | 7.10% | 10 | 23 |
| Genomic | 3000 | 3000 | 0 | 0.00% | 0 | 0 |

## Join Coverage

- Production base: 11,657 rows, years 1960-2018, 242 countries.
- Climate processed years: 2024-2026; overlap with production years: 0 years.
- Water processed years: 1989-2019; overlap with production years: 30 years.
- Final dataset rows: 11,657; rows dropped from the production base during final construction: 0.

## Final-Dataset Variation Findings

- Constant columns: temperature_celsius, precip_mm, humidity, genomic_GC_Content_global_mean, genomic_AT_Content_global_mean, genomic_Sequence_Length_global_mean, genomic_Num_A_global_mean, genomic_Num_T_global_mean, genomic_Num_C_global_mean, genomic_Num_G_global_mean, genomic_kmer_3_freq_global_mean, genomic_Mutation_Flag_global_mean, genomic_Disease_Risk_global_mode.
- Genomic columns with one unique value: 10 of 10.
- Water unique-value counts within each year (minimum, maximum):
  - `water_Salinity (ppt)`: 1, 1
  - `water_pH`: 1, 1
  - `water_SecchiDepth (m)`: 1, 1
  - `water_WaterDepth (m)`: 1, 1
  - `water_WaterTemp (C)`: 1, 1
  - `water_AirTemp (C)`: 1, 1

## Low-Variance Columns

| Column | Unique values in final dataset |
|---|---:|
| `temperature_celsius` | 1 |
| `precip_mm` | 1 |
| `humidity` | 1 |
| `genomic_GC_Content_global_mean` | 1 |
| `genomic_AT_Content_global_mean` | 1 |
| `genomic_Sequence_Length_global_mean` | 1 |
| `genomic_Num_A_global_mean` | 1 |
| `genomic_Num_T_global_mean` | 1 |
| `genomic_Num_C_global_mean` | 1 |
| `genomic_Num_G_global_mean` | 1 |
| `genomic_kmer_3_freq_global_mean` | 1 |
| `genomic_Mutation_Flag_global_mean` | 1 |
| `genomic_Disease_Risk_global_mode` | 1 |

## Interpretation

- Water is joined by year in `preprocessing/final_dataset_builder.py`; therefore every country receives the same water value for a given year.
- The current climate data has no production-year overlap, so the climate features in the final dataset are constant after the left merge and subsequent imputation.
- Genomic features are added as global means or a global mode, so they do not provide row-level genomic variation in the final dataset.
- Missing values are resolved in the final builder, but imputation provenance is not currently retained as row-level flags.

## Recommended Follow-up

1. Replace the climate source or align its time coverage with the production years before claiming country-year climate effects.
2. Add geographic water data or a documented geographic interpolation strategy if country-level water inference is required.
3. Join genomic records through a defensible sample, species, or strain key; otherwise keep genomic outputs explicitly exploratory.
4. Add missingness indicators and report observed-versus-imputed coverage in each model evaluation.
