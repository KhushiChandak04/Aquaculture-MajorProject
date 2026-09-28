# Validation and Leakage Audit

Generated: 2026-09-28 14:49:55

## Dataset Integrity
- Samples: 11657
- Duplicate-row ratio: 0.0000

## Robustness Checks
### Random Stratified Split (80/20)
- Accuracy: 0.9520
- F1 macro: 0.9520

### Temporal Split (first 80% years train, last 20% years test)
- Accuracy: 0.8070
- F1 macro: 0.8026

## Drift Signal
- F1 gap (random - temporal): 0.1494

## Walk-Forward Validation
- Each test year is evaluated using only earlier years for training.
- Scaling and categorical encoding are fitted within each training fold.
- Production class thresholds are derived from the training fold and applied to that test year.
- Evaluated years: 2011-2018.
- Mean walk-forward accuracy: 0.9258.
- Mean walk-forward F1 macro: 0.9196.
- Detailed results: `results\temporal_walk_forward.csv`.

## Data Integrity Findings
- Duplicate country-year rows: 0.
- Constant final-dataset columns: temperature_celsius, precip_mm, humidity, genomic_GC_Content_global_mean, genomic_AT_Content_global_mean, genomic_Sequence_Length_global_mean, genomic_Num_A_global_mean, genomic_Num_T_global_mean, genomic_Num_C_global_mean, genomic_Num_G_global_mean, genomic_kmer_3_freq_global_mean, genomic_Mutation_Flag_global_mean, genomic_Disease_Risk_global_mode.
- No synthetic records are created by this audit.
- Water geography and genomic sample assignments are not reconstructed without official source keys.
