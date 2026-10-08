# Validation and Leakage Audit

Generated: 2026-10-08 13:40:24

## Dataset Integrity
- Samples: 11657
- Duplicate-row ratio: 0.0000

## Robustness Checks
### Random Stratified Split (80/20)
- Accuracy: 0.9485
- F1 macro: 0.9487

### Temporal Split (first 80% years train, last 20% years test)
- Accuracy: 0.8049
- F1 macro: 0.8007

## Drift Signal
- F1 gap (random - temporal): 0.1480

## Walk-Forward Validation
- Each test year is evaluated using only earlier years for training.
- Scaling and categorical encoding are fitted within each training fold.
- Production class thresholds are derived from the training fold and applied to that test year.
- Evaluated years: 2011-2018.
- Mean walk-forward accuracy: 0.9169.
- Mean walk-forward F1 macro: 0.9094.
- Detailed results: `results\temporal_walk_forward.csv`.

## Data Integrity Findings
- Duplicate country-year rows: 0.
- Constant final-dataset columns: None.
- No synthetic records are created by this audit.
- Water geography and genomic sample assignments are not reconstructed without official source keys.
