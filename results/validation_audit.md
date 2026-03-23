# Validation and Leakage Audit

Generated: 2026-03-23 21:33:49

## Dataset Integrity
- Samples: 4445
- Duplicate-row ratio: 0.0000

## Robustness Checks
### Random Stratified Split (80/20)
- Accuracy: 0.9550
- F1 macro: 0.9552

### Temporal Split (first 80% years train, last 20% years test)
- Accuracy: 0.9055
- F1 macro: 0.9043

## Drift Signal
- F1 gap (random - temporal): 0.0509
