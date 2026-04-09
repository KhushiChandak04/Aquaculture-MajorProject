# Validation and Leakage Audit

Generated: 2026-04-09 23:55:01

## Dataset Integrity
- Samples: 11657
- Duplicate-row ratio: 0.0000

## Robustness Checks
### Random Stratified Split (80/20)
- Accuracy: 0.9520
- F1 macro: 0.9520

### Temporal Split (first 80% years train, last 20% years test)
- Accuracy: 0.8079
- F1 macro: 0.8034

## Drift Signal
- F1 gap (random - temporal): 0.1486
