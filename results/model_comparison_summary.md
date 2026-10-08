# Model Performance Comparison - Final Summary

## Best Models Across Tracks

This report is derived directly from the current benchmark CSVs in `results/`.

### Model Overview

| Track | Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms/1000) |
|---|---|---:|---:|---:|---:|---:|---:|
| Productivity | LightGBM | 0.9335 | 0.9337 | 0.9335 | 0.9336 | 0.3892 | 11.8669 |
| Sustainability | XGBoost_rebuild | 0.8362 | 0.8525 | 0.8362 | 0.8389 | N/A | N/A |
| Genomic | HistGB | 0.8889 | 0.8909 | 0.8893 | 0.8899 | 0.6249 | 24.4621 |

### Key Performance Insights

**Best F1-Score:** Productivity - LightGBM (0.9336)
**Best Accuracy:** Productivity - LightGBM (0.9335)

### Visualization

A bar chart has been generated from the current benchmark outputs.

See: `results/model_comparison_visualization.png`

---

**Generated:** 2026-10-08 13:41:31
