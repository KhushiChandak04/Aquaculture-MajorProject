# Model Performance Comparison - Final Summary

## Best Models Across Tracks

This report is derived directly from the current benchmark CSVs in `results/`.

### Model Overview

| Track | Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms/1000) |
|---|---|---:|---:|---:|---:|---:|---:|
| Productivity | ExtraTrees | 0.9601 | 0.9607 | 0.9601 | 0.9603 | 1.3558 | 32.1874 |
| Sustainability | MLP | 0.9481 | 0.9481 | 0.9481 | 0.9479 | N/A | N/A |
| Genomic | HistGB | 0.9468 | 0.9469 | 0.9471 | 0.9470 | 2.0698 | 20.3966 |

### Key Performance Insights

**Best F1-Score:** Productivity - ExtraTrees (0.9603)
**Best Accuracy:** Productivity - ExtraTrees (0.9601)

### Visualization

A bar chart has been generated from the current benchmark outputs.

See: `results/model_comparison_visualization.png`

---

**Generated:** 2026-08-04 23:41:14
