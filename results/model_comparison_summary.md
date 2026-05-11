# Model Performance Comparison - Final Summary

## Top 3 Models Across All Tracks

This section presents a comprehensive comparison of the three best-performing models evaluated across the Aquaculture Intelligence Platform.

### Model Overview

| Sr. No | Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|--------|-------|----------|-----------|--------|----------|------------------|
| 1 | XGBoost | 0.9684 | 0.9648 | 0.9639 | 0.9662 | 1.5402 |
| 2 | MLP Classifier | 0.9721 | 0.9694 | 0.9688 | 0.9702 | N/A |
| 3 | HistGradientBoosting | 0.9612 | 0.9585 | 0.9598 | 0.9597 | 2.5815 |

### Key Performance Insights

#### Ranking by Metric

**1. Accuracy:**
- 1st: MLP Classifier (0.9721)
- 2nd: XGBoost (0.9684)
- 3rd: HistGradientBoosting (0.9612)

**2. Precision:**
- 1st: MLP Classifier (0.9694)
- 2nd: XGBoost (0.9648)
- 3rd: HistGradientBoosting (0.9585)

**3. Recall:**
- 1st: MLP Classifier (0.9688)
- 2nd: XGBoost (0.9639)
- 3rd: HistGradientBoosting (0.9598)

**4. F1-Score:**
- 1st: MLP Classifier (0.9702)
- 2nd: XGBoost (0.9662)
- 3rd: HistGradientBoosting (0.9597)

**5. Training Time:**
- Fastest: XGBoost (1.5402 seconds)
- 2nd: HistGradientBoosting (2.5815 seconds)
- 3rd: MLP Classifier (N/A - varies by framework)

### Statistical Summary

**Accuracy:**
- Mean: 0.9672
- Std Dev: 0.0055
- Range: 0.9612 - 0.9721

**F1-Score:**
- Mean: 0.9654
- Std Dev: 0.0053
- Range: 0.9597 - 0.9702

**Precision:**
- Mean: 0.9642
- Std Dev: 0.0055
- Range: 0.9585 - 0.9694

**Recall:**
- Mean: 0.9642
- Std Dev: 0.0045
- Range: 0.9598 - 0.9688

### Model Recommendations

1. **MLP Classifier** — Best overall performance across all metrics (Accuracy: 0.9721, F1: 0.9702)
   - Recommended for: High-stakes predictions requiring maximum accuracy
   - Trade-off: Training time not specified (typically moderate for neural networks)

2. **XGBoost** — Excellent balance of performance and speed
   - Accuracy: 0.9684, F1: 0.9662
   - Training Time: 1.5402 seconds (fastest)
   - Recommended for: Production deployments requiring fast inference

3. **HistGradientBoosting** — Solid alternative with gradient boosting advantages
   - Accuracy: 0.9612, F1: 0.9597
   - Training Time: 2.5815 seconds
   - Recommended for: Interpretability and feature importance analysis

### Performance Metrics Explanation

- **Accuracy:** Overall correctness (correct predictions / total predictions)
- **Precision:** True positives / (true positives + false positives) — minimize false alarms
- **Recall:** True positives / (true positives + false negatives) — minimize missed cases
- **F1-Score:** Harmonic mean of Precision and Recall — balanced metric
- **Training Time:** Time to fit the model on training data

### Visualization

A comprehensive bar chart has been generated showing:
- Multi-metric performance comparison
- Individual metric rankings
- Training time efficiency comparison

See: `results/model_comparison_visualization.png`

---

**Generated:** 2026-05-11 15:47:16
