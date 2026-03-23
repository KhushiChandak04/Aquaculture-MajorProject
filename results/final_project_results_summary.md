# Final Project Results Summary

Generated: 2026-03-23 19:14:44

## Artifact Status
- productivity_model.pkl: Available
- sustainability_model.pkl: Available
- feature_selector.pkl: Available

## Best Productivity Model Snapshot
- model: RandomForest
- accuracy: 0.8481
- precision_macro: 0.8970
- recall_macro: 0.7797
- f1_macro: 0.8170
- train_seconds: 7.6653
- infer_ms_per_1000: 234.6427
- model_size_mb: 0.3476
- peak_train_ram_mb: 3.1249

## Best Sustainability Model Snapshot
- model: MLP
- accuracy: 1.0000
- precision: 1.0000
- recall: 1.0000
- f1: 1.0000

## Best Genomic Model Snapshot
- Model: HistGB
- Training_Time_s: 2.4543
- Inference_Time_s: 0.0110
- Accuracy: 0.9336
- Precision_macro: 0.9318
- Recall_macro: 0.9318
- F1_Score_macro: 0.9317

## Top Genomic Feature Drivers
- precip_mm: mean_importance=0.000000, std=0.000000
- temperature_celsius: mean_importance=0.000000, std=0.000000
- genomic_Num_G_global_mean: mean_importance=0.000000, std=0.000000
- genomic_Sequence_Length_global_mean: mean_importance=0.000000, std=0.000000
- country: mean_importance=-0.003060, std=0.001085

## Inference and Explainability Status
- Real-time multi-model predictions: Active in Streamlit
- Local SHAP explanations: Integrated with safe fallback
- Forecasting: ARIMA with trend fallback
- Scenario simulation: Baseline vs perturbation analysis
- Policy recommendations: Rule-based integrated layer
