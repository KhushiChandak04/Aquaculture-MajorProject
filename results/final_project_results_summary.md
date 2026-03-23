# Final Project Results Summary

Generated: 2026-03-23 21:33:49

## Artifact Status
- productivity_model.pkl: Available
- sustainability_model.pkl: Available
- feature_selector.pkl: Available

## Best Productivity Model Snapshot
- model: ExtraTrees
- accuracy: 0.9606
- precision_macro: 0.9606
- recall_macro: 0.9606
- f1_macro: 0.9605
- train_seconds: 0.6282
- infer_ms_per_1000: 85.6876
- model_size_mb: 40.2573
- peak_train_ram_mb: 44.5938

## Best Sustainability Model Snapshot
- model: MLP_rebuild
- accuracy: 0.9595
- precision: 0.9597
- recall: 0.9595
- f1: 0.9596

## Best Genomic Model Snapshot
- Model: HistGB
- Training_Time_s: 3.1939
- Inference_Time_s: 0.0148
- Accuracy: 0.9516
- Precision_macro: 0.9516
- Recall_macro: 0.9516
- F1_Score_macro: 0.9516

## Top Genomic Feature Drivers
- country: mean_importance=0.664679, std=0.005385
- year: mean_importance=0.079381, std=0.002635
- genomic_GC_Content_global_mean: mean_importance=0.000000, std=0.000000
- genomic_Num_A_global_mean: mean_importance=0.000000, std=0.000000
- genomic_Num_T_global_mean: mean_importance=0.000000, std=0.000000

## Inference and Explainability Status
- Real-time multi-model predictions: Active in Streamlit
- Local SHAP explanations: Integrated with safe fallback
- Forecasting: ARIMA with trend fallback
- Scenario simulation: Baseline vs perturbation analysis
- Policy recommendations: Hybrid model-assisted scoring plus rule layer
