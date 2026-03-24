# Final Project Results Summary

Generated: 2026-03-25 00:14:51

## Artifact Status
- productivity_model.pkl: Available
- sustainability_model.pkl: Available
- feature_selector.pkl: Available

## Best Productivity Model Snapshot
- model: XGBoost
- accuracy: 0.8945
- precision_macro: 0.9154
- recall_macro: 0.8387
- f1_macro: 0.8626
- train_seconds: 0.7384
- infer_ms_per_1000: 5.8325
- model_size_mb: 0.9775
- peak_train_ram_mb: 5.7574
- group: baseline_5_models
- recommended: no

## Best Sustainability Model Snapshot
- model: MLP
- accuracy: 0.9485
- precision: 0.9486
- recall: 0.9485
- f1: 0.9483

## All Sustainability Models
| model | accuracy | precision | recall | f1 |
|---|---|---|---|---|
| MLP | 0.9485 | 0.9486 | 0.9485 | 0.9483 |

## Best Genomic Model Snapshot
- Model: HistGB
- Training_Time_s: 2.7864
- Inference_Time_s: 0.0226
- Accuracy: 0.9464
- Precision_macro: 0.9464
- Recall_macro: 0.9464
- F1_Score_macro: 0.9464

## All Genomic Models
| Model | Training_Time_s | Inference_Time_s | Accuracy | Precision_macro | Recall_macro | F1_Score_macro |
|---|---|---|---|---|---|---|
| HistGB | 2.7864 | 0.0226 | 0.9464 | 0.9464 | 0.9464 | 0.9464 |

## Top Genomic Feature Drivers
- country: mean_importance=0.647512, std=0.003921
- year: mean_importance=0.272763, std=0.003048
- genomic_GC_Content_global_mean: mean_importance=0.000000, std=0.000000
- genomic_Num_A_global_mean: mean_importance=0.000000, std=0.000000
- genomic_Num_T_global_mean: mean_importance=0.000000, std=0.000000

## Inference and Explainability Status
- Real-time multi-model predictions: Active in Streamlit
- Local SHAP explanations: Integrated with safe fallback
- Forecasting: ARIMA with trend fallback
- Scenario simulation: Baseline vs perturbation analysis
- Policy recommendations: Hybrid model-assisted scoring plus rule layer
