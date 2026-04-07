# Final Project Results Summary

Generated: 2026-04-07 23:51:06

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

## All Productivity Models
| model | accuracy | precision_macro | recall_macro | f1_macro | train_seconds | infer_ms_per_1000 | model_size_mb | peak_train_ram_mb | group | recommended |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.8945 | 0.9154 | 0.8387 | 0.8626 | 0.7384 | 5.8325 | 0.9775 | 5.7574 | baseline_5_models | no |
| DecisionTree | 0.8752 | 0.8976 | 0.8192 | 0.8446 | 0.0657 | 2.1987 | 0.0145 | 5.7581 | baseline_5_models | no |
| RandomForest | 0.8606 | 0.8477 | 0.8164 | 0.8290 | 1.7178 | 19.9018 | 0.6335 | 9.3015 | baseline_5_models | no |
| LogReg_Tuned_C0.30 | 0.7380 | 0.7246 | 0.7342 | 0.7240 | 0.1663 | 1.6910 | 0.0126 | 2.8502 | logreg_tuning | yes |
| LogisticRegression_XAI_Improved | 0.7389 | 0.7225 | 0.7337 | 0.7234 | 0.2497 | 5.0989 | 0.0126 | 2.8480 | weak_model_branch | no |
| LogReg_Tuned_C0.50 | 0.7389 | 0.7225 | 0.7337 | 0.7234 | 0.1883 | 1.5200 | 0.0126 | 2.8394 | logreg_tuning | no |
| LogisticRegression | 0.7363 | 0.7149 | 0.7291 | 0.7183 | 0.3163 | 2.0623 | 0.0142 | 5.8099 | baseline_5_models | no |
| LogReg_Tuned_C0.80 | 0.7324 | 0.7140 | 0.7266 | 0.7157 | 0.1850 | 1.7074 | 0.0126 | 2.8382 | logreg_tuning | no |
| LogReg_Tuned_C1.20 | 0.7316 | 0.7121 | 0.7251 | 0.7141 | 0.2072 | 1.5921 | 0.0126 | 2.8390 | logreg_tuning | no |

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
- Training_Time_s: 3.7954
- Inference_Time_s: 0.0286
- Accuracy: 0.9464
- Precision_macro: 0.9464
- Recall_macro: 0.9464
- F1_Score_macro: 0.9464

## All Genomic Models
| Model | Training_Time_s | Inference_Time_s | Accuracy | Precision_macro | Recall_macro | F1_Score_macro |
|---|---|---|---|---|---|---|
| HistGB | 3.7954 | 0.0286 | 0.9464 | 0.9464 | 0.9464 | 0.9464 |

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
