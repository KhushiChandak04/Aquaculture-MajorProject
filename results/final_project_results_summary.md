# Final Project Results Summary

Generated: 2026-03-24 23:20:24

## Artifact Status
- productivity_model.pkl: Available
- sustainability_model.pkl: Available
- feature_selector.pkl: Available

## Best Productivity Model Snapshot
- model: ExtraTrees
- accuracy: 0.9601
- precision_macro: 0.9607
- recall_macro: 0.9601
- f1_macro: 0.9603
- train_seconds: 1.3790
- infer_ms_per_1000: 40.3610
- model_size_mb: 103.8424
- peak_train_ram_mb: 138.2852

## All Productivity Models
| model | accuracy | precision_macro | recall_macro | f1_macro | train_seconds | infer_ms_per_1000 | model_size_mb | peak_train_ram_mb |
|---|---|---|---|---|---|---|---|---|
| ExtraTrees | 0.9601 | 0.9607 | 0.9601 | 0.9603 | 1.3790 | 40.3610 | 103.8424 | 138.2852 |
| RandomForest | 0.9515 | 0.9530 | 0.9516 | 0.9518 | 1.0634 | 32.2794 | 65.2930 | 81.5938 |
| LogisticRegression | 0.9057 | 0.9059 | 0.9056 | 0.9057 | 0.2751 | 5.1268 | 0.0141 | 1.4492 |
| XGBoost | 0.7920 | 0.8145 | 0.7919 | 0.7960 | 0.5430 | 5.0452 | 0.7373 | 98.1250 |
| DecisionTree | 0.4082 | 0.2773 | 0.4077 | 0.3264 | 0.0440 | 3.0877 | 0.0111 | 1.6016 |

## Best Sustainability Model Snapshot
- model: XGBoost_rebuild
- accuracy: 0.8298
- precision: 0.8474
- recall: 0.8297
- f1: 0.8326

## All Sustainability Models
| model | accuracy | precision | recall | f1 |
|---|---|---|---|---|
| XGBoost_rebuild | 0.8298 | 0.8474 | 0.8297 | 0.8326 |

## Best Genomic Model Snapshot
- Model: HistGB
- Training_Time_s: 2.1929
- Inference_Time_s: 0.0214
- Accuracy: 0.9464
- Precision_macro: 0.9464
- Recall_macro: 0.9464
- F1_Score_macro: 0.9464

## All Genomic Models
| Model | Training_Time_s | Inference_Time_s | Accuracy | Precision_macro | Recall_macro | F1_Score_macro |
|---|---|---|---|---|---|---|
| HistGB | 2.1929 | 0.0214 | 0.9464 | 0.9464 | 0.9464 | 0.9464 |

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
