# Final Project Results Summary

Generated: 2026-03-24 23:13:54

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

## All Productivity Models
| model | accuracy | precision_macro | recall_macro | f1_macro | train_seconds | infer_ms_per_1000 | model_size_mb | peak_train_ram_mb |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.8945 | 0.9154 | 0.8387 | 0.8626 | 0.7384 | 5.8325 | 0.9775 | 5.7574 |
| DecisionTree | 0.8752 | 0.8976 | 0.8192 | 0.8446 | 0.0657 | 2.1987 | 0.0145 | 5.7581 |
| ExtraTrees | 0.8611 | 0.8953 | 0.8022 | 0.8323 | 1.1204 | 24.9738 | 3.1018 | 8.4211 |
| RandomForest | 0.8606 | 0.8477 | 0.8164 | 0.8290 | 1.7178 | 19.9018 | 0.6335 | 9.3015 |
| LogisticRegression | 0.7363 | 0.7149 | 0.7291 | 0.7183 | 0.3163 | 2.0623 | 0.0142 | 5.8099 |

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
- Training_Time_s: 2.3102
- Inference_Time_s: 0.0260
- Accuracy: 0.9468
- Precision_macro: 0.9469
- Recall_macro: 0.9471
- F1_Score_macro: 0.9470

## All Genomic Models
| Model | Training_Time_s | Inference_Time_s | Accuracy | Precision_macro | Recall_macro | F1_Score_macro |
|---|---|---|---|---|---|---|
| HistGB | 2.3102 | 0.0260 | 0.9468 | 0.9469 | 0.9471 | 0.9470 |
| KNN_k7 | 0.0130 | 0.0126 | 0.4828 | 0.4805 | 0.4846 | 0.4784 |
| AdaBoost | 0.4734 | 0.0100 | 0.4554 | 0.4559 | 0.4571 | 0.4498 |
| KNN_k5 | 0.0136 | 0.0124 | 0.4520 | 0.4477 | 0.4544 | 0.4433 |
| SVM_rbf | 3.1564 | 1.2969 | 0.4168 | 0.4159 | 0.4198 | 0.4087 |
| KNN_k3 | 0.0159 | 0.0179 | 0.4138 | 0.4128 | 0.4146 | 0.4072 |
| NaiveBayes | 0.0073 | 0.0010 | 0.3997 | 0.3967 | 0.4026 | 0.3915 |
| SVM_linear | 3.9678 | 0.2327 | 0.3859 | 0.3822 | 0.3873 | 0.3831 |

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
