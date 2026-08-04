# Final Project Results Summary

<p align="center"><strong>Consolidated Evidence Report for Productivity, Sustainability, and Genomic Tracks</strong></p>

<p align="center">
	<img src="https://img.shields.io/badge/Report_Type-Consolidated_Results-1D3557" alt="Report Type" />
	<img src="https://img.shields.io/badge/Scope-Tri_Track_Modeling-2A9D8F" alt="Scope" />
	<img src="https://img.shields.io/badge/Explainability-SHAP_Integrated-0C7BDC" alt="Explainability" />
</p>

Generated: 2026-08-04

## 1. Artifact Readiness Matrix
| Artifact | Path | Status |
|---|---|---|
| Productivity model | models/productivity_model.pkl | Available |
| Sustainability model | models/sustainability_model.pkl | Available |
| Genomic model | models/feature_selector.pkl | Available |
| Productivity metrics | results/productivity_metrics.csv | Available |
| Sustainability metrics | results/sustainability_metrics.csv | Available |
| Genomic metrics | results/janhavi_model_metrics.csv | Available |
| Genomic feature importance | results/genomic_feature_importance.csv | Available |
| Validation audit | results/validation_audit.md | Available |
| Novelty evidence | results/novelty_evidence.md | Available |

## 2. Best Model Snapshots

### 2.1 Productivity Track (Best: XGBoost)
| Metric | Value |
|---|---|
| Accuracy | 0.7920 |
| Precision (macro) | 0.8145 |
| Recall (macro) | 0.7919 |
| F1 (macro) | 0.7960 |
| Training Time (s) | 0.4949 |
| Inference per 1000 rows (ms) | 5.3200 |
| Model Size (MB) | 0.7373 |
| Peak Train RAM (MB) | 98.7461 |

### 2.2 Sustainability Track (Best: MLP)
| Metric | Value |
|---|---|
| Accuracy | 0.9481 |
| Precision (macro) | 0.9481 |
| Recall (macro) | 0.9481 |
| F1 (macro) | 0.9479 |

### 2.3 Genomic Track (Best: HistGB)
| Metric | Value |
|---|---|
| Accuracy | 0.9468 |
| Precision (macro) | 0.9469 |
| Recall (macro) | 0.9471 |
| F1 (macro) | 0.9470 |
| Training Time (s) | 2.0698 |
| Inference Time (s) | 0.0204 |

## 3. Full Benchmark Tables

### 3.1 Productivity Models
| model | accuracy | precision_macro | recall_macro | f1_macro | train_seconds | infer_ms_per_1000 | model_size_mb | peak_train_ram_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ExtraTrees | 0.9601 | 0.9607 | 0.9601 | 0.9603 | 1.3558 | 32.1874 | 103.8424 | 137.5898 |
| RandomForest | 0.9515 | 0.9530 | 0.9516 | 0.9518 | 0.9271 | 35.8677 | 65.2930 | 79.0312 |
| LogisticRegression | 0.9057 | 0.9059 | 0.9056 | 0.9057 | 0.1496 | 2.3860 | 0.0141 | 2.7539 |
| XGBoost | 0.7920 | 0.8145 | 0.7919 | 0.7960 | 0.4949 | 5.3200 | 0.7373 | 98.7461 |
| DecisionTree | 0.4082 | 0.2773 | 0.4077 | 0.3264 | 0.0338 | 3.0435 | 0.0111 | 1.7578 |

### 3.2 Sustainability Models
| model | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| MLP | 0.9481 | 0.9481 | 0.9481 | 0.9479 |
| 1D CNN | 0.3992 | 0.4012 | 0.3987 | 0.3731 |
| GRU | 0.3358 | 0.2585 | 0.3358 | 0.2073 |
| LSTM | 0.4014 | 0.4155 | 0.4011 | 0.3719 |
| CNN-LSTM | 0.4258 | 0.4159 | 0.4255 | 0.4094 |

### 3.3 Genomic Models
| Model | Training_Time_s | Inference_Time_s | Accuracy | Precision_macro | Recall_macro | F1_Score_macro |
|---|---:|---:|---:|---:|---:|---:|
| NaiveBayes | 0.0100 | 0.0009 | 0.3997 | 0.3967 | 0.4026 | 0.3915 |
| KNN_k3 | 0.0156 | 0.0104 | 0.4138 | 0.4128 | 0.4146 | 0.4072 |
| KNN_k5 | 0.0167 | 0.0099 | 0.4520 | 0.4477 | 0.4544 | 0.4433 |
| KNN_k7 | 0.0163 | 0.0120 | 0.4828 | 0.4805 | 0.4846 | 0.4784 |
| SVM_linear | 3.3069 | 0.2049 | 0.3859 | 0.3822 | 0.3873 | 0.3831 |
| SVM_rbf | 2.7075 | 0.9110 | 0.4168 | 0.4159 | 0.4198 | 0.4087 |
| AdaBoost | 0.5858 | 0.0119 | 0.4554 | 0.4559 | 0.4571 | 0.4498 |
| HistGB | 2.0698 | 0.0204 | 0.9468 | 0.9469 | 0.9471 | 0.9470 |

## 4. Top Genomic Feature Drivers
| Feature | Mean Importance | Std |
|---|---:|---:|
| country | 0.010071 | 0.000096 |
| water_Salinity (ppt) | 0.000000 | 0.000000 |
| water_SecchiDepth (m) | 0.000000 | 0.000000 |
| water_WaterTemp (C) | 0.000000 | 0.000000 |
| year | 0.000000 | 0.000000 |

## 5. Integrated Inference Capability Status
1. Real-time synchronized multi-model predictions: Active.
2. Track-wise SHAP explainability with fallback paths: Active.
3. Forecasting with ARIMA/trend fallback: Active.
4. Scenario simulation with confidence deltas: Active.
5. Policy recommendation layer with model-conditioned logic: Active.

## 6. Governance Notes
1. This summary is a consolidated operational-research view and should be read alongside:
	 - results/validation_audit.md
	 - results/xai_evidence_report.md
	 - results/novelty_evidence.md
2. Model outputs are predictive evidence and should be interpreted with domain and ecological validation.
