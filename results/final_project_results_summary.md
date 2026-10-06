# Final Project Results Summary

<p align="center"><strong>Consolidated Evidence Report for Productivity, Sustainability, and Genomic Tracks</strong></p>

<p align="center">
	<img src="https://img.shields.io/badge/Report_Type-Consolidated_Results-1D3557" alt="Report Type" />
	<img src="https://img.shields.io/badge/Scope-Tri_Track_Modeling-2A9D8F" alt="Scope" />
	<img src="https://img.shields.io/badge/Explainability-SHAP_Integrated-0C7BDC" alt="Explainability" />
</p>

Generated: 2026-10-07

## Feature Cleanup Interpretation

The current results use the rebuilt official 9-column dataset: country, year, production, and six time-bucket water-quality features. Climate columns were excluded because the source covers 2024-2026 while production covers 1960-2018. Genomic global aggregates were excluded because no production-compatible join key exists. Country is label-encoded, year is numeric, decade is derived from year, and the production target uses log1p quantile construction. Earlier optimistic metrics that used zero-variance climate/genomic columns are not valid predictive evidence. The current PSG accuracy is 0.7407 and macro-F1 is 0.7405; these are the honest augmented-feature results.

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

## 2. Best Model Snapshots

### 2.1 Productivity Track (Best: XGBoost)
| Metric | Value |
|---|---|
| Accuracy | 0.8606 |
| Precision (macro) | 0.8606 |
| Recall (macro) | 0.8606 |
| F1 (macro) | 0.8606 |
| Training Time (s) | 0.6904 |
| Inference per 1000 rows (ms) | 4.9575 |
| Model Size (MB) | 1.4641 |
| Peak Train RAM (MB) | 91.6836 |

### 2.2 Sustainability Track (Best: XGBoost_rebuild)
| Metric | Value |
|---|---|
| Accuracy | 0.8362 |
| Precision (macro) | 0.8525 |
| Recall (macro) | 0.8362 |
| F1 (macro) | 0.8389 |

### 2.3 Genomic Track (Best: HistGB)
| Metric | Value |
|---|---|
| Accuracy | 0.8889 |
| Precision (macro) | 0.8909 |
| Recall (macro) | 0.8893 |
| F1 (macro) | 0.8899 |
| Training Time (s) | 0.5409 |
| Inference Time (s) | 0.0237 |

## 3. Full Benchmark Tables

### 3.1 Productivity Models
| model | train_accuracy | test_accuracy | accuracy | precision_macro | recall_macro | f1_macro | train_seconds | infer_ms_per_1000 | model_size_mb | peak_train_ram_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.9692 | 0.9335 | 0.9335 | 0.9337 | 0.9335 | 0.9336 | 1.7479 | 10.6947 | 2.4548 | 1.2539 |
| XGBoost | 0.9039 | 0.8606 | 0.8606 | 0.8606 | 0.8606 | 0.8606 | 0.6904 | 4.9575 | 1.4641 | 91.6836 |
| DecisionTree | 0.6773 | 0.6522 | 0.6522 | 0.6656 | 0.6519 | 0.6453 | 0.0192 | 1.7934 | 0.0320 | 0.1953 |
| ExtraTrees | 1.0000 | 0.5000 | 0.5000 | 0.5002 | 0.4999 | 0.5000 | 0.5046 | 31.2805 | 294.8443 | 315.4531 |
| RandomForest | 1.0000 | 0.4687 | 0.4687 | 0.4699 | 0.4686 | 0.4691 | 0.4297 | 33.0771 | 110.7611 | 138.8711 |
| LogisticRegression | 0.4065 | 0.3911 | 0.3911 | 0.3866 | 0.3908 | 0.3835 | 0.5340 | 1.5644 | 0.0071 | 1.3281 |

### 3.2 Sustainability Models
| model | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| XGBoost_rebuild | 0.8362 | 0.8525 | 0.8362 | 0.8389 |

### 3.3 Genomic Models
| Model | Training_Time_s | Inference_Time_s | Accuracy | Precision_macro | Recall_macro | F1_Score_macro |
|---|---:|---:|---:|---:|---:|---:|
| NaiveBayes | 0.0089 | 0.0010 | 0.3919 | 0.3937 | 0.3965 | 0.3693 |
| KNN_k3 | 0.0157 | 0.0113 | 0.5776 | 0.5800 | 0.5790 | 0.5754 |
| KNN_k5 | 0.0171 | 0.0121 | 0.5823 | 0.5814 | 0.5843 | 0.5780 |
| KNN_k7 | 0.0160 | 0.0126 | 0.5836 | 0.5833 | 0.5851 | 0.5816 |
| SVM_linear | 3.3416 | 0.2034 | 0.3778 | 0.3978 | 0.3784 | 0.3499 |
| SVM_rbf | 2.6245 | 0.9326 | 0.4095 | 0.4080 | 0.4128 | 0.3993 |
| AdaBoost | 0.5850 | 0.0115 | 0.4494 | 0.4473 | 0.4513 | 0.4440 |
| HistGB | 0.5409 | 0.0237 | 0.8889 | 0.8909 | 0.8893 | 0.8899 |

## 4. Top Genomic Feature Drivers
| Feature | Mean Importance | Std |
|---|---:|---:|
| water_SecchiDepth (m) | 0.193433 | 0.001801 |
| water_pH | 0.045754 | 0.001292 |
| water_WaterDepth (m) | 0.025036 | 0.000903 |
| water_AirTemp (C) | 0.019889 | 0.000748 |
| country | 0.002295 | 0.000291 |

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
2. Model outputs are predictive evidence and should be interpreted with domain and ecological validation.
