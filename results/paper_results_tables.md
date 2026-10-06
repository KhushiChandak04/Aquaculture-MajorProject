# Paper Results Tables

This file is generated from the current benchmark CSVs in results/.

## Feature Cleanup Note

The current benchmark uses the official 9-column dataset: country, year, production, and six time-bucket water-quality features. Climate columns were excluded because the climate extract has no production-year overlap, and genomic global aggregates were excluded because no production-compatible join key exists. Earlier benchmarks containing zero-variance climate/genomic columns are not valid evidence of predictive performance. The current PSG result is the honest post-cleanup baseline.

## Table II - Productivity Models
| model | train_accuracy | test_accuracy | accuracy | precision_macro | recall_macro | f1_macro | train_seconds | infer_ms_per_1000 | model_size_mb | peak_train_ram_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.9692 | 0.9335 | 0.9335 | 0.9337 | 0.9335 | 0.9336 | 1.7479 | 10.6947 | 2.4548 | 1.2539 |
| XGBoost | 0.9039 | 0.8606 | 0.8606 | 0.8606 | 0.8606 | 0.8606 | 0.6904 | 4.9575 | 1.4641 | 91.6836 |
| DecisionTree | 0.6773 | 0.6522 | 0.6522 | 0.6656 | 0.6519 | 0.6453 | 0.0192 | 1.7934 | 0.0320 | 0.1953 |
| ExtraTrees | 1.0000 | 0.5000 | 0.5000 | 0.5002 | 0.4999 | 0.5000 | 0.5046 | 31.2805 | 294.8443 | 315.4531 |
| RandomForest | 1.0000 | 0.4687 | 0.4687 | 0.4699 | 0.4686 | 0.4691 | 0.4297 | 33.0771 | 110.7611 | 138.8711 |
| LogisticRegression | 0.4065 | 0.3911 | 0.3911 | 0.3866 | 0.3908 | 0.3835 | 0.5340 | 1.5644 | 0.0071 | 1.3281 |

## Table III - Sustainability Models
| model | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| XGBoost_rebuild | 0.8362 | 0.8525 | 0.8362 | 0.8389 |

## Table IV - Genomic Models
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

