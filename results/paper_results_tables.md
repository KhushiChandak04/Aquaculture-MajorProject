# Paper Results Tables

This file is generated from the current benchmark CSVs in results/.

## Table II - Productivity Models
| model | accuracy | precision_macro | recall_macro | f1_macro | train_seconds | infer_ms_per_1000 | model_size_mb | peak_train_ram_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ExtraTrees | 0.9601 | 0.9607 | 0.9601 | 0.9603 | 1.3558 | 32.1874 | 103.8424 | 137.5898 |
| RandomForest | 0.9515 | 0.9530 | 0.9516 | 0.9518 | 0.9271 | 35.8677 | 65.2930 | 79.0312 |
| LogisticRegression | 0.9057 | 0.9059 | 0.9056 | 0.9057 | 0.1496 | 2.3860 | 0.0141 | 2.7539 |
| XGBoost | 0.7920 | 0.8145 | 0.7919 | 0.7960 | 0.4949 | 5.3200 | 0.7373 | 98.7461 |
| DecisionTree | 0.4082 | 0.2773 | 0.4077 | 0.3264 | 0.0338 | 3.0435 | 0.0111 | 1.7578 |

## Table III - Sustainability Models
| model | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| MLP | 0.9481 | 0.9481 | 0.9481 | 0.9479 |
| 1D CNN | 0.3992 | 0.4012 | 0.3987 | 0.3731 |
| GRU | 0.3358 | 0.2585 | 0.3358 | 0.2073 |
| LSTM | 0.4014 | 0.4155 | 0.4011 | 0.3719 |
| CNN-LSTM | 0.4258 | 0.4159 | 0.4255 | 0.4094 |

## Table IV - Genomic Models
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

