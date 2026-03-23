# Aquaculture Intelligence Platform

Research-grade, explainable AI system for aquaculture risk intelligence, sustainability scoring, and genomic-aware decision support.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-0C7BDC)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Forecasting-4B8BBE)

## Executive Summary
This repository implements an end-to-end analytical stack that fuses climate, water-quality, production, and genomic signals into a unified inference workflow. The platform is built for interpretable decision support, including local explainability, time-series forecasting, scenario stress tests, and policy-oriented recommendations.

## Research Objectives
1. Build reproducible, multi-model pipelines over a shared processed dataset.
2. Compare model behavior across productivity, sustainability, and genomic tracks.
3. Expose model decisions using explainability outputs.
4. Convert technical inference into action-oriented farm guidance.

## Team and Workstreams
| Contributor | Focus Area | Notebook |
|---|---|---|
| Khushi | Productivity and disease-pressure modeling, integration architecture | notebooks/02_productivity_khushi.ipynb |
| Shravya | Sustainability classification with deep learning variants | notebooks/03_sustainability_shravya.ipynb |
| Janhavi | Genomic feature selection and model diagnostics | notebooks/04_genomic_xai_janhavi.ipynb |

## System Architecture
1. Shared preprocessing creates a standardized analytical dataset.
2. Independent model tracks train and export reproducible artifacts.
3. Streamlit integration layer aligns features and runs synchronized inference.
4. Explainability and forecasting modules add model accountability and planning context.
5. Scenario and policy modules transform model outputs into actionable recommendations.

## Repository Structure
```text
Aquaculture-MajorProject/
|- data/
|  |- raw/
|  |- processed/
|- models/
|  |- productivity_model.pkl
|  |- sustainability_model.pkl
|  |- feature_selector.pkl
|- notebooks/
|  |- 01_shared_preprocessing.ipynb
|  |- 02_productivity_khushi.ipynb
|  |- 03_sustainability_shravya.ipynb
|  |- 04_genomic_xai_janhavi.ipynb
|- preprocessing/
|- results/
|  |- productivity_metrics.csv
|  |- sustainability_metrics.csv
|  |- genomic_feature_importance.csv
|  |- final_project_results_summary.md
|- scripts/
|  |- build_sustainability_model.py
|  |- generate_final_results_summary.py
|- streamlit_app/
|  |- app.py
|- run_all.py
|- requirements.txt
|- README.md
```

## Model Artifacts
| Track | Artifact | Output Type |
|---|---|---|
| Productivity | models/productivity_model.pkl | disease-pressure class inference |
| Sustainability | models/sustainability_model.pkl | sustainability class plus confidence |
| Genomic | models/feature_selector.pkl | genomic-informed class signal |

## Dashboard Modules
### Predictions
Always-on interactive inference for all integrated model tracks with profile controls and interpretation blocks.

### Explainability
Model evidence layer with SHAP contribution views, benchmark metrics, and model-comparison artifacts.

### Forecasting
Production trend projection using ARIMA(1,1,1) with robust trend fallback.

### Scenario Simulation
What-if parameter perturbation for thermal, rainfall, and salinity stress analysis.

### Policy Recommendations
Rule-guided action layer translating model outcomes into operational controls.

## Advanced Features Status
1. Multi-model dashboard integration: Implemented
2. Local SHAP explainability with fallback: Implemented
3. Time-series forecasting with fallback: Implemented
4. Scenario stress testing: Implemented
5. Policy recommendation layer: Implemented
6. Consolidated reporting pipeline: Implemented

## Tech Stack
| Layer | Tools |
|---|---|
| Language | Python |
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost |
| DL | TensorFlow, Keras |
| Explainability | SHAP |
| Forecasting | statsmodels (ARIMA) |
| Visualization | matplotlib, seaborn |
| App Layer | Streamlit |
| Artifact IO | joblib |

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/Aquaculture-MajorProject.git
cd Aquaculture-MajorProject

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python run_all.py
python -m streamlit run streamlit_app/app.py
```

`python run_all.py` now executes a full rebuild workflow:
1. shared preprocessing
2. productivity retraining + benchmark refresh
3. genomic retraining + metric refresh
4. sustainability retraining
5. genomic importance recomputation
6. validation and leakage audit generation
7. consolidated summary regeneration
8. quality gate validation

## Reproducibility and Quality Gate
```bash
python scripts/quality_gate.py
```

## Generated Research Artifacts
- results/final_project_results_summary.md
- results/validation_audit.md
- results/novelty_evidence.md

## Consolidated Results
The unified final summary is generated at:

results/final_project_results_summary.md

It captures artifact readiness, best-model snapshots, top genomic drivers, and system-level inference coverage.

## Quality Notes
1. Integration is dynamic across all tab modules.
2. Runtime handles mixed model bundle formats safely.
3. Explainability and forecasting include fallback strategies for reliability.
4. The repository layout supports reproducible experimentation and extension.
