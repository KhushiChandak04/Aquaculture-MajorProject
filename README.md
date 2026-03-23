# Aquaculture Intelligence Platform

Production-grade, research-oriented, explainable AI pipeline for aquaculture risk intelligence, sustainability analytics, and genomic-driven model interpretation.

## Project Vision
This repository delivers an integrated multi-model decision system for aquaculture operations. It combines climate, water quality, production, and genomic signals into a single interpretable intelligence stack with real-time inference, explainability, forecasting, scenario stress-testing, and policy guidance.

## Team and Research Tracks
| Contributor | Research Focus | Notebook |
|---|---|---|
| Khushi | Productivity and disease-pressure modeling, integration architecture | notebooks/02_productivity_khushi.ipynb |
| Shravya | Sustainability classification with deep architectures | notebooks/03_sustainability_shravya.ipynb |
| Janhavi | Genomic feature selection and model diagnostics | notebooks/04_genomic_xai_janhavi.ipynb |

## System Architecture
1. Unified preprocessing layer standardizes multi-source data into one analytical dataset.
2. Three specialized model tracks train independently with reproducible artifacts.
3. Streamlit orchestration layer aligns features and executes synchronized inference.
4. Explainability and forecasting modules convert predictions into operational insight.
5. Scenario and policy layers translate signals into action-oriented decisions.

## Repository Layout
```
Aquaculture-MajorProject/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── productivity_model.pkl
│   ├── sustainability_model.pkl
│   └── feature_selector.pkl
├── notebooks/
│   ├── 01_shared_preprocessing.ipynb
│   ├── 02_productivity_khushi.ipynb
│   ├── 03_sustainability_shravya.ipynb
│   └── 04_genomic_xai_janhavi.ipynb
├── preprocessing/
├── results/
│   ├── productivity_metrics.csv
│   ├── sustainability_metrics.csv
│   ├── genomic_feature_importance.csv
│   └── final_project_results_summary.md
├── scripts/
│   ├── build_sustainability_model.py
│   └── generate_final_results_summary.py
└── streamlit_app/
    └── app.py
```

## Model Artifacts and Outputs
| Track | Primary Artifact | Core Output |
|---|---|---|
| Productivity | models/productivity_model.pkl | disease-pressure class estimation |
| Sustainability | models/sustainability_model.pkl | sustainability class prediction with confidence |
| Genomic | models/feature_selector.pkl | genomic-informed predictive class signal |

## Advanced Enhancements Status
1. Interactive dashboard: Implemented
2. SHAP local interpretability: Implemented with safe fallback
3. Forecasting engine: Implemented with ARIMA and robust trend fallback
4. Scenario simulation: Implemented for multi-parameter stress testing
5. Policy recommendation layer: Implemented as interpretable rule-based engine
6. Consolidated project reporting: Implemented via results/final_project_results_summary.md

## Dashboard Tab Guide
### Predictions
Purpose:
Real-time integrated inference from productivity, sustainability, and genomic models on a common farm profile.

Interpretation:
- Productivity class reflects modeled disease-pressure tendency.
- Sustainability class reflects long-horizon operational resilience.
- Genomic class reflects feature-driven genomic response signal.
- SHAP panel explains local drivers for the current productivity prediction.

### Explainability
Purpose:
Model transparency and evaluation defense through metrics, importance, and benchmark artifacts.

Interpretation:
- SHAP explains local contribution direction and magnitude.
- Feature-importance tables summarize global influence.
- Comparison and efficiency plots justify final model selection.

### Forecasting
Purpose:
Projection of production trajectory from historical yearly production.

Interpretation:
- ARIMA(1,1,1) captures lag dynamics, differenced trend, and residual memory.
- Trend fallback ensures continuity when ARIMA backend is unavailable.
- Forecast should be combined with scenario outputs for planning confidence.

### Scenario Simulation
Purpose:
What-if analysis under parameter perturbations such as temperature, rainfall, and salinity shifts.

Interpretation:
- Baseline vs scenario class transitions indicate risk escalation, stability, or mitigation potential.
- Supports stress-testing of operational decisions before field deployment.

### Policy Recommendations
Purpose:
Actionable decision layer that maps model outputs to operational guidance.

Interpretation:
- Current engine is interpretable and rule-based.
- Recommendations prioritize water quality control, stress mitigation, and genomic monitoring.
- Designed for upgrade to model-driven policy optimization in future work.

## Tech Stack
- Python
- pandas and NumPy for analytical data handling
- scikit-learn for preprocessing and classical ML
- TensorFlow and Keras for deep neural architectures
- SHAP for local explainable AI attribution
- statsmodels for ARIMA forecasting
- matplotlib and seaborn for scientific visualization
- Streamlit for interactive model operations and decision UI
- joblib for artifact serialization and reproducibility

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/Aquaculture-MajorProject.git
cd Aquaculture-MajorProject

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline and Dashboard
```bash
python run_all.py
python scripts/generate_final_results_summary.py
python -m streamlit run streamlit_app/app.py
```

## Consolidated Final Results
Single-file project summary is generated at:

results/final_project_results_summary.md

This file aggregates artifact readiness, best model snapshots, top genomic drivers, and system-level inference status in one concise deliverable.

## Research and Delivery Quality Notes
- End-to-end integration is dynamic across all tab modules.
- Runtime safely handles multiple model bundle formats.
- Explainability and forecasting components include fallback strategies for reliability.
- The repository is structured for reproducible evaluation and scalable extension.
