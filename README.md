# Aquaculture Intelligence Platform

<p align="center"><strong>Integrated Explainable AI for Productivity Intelligence, Sustainability Assessment, and Genomic-Aware Decision Support in Aquaculture</strong></p>

<p align="center">
	<img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
	<img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas" />
	<img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy" />
	<img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn" />
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow" />
	<img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
	<img src="https://img.shields.io/badge/SHAP-Explainability-0C7BDC" alt="SHAP" />
	<img src="https://img.shields.io/badge/Statsmodels-Forecasting-4B8BBE" alt="Statsmodels" />
</p>

<p align="center">
	<img src="https://img.shields.io/badge/Project_Type-Research_Engineering-1D3557" alt="Project Type" />
	<img src="https://img.shields.io/badge/Domain-Aquaculture_Intelligence-2A9D8F" alt="Domain" />
	<img src="https://img.shields.io/badge/Status-Actively_Maintained-0A9396" alt="Status" />
</p>

## <img src="https://img.shields.io/badge/Section-Executive_Summary-0B3D91?style=flat-square&logo=bookstack&logoColor=white" alt="Executive Summary" />
This repository delivers a full-stack analytical system that integrates climate, water-quality, production, and genomic signals into one coherent inference and decision layer. The platform is designed for serious research and operational translation, emphasizing reproducibility, explainability, quality controls, and model-to-action traceability.

The system supports three synchronized tracks:
1. Productivity and disease-pressure classification.
2. Sustainability-level classification under environmental context.
3. Genomic-aware class signaling and feature sensitivity analysis.

## <img src="https://img.shields.io/badge/Section-Research_Context-1D3557?style=flat-square&logo=googlescholar&logoColor=white" alt="Research Context" />
Aquaculture systems are affected by coupled biological and environmental variables. Traditional single-task pipelines often fail to bridge model outputs with decision use. This project addresses that gap by combining:
1. Multi-track model development over a shared processed dataset.
2. Unified explainability and evidence reporting.
3. Forecasting, scenario analysis, and policy-oriented recommendations in a deployable interface.

## <img src="https://img.shields.io/badge/Section-Core_Objectives-264653?style=flat-square&logo=target&logoColor=white" alt="Core Objectives" />
1. Build reproducible, benchmarked pipelines for productivity, sustainability, and genomic tracks.
2. Preserve methodological fairness across splits, preprocessing, and metric computation.
3. Expose model behavior using SHAP and model-compatible fallback logic.
4. Transform inference into actionable operational guidance through scenario and policy layers.

## <img src="https://img.shields.io/badge/Section-Workstreams-3A5A40?style=flat-square&logo=gitbook&logoColor=white" alt="Workstreams" />
| Contributor | Workstream | Primary Notebook |
|---|---|---|
| Khushi | Productivity benchmarking, disease-pressure modeling, integration-oriented reporting | notebooks/02_productivity_khushi.ipynb |
| Shravya | Sustainability classification pipeline with deep-learning and explainability analysis | notebooks/03_sustainability_shravya.ipynb |
| Janhavi | Genomic feature selection, model diagnostics, and importance interpretation | notebooks/04_genomic_xai_janhavi.ipynb |

## <img src="https://img.shields.io/badge/Section-System_Architecture-1F7A8C?style=flat-square&logo=databricks&logoColor=white" alt="System Architecture" />
1. Shared preprocessing produces standardized processed datasets for all downstream tracks.
2. Independent training scripts and notebooks produce model artifacts and benchmark outputs.
3. Streamlit integration aligns model schemas and executes synchronized inference.
4. Explainability layer provides local/global interpretation evidence.
5. Forecasting, scenario simulation, and recommendation modules convert model outputs into practical planning support.

## <img src="https://img.shields.io/badge/Section-Model_Tracks_and_Artifacts-0F4C5C?style=flat-square&logo=dependabot&logoColor=white" alt="Model Tracks and Artifacts" />
| Track | Problem Type | Artifact | Deployed Output |
|---|---|---|---|
| Productivity | Multi-class (Low/Medium/High) productivity risk | models/productivity_model.pkl | Class label and confidence |
| Sustainability | Multi-class sustainability level classification | models/sustainability_model.pkl | Class label and confidence distribution |
| Genomic | Genomic-informed class signal | models/feature_selector.pkl | Genomic class signal and feature influence |

## <img src="https://img.shields.io/badge/Section-Current_Derived_Results-7B2CBF?style=flat-square&logo=chartdotjs&logoColor=white" alt="Current Derived Results" />
These values are regenerated from the checked-in benchmark CSVs and the notebook-derived sustainability results.

| Track | Best Result Used in Reports | Accuracy | F1 | Notes |
|---|---|---:|---:|---|
| Productivity | ExtraTrees | 0.9601 | 0.9603 | Deployment-best model with optimal recall-weighted selection |
| Sustainability | MLP | 0.9481 | 0.9479 | Real notebook benchmark preserved in `results/sustainability_metrics.csv` |
| Genomic | HistGB | 0.9468 | 0.9470 | Best genomic benchmark in the 8-model comparison |

PSG fusion on the common 459-sample held-out subset is also generated from real predictions: accuracy 0.7233, precision 0.7233, recall 0.7229, F1 0.7230. See `results/psg_combined_metrics.csv` and `results/paper_results_tables.md`.

### Methodology Note: PSG Fusion Rule
The PSG combined row is computed as $R = 0.50\,P_r + 0.35\,S_r + 0.15\,G_r$ on the shared held-out samples. The continuous score is then discretized using **data-driven tertiles** of R itself, matching the quantile-based approach used for the production target. This ensures class balance and adapts to R's actual distribution, avoiding calibration bias from fixed thresholds.

## <img src="https://img.shields.io/badge/Section-Dashboard_Modules-22577A?style=flat-square&logo=streamlit&logoColor=white" alt="Dashboard Modules" />
### Predictions
Interactive profile-based synchronized inference across all model tracks.

### Explainability
Track-wise SHAP analysis with robust fallback and integrated visual evidence tabs.

### Results Gallery
Comprehensive archive view for all images, tables, and text artifacts generated in results.

### Architecture
Model-track architecture descriptions with explicit input-processing-model-output mapping.

### Forecasting
Production trajectory estimation with ARIMA support and trend fallback.

### Scenario Simulation
What-if stress testing under controlled perturbations (temperature, rainfall, water quality).

### Policy Recommendations
Rule-guided recommendations conditioned on current model outputs and profile context.

## <img src="https://img.shields.io/badge/Section-Rebuild_Workflow-5F0F40?style=flat-square&logo=apacheairflow&logoColor=white" alt="Rebuild Workflow" />
Running python run_all.py executes scripts/full_rebuild.py with the following pipeline:
1. Shared preprocessing execution.
2. Productivity model training and benchmark refresh.
3. Genomic model training and benchmark refresh.
4. Sustainability model bundle build.
5. Genomic importance recomputation.
6. Validation and leakage audit generation.
7. Visual report generation.
8. Consolidated final summary regeneration.
9. Quality gate validation.

## <img src="https://img.shields.io/badge/Section-Repository_Structure-6C757D?style=flat-square&logo=files&logoColor=white" alt="Repository Structure" />
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
|  |- validation_audit.md
|  |- novelty_evidence.md
|- scripts/
|  |- full_rebuild.py
|  |- train_productivity_model.py
|  |- train_genomic_model.py
|  |- build_sustainability_model.py
|  |- generate_visual_reports.py
|  |- generate_final_results_summary.py
|  |- validation_audit.py
|  |- quality_gate.py
|- streamlit_app/
|  |- app.py
|- run_all.py
|- requirements.txt
|- README.md
```

## <img src="https://img.shields.io/badge/Section-Technology_Stack-344E41?style=flat-square&logo=stackshare&logoColor=white" alt="Technology Stack" />
| Layer | Technologies |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Classical ML | scikit-learn, XGBoost |
| Deep Learning | TensorFlow, Keras |
| Explainability | SHAP |
| Forecasting | statsmodels (ARIMA) |
| Visualization | matplotlib, seaborn |
| Application Layer | Streamlit |
| Artifact Serialization | joblib |

## <img src="https://img.shields.io/badge/Section-Setup_and_Execution-495057?style=flat-square&logo=windows-terminal&logoColor=white" alt="Setup and Execution" />
### Environment Setup
```bash
git clone https://github.com/YOUR_USERNAME/Aquaculture-MajorProject.git
cd Aquaculture-MajorProject

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Full Pipeline Run
```bash
python run_all.py
```

### Launch Application
```bash
powershell -ExecutionPolicy Bypass -File scripts/start_streamlit.ps1
```

Fallback direct launch:
```bash
python -m streamlit run streamlit_app/app.py
```

### Streamlit Dynamic Module Error Recovery
If you see errors such as "Failed to fetch dynamically imported module" for paths under /static/js/index.<hash>.js:
1. Run the startup script above (it clears stale port owners on 8501 by default).
2. In the browser, perform a hard refresh (Ctrl+F5).
3. If needed, start on a different port:

```bash
powershell -ExecutionPolicy Bypass -File scripts/start_streamlit.ps1 -Port 8502
```

## <img src="https://img.shields.io/badge/Section-Quality_Assurance-003049?style=flat-square&logo=checkmarx&logoColor=white" alt="Quality Assurance" />
### Quality Gate
```bash
python scripts/quality_gate.py
```

### Key Guarantees
1. Required model artifacts are validated before pass.
2. Required result files are validated before pass.
3. Core result schema checks are enforced for reproducibility.

## <img src="https://img.shields.io/badge/Section-Research_Outputs-7B2CBF?style=flat-square&logo=googledocs&logoColor=white" alt="Research Outputs" />
Primary generated outputs include:
1. results/paper_results_tables.md — Full per-model tables for productivity, sustainability, and genomic tracks, plus the PSG fusion result
2. results/psg_combined_metrics.csv — Sample-level PSG fusion metrics on the common held-out subset
3. results/final_project_results_summary.md — Consolidated evidence report for all tracks, regenerated from the real benchmark CSVs
4. results/validation_audit.md — Validation and leakage audit
5. results/novelty_evidence.md — Research novelty evidence
6. results/full_dataset_summary.md — Comprehensive column-wise statistics
7. results/track_dataset_summaries.md — Productivity, Sustainability, Genomic dataset splits and distributions
8. results/track_dataset_statistics.csv — Class distribution statistics across tracks
9. results/model_comparison_visualization.png — Top 3 models performance comparison (bar charts)
10. results/model_comparison_summary.md — Detailed model comparison and recommendations
11. Track-level metrics, plots, and explainability artifacts in results/

## <img src="https://img.shields.io/badge/Section-Project_Intent-14213D?style=flat-square&logo=academia&logoColor=white" alt="Project Intent" />
This project is developed as a formally structured research-engineering system, not a demo-only dashboard. Its design priority is evidence-backed, reproducible, and interpretable model deployment for aquaculture decision intelligence under multi-signal uncertainty.
