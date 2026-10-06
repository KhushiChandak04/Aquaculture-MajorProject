# Streamlit Application Technical Documentation

<p align="center"><strong>Deployment, Inference, Explainability, and Policy Decision Interface</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Interface-Streamlit_Application-1D3557" alt="Interface" />
  <img src="https://img.shields.io/badge/Scope-Tri_Track_Inference-2A9D8F" alt="Scope" />
  <img src="https://img.shields.io/badge/Explainability-SHAP_Integrated-0C7BDC" alt="Explainability" />
</p>

## 1. Purpose
This application is the operational front-end for the aquaculture intelligence pipeline. It integrates predictive modeling,
explainability outputs, scenario-based sensitivity analysis, forecasting, and policy-level recommendation support
within a unified user interface.

## 2. Functional Scope
1. Synchronized tri-track prediction:
   - Productivity risk classification.
   - Sustainability class prediction.
   - Genomic track classification support.
2. Explainability evidence:
   - SHAP-based local reasoning where available.
   - Robust fallback influence estimation for constrained execution paths.
3. Decision support modules:
   - Scenario stress simulation with confidence delta reporting.
   - Forecasting views with method-aware fallback behavior.
   - Policy recommendation block with risk-prioritized intervention guidance.
4. Evidence archival access:
   - Results Gallery renders all discovered result images.
   - CSV, Markdown, and text artifacts are accessible in the same runtime.

## 3. Application Structure
Primary implementation file:
- streamlit_app/app.py

Core UI sections:
1. Prediction inputs and synchronized outputs.
2. Explainability tabs grouped by evidence category.
3. Results Gallery for full project artifact visibility.
4. Architecture and model governance panels.
5. Forecasting and scenario simulation modules.
6. Policy recommendations and intervention notes.

## 4. Data and Model Dependencies
The application expects the following artifact classes to be present:
1. Trained model bundles in models/.
2. Processed datasets in data/processed/.
3. Result and evidence artifacts in results/.
4. Supporting metadata for preprocessing and feature alignment.

Representative artifacts:
- models/productivity_model.pkl
- models/sustainability_model.pkl
- models/feature_selector.pkl
- results/productivity_metrics.csv
- results/sustainability_metrics.csv
- results/janhavi_model_metrics.csv

## 5. Execution Workflow
1. Launch command (from project root):
   streamlit run streamlit_app/app.py
2. Runtime actions:
   - Load model bundles and preprocessing metadata.
   - Render input controls and profile configuration.
   - Execute synchronized inference across active tracks.
   - Generate explainability evidence and fallback signals.
   - Present outputs, visual diagnostics, and recommendations.
3. Archive review:
   - Open Results Gallery to inspect all available images and tabular/doc artifacts.

## 6. Explainability and Reliability Notes
1. SHAP evidence is provided when model compatibility and runtime resources permit.
2. Fallback influence paths preserve interpretability continuity.
3. Near-zero local attributions are accompanied by global/fallback context.
4. Explanations are predictive evidence and must be interpreted with domain validation.

## 7. Forecasting and Scenario Logic
1. Forecast module uses selected forecasting strategy and fallback behavior when strict assumptions are not met.
2. Scenario controls permit stress-testing environmental and operational factors.
3. Confidence delta is reported to support comparative interpretation against baseline profiles.

## 8. Operational Quality Controls
1. Input validation and schema-aligned transformation are enforced before inference.
2. Non-critical visualization errors degrade gracefully to avoid dashboard interruption.
3. Artifact discovery is recursive to reduce missing-evidence risk in nested result structures.

## 9. Troubleshooting
1. If model artifacts are missing:
   - Re-run training or artifact generation pipelines.
2. If explainability widgets are unavailable:
   - Verify SHAP compatibility and model bundle integrity.
3. If results visuals are incomplete:
   - Confirm artifacts exist under results/ and supported image formats are present.
4. If startup fails:
   - Validate dependency installation from requirements.txt.

## 10. Governance and Interpretation Policy
1. The dashboard supports decision analysis and not autonomous policy enforcement.
2. Final production decisions should involve domain experts and ecological constraints.
3. Model confidence should be considered alongside measurement uncertainty and data quality.

## 11. Related Documentation
- README.md
- results/final_project_results_summary.md
- results/xai_evidence_report.md
- results/validation_audit.md
