# Explainable AI Evidence and Novelty Positioning

<p align="center"><strong>Interpretability, Reliability, and Research Contribution Evidence</strong></p>

<p align="center">
	<img src="https://img.shields.io/badge/Explainability-SHAP_Integrated-0C7BDC" alt="Explainability" />
	<img src="https://img.shields.io/badge/Coverage-Productivity_Sustainability_Genomic-1D3557" alt="Coverage" />
	<img src="https://img.shields.io/badge/Use_Case-Decision_Support-2A9D8F" alt="Use Case" />
</p>

## 1. Explainability Design Principles
1. The platform provides prediction outputs together with feature-level rationale.
2. Interpretation combines local explanation and model-compatible global/fallback evidence.
3. Explainability is implemented across all three tracks to preserve cross-track analytical consistency.

## 2. Track-Wise Explainability Implementation

### 2.1 Productivity Track
1. Primary path: SHAP attribution over aligned model input schema.
2. Secondary path: fallback influence estimation when SHAP path fails.
3. Escalation path: global feature influence when local signal is near-zero.

### 2.2 Sustainability Track
1. Supports bundle-aware explanation for mixed artifact formats.
2. Uses SHAP for active model path where feasible.
3. Falls back to weighted delta influence under constrained explainability conditions.

### 2.3 Genomic Track
1. Uses schema-aligned numeric matrix construction for explainability stability.
2. Prefers tree-compatible SHAP path for HistGradientBoostingClassifier.
3. Provides robust fallback influence path when SHAP execution is unavailable.

## 3. Evidence Surfaces Available in the System
1. Local feature attributions with signed and absolute impact values.
2. Grouped model visual evidence tabs for comparative diagnostics.
3. Train versus test evidence figures for generalization inspection.
4. Scenario-based confidence deltas for stress-response analysis.
5. Full artifact archive in Results Gallery for reproducible evidence review.

## 4. Interpretation Protocol for Review Committees
1. Evaluate top absolute-impact features first, then inspect signed direction.
2. Corroborate local explanations with global/fallback influence where applicable.
3. Use train-test visual diagnostics to identify overfitting or underfitting risk.
4. Confirm scenario sensitivity through both score movement and confidence deltas.
5. Treat model explanations as predictive rationale, not biological causality proof.

## 5. Reliability and Governance Constraints
1. Explanations are data-distribution dependent and may shift across unseen populations.
2. Categorical factorization can impose ordinal artifacts in some genomic contexts.
3. Proxy targets improve operational modeling utility but do not replace field validation.
4. Final decisions should combine model evidence with domain expert review.

## 6. Novelty Evidence and Gap Positioning

### 6.1 System-Level Novel Contributions
1. Tri-track integration in one deployable inference and explanation environment.
2. End-to-end bridge from model output to scenario stress testing and policy guidance.
3. Unified operational context framing for region-relevant aquaculture decision support.

### 6.2 Baseline Gap Matrix
| Common prior pattern | Limitation | This project contribution |
|---|---|---|
| Single-task aquaculture prediction | No cross-track synthesis | Synchronized tri-track inference |
| Black-box output reporting | Low trust and weak actionability | SHAP-supported explainability with fallback controls |
| Static notebook-only benchmarking | Limited deployment relevance | Interactive dashboard with scenario and policy modules |
| Generic recommendation outputs | Limited operational grounding | Model-conditioned, risk-prioritized recommendations |

### 6.3 Defensible Novelty Statement
This project does not claim novelty in isolated algorithms. The novelty lies at the system-integration layer:
an explainable, multi-track, policy-oriented aquaculture intelligence pipeline with synchronized inference,
evidence-rich diagnostics, and deployment-ready operational interfaces.

## 7. Repository Evidence Anchors
1. Model artifacts: models/
2. Integrated dashboard runtime: streamlit_app/app.py
3. Consolidated outcomes: results/final_project_results_summary.md
4. Validation controls: results/validation_audit.md
5. Benchmark evidence: results/productivity_metrics.csv and results/sustainability_metrics.csv

## 8. Forward Research Extensions
1. Expand external temporal and multi-region validation cohorts.
2. Replace proxy policy scoring with optimization or reinforcement-learning policy engines.
3. Introduce uncertainty calibration and confidence interval reporting for all tracks.
4. Add causal and counterfactual analysis modules for higher decision assurance.
