# Explainable AI Evidence Summary

## Explainability Design
- The system provides both prediction outputs and feature-level rationale.
- It combines local explanations (current profile) and global explanations (overall model behavior).
- Evidence is presented across productivity, sustainability, and genomic tracks.

## Evidence Available in the Dashboard
1. Local feature attribution
- Explainability shows top features with signed impact and absolute impact.
- Positive impact indicates an increase in predicted productivity.
- Negative impact indicates a reduction in predicted productivity.

2. Global importance fallback
- When local impacts are near zero, the dashboard automatically shows global feature importance.
- This prevents ambiguous interpretation from near-zero local signals.

3. Cross-model comparison visuals
- Spiral charts compare Accuracy, Precision, Recall, and F1 by model and track.
- These visuals support model selection discussions with consistent metrics.

4. Generalization diagnostics
- Train vs test accuracy is displayed for productivity, sustainability, and genomic tracks.
- A combined all-track chart is included for quick cross-track comparison.

5. Scenario sensitivity and confidence deltas
- Baseline and scenario outputs are presented side by side.
- Confidence deltas are model-derived and reflect response to input changes.

## Interpretation Notes
- Review direction and magnitude of feature impacts before operational decisions.
- Use train/test gap as an overfitting indicator.
- Validate scenario changes with both score shifts and confidence deltas.

## Conclusion
Each prediction is accompanied by interpretable feature evidence and model-quality diagnostics, supporting transparent and research-oriented analysis.

# Novelty Evidence and Gap Positioning

## What Is New in This Work
- End-to-end tri-track operational integration in one deployable interface: productivity risk, sustainability class, and genomic-driven sensitivity.
- Explainability-to-action chain in a single workflow: SHAP interpretation, scenario stress testing, forecast projection, and policy prioritization.
- India-context decision framing integrated directly into runtime profile controls and intervention narratives.

## Claimed Novelty in This Project
- Integrated tri-track stack: productivity + sustainability + genomic feature intelligence in one operational dashboard.
- Explainability-to-policy bridge: model outputs translated into interpretable decision actions.
- India-context operational framing: country-context simulation and farm-oriented stress controls.

## Baseline Gap Table
| Common prior pattern | Limitation | This project's contribution |
|---|---|---|
| Single-task aquaculture prediction | No cross-track synthesis | Multi-track synchronized inference |
| Black-box model reporting | Low trust and poor actionability | SHAP-backed interpretation with guided narratives |
| Static benchmarking notebooks | Weak deployment relevance | Interactive scenario and forecasting modules |
| Generic recommendations | Low domain grounding | Model-assisted, risk-prioritized policy guidance |

## Defensible Novelty Statement
This work does not claim to invent each algorithmic component independently. The novelty claim is at the systems level:
an integrated, explainable, policy-oriented aquaculture intelligence pipeline with synchronized multi-model inference and
India-focused operational framing.

## Evidence Anchors in Repository
- Integrated model artifacts in models/
- Unified dashboard with multi-tab inference in streamlit_app/app.py
- Consolidated result summary in results/final_project_results_summary.md
- Validation and leakage audit in results/validation_audit.md
- Robustness and benchmark metrics in results/productivity_metrics.csv and results/sustainability_metrics.csv

## Limitations and Next Research Extension
- External multi-region validation should be expanded.
- Policy layer can be upgraded from model-assisted scoring to reinforcement learning optimization.
