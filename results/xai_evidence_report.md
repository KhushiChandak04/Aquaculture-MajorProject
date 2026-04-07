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
