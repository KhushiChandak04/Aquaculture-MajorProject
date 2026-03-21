$path = "notebooks/04_genomic_xai_janhavi.ipynb"
$nb = Get-Content $path -Raw | ConvertFrom-Json

foreach ($cell in $nb.cells) {
    if ($cell.cell_type -ne "code") { continue }

    $src = ($cell.source -join "`n")

    if ($src -match "TODO: Implement feature selection") {
        $cell.source = @(
            '# ============================================================`n',
            '# FEATURE SELECTION SUMMARY (FINAL)`n',
            '# ============================================================`n',
            'selector = {`n',
            '    "method": "mutual_info_top5",`n',
            '    "selected_features": top_5_features,`n',
            '    "n_selected": len(top_5_features),`n',
            '}`n',
            '`n',
            'print("Selected", selector["n_selected"], "features out of", X_train.shape[1])`n',
            'print("Selected features:")`n',
            'print(selector["selected_features"])`n',
            '`n',
            'selected_features_df = pd.DataFrame({`n',
            '    "feature": top_5_features,`n',
            '    "selection_score": mi_df.set_index("Feature").loc[top_5_features, "MI_Score"].values,`n',
            '}).sort_values("selection_score", ascending=False)`n',
            '`n',
            'print("\nSelected feature table:")`n',
            'print(selected_features_df.to_string(index=False))'
        )
        $cell.outputs = @()
        $cell.execution_count = $null
    }
    elseif ($src -match "TODO: Generate SHAP plots and importance scores") {
        $cell.source = @(
            '# ============================================================`n',
            '# SHAP / FEATURE IMPORTANCE (PERMUTATION + OPTIONAL SHAP)`n',
            '# ============================================================`n',
            'perm = permutation_importance(`n',
            '    best_model,`n',
            '    X_test_reduced_scaled,`n',
            '    y_test,`n',
            '    n_repeats=20,`n',
            '    random_state=42,`n',
            '    scoring="f1_macro",`n',
            ')`n',
            '`n',
            'importances = pd.DataFrame({`n',
            '    "feature": X_test_reduced_scaled.columns,`n',
            '    "importance_mean": perm.importances_mean,`n',
            '    "importance_std": perm.importances_std,`n',
            '}).sort_values("importance_mean", ascending=False).reset_index(drop=True)`n',
            '`n',
            'print("Top feature importances:")`n',
            'print(importances.head(10).to_string(index=False))`n',
            '`n',
            'if hasattr(best_model, "feature_importances_") and hasattr(best_model, "predict"):`n',
            '    try:`n',
            '        explainer = shap.Explainer(best_model, X_train_reduced_scaled)`n',
            '        shap_values = explainer(X_test_reduced_scaled)`n',
            '        shap.plots.beeswarm(shap_values, max_display=10, show=True)`n',
            '    except Exception as exc:`n',
            '        print(f"SHAP summary skipped: {exc}")`n',
            'else:`n',
            '    print("SHAP summary skipped: model type is not tree-based.")'
        )
        $cell.outputs = @()
        $cell.execution_count = $null
    }
    elseif ($src -match "TODO: Uncomment to save selector and importance CSV") {
        $cell.source = @(
            '# ============================================================`n',
            '# SAVE FEATURE SELECTOR & IMPORTANCE (FINAL)`n',
            '# ============================================================`n',
            'MODEL_PATH = os.path.join("..", "models", "feature_selector.pkl")`n',
            'RESULT_PATH = os.path.join("..", "results", "genomic_feature_importance.csv")`n',
            '`n',
            'os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)`n',
            'os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)`n',
            '`n',
            'joblib.dump(best_model, MODEL_PATH)`n',
            'print(f"Feature selector saved -> {MODEL_PATH}")`n',
            '`n',
            'importances.to_csv(RESULT_PATH, index=False)`n',
            'print(f"Feature importance saved -> {RESULT_PATH}")`n',
            '`n',
            '_loaded = joblib.load(MODEL_PATH)`n',
            'print("Reload check:", type(_loaded))'
        )
        $cell.outputs = @()
        $cell.execution_count = $null
    }
}

$nb | ConvertTo-Json -Depth 100 | Set-Content $path
Write-Output "Notebook TODO cells updated on disk."
