"""Aquaculture XAI dashboard shell with advanced enhancement modules.

This app is designed to work even when teammate artifacts are still pending.
"""

from pathlib import Path
import importlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

try:
    feature_model = joblib.load("../models/feature_selector.pkl")
    feature_model_loaded = True
except:
    feature_model_loaded = False

try:
    ARIMA = importlib.import_module("statsmodels.tsa.arima.model").ARIMA
    HAS_ARIMA = True
except Exception:
    ARIMA = None
    HAS_ARIMA = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = BASE_DIR / "data" / "processed" / "final_dataset.csv"

PRODUCTIVITY_MODEL_PATH = MODELS_DIR / "productivity_model.pkl"
SUSTAINABILITY_MODEL_PATH = MODELS_DIR / "sustainability_model.pkl"
FEATURE_SELECTOR_PATH = MODELS_DIR / "feature_selector.pkl"

PRODUCTIVITY_METRICS_PATH = RESULTS_DIR / "productivity_metrics.csv"
MODEL_COMPARISON_PLOT_PATH = RESULTS_DIR / "khushi_final_model_comparison.png"
XAI_FEATURE_PATH = RESULTS_DIR / "khushi_weak_model_xai_features.csv"
GENOMIC_IMPORTANCE_PATH = RESULTS_DIR / "genomic_feature_importance.csv"


def safe_read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def class_label_from_prediction(pred_value, model=None):
    if isinstance(pred_value, (int, np.integer)):
        if model is not None:
            try:
                est = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model
                if hasattr(est, "classes_"):
                    classes = list(est.classes_)
                    idx = int(pred_value)
                    if 0 <= idx < len(classes):
                        return str(classes[idx])
            except Exception:
                pass

        mapping = {0: "High", 1: "Low", 2: "Medium"}
        return mapping.get(int(pred_value), f"Class {pred_value}")
    return str(pred_value)


def model_feature_frame(df: pd.DataFrame):
    drop_cols = [
        "genomic_Disease_Risk_global_mode",
        "disease_risk_target",
        "disease_risk_score",
    ]
    keep_cols = [c for c in df.columns if c not in drop_cols]
    return df[keep_cols].copy()


@st.cache_data
def representative_profile(df: pd.DataFrame, _model, target_label: str):
    if df is None or _model is None:
        return None

    X = model_feature_frame(df)
    try:
        preds = _model.predict(X)
    except Exception:
        return None

    pred_labels = [class_label_from_prediction(p, _model) for p in preds]
    for idx, lbl in enumerate(pred_labels):
        if lbl.lower() == target_label.lower():
            return X.iloc[idx].to_dict()
    return None


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_feature_selector_model(path: Path):
    """Load Janhavi's feature selector model with explicit error handling."""
    try:
        if not path.exists():
            return None, f"Feature selector file not found: {path}"
        model = joblib.load(path)
        return model, None
    except FileNotFoundError:
        return None, f"Feature selector file not found: {path}"
    except Exception as exc:
        return None, f"Failed to load feature selector model from {path}: {exc}"


@st.cache_data
def load_base_data():
    return safe_read_csv(DATA_FILE)


def default_input_row(df: pd.DataFrame):
    row = {}
    for c in df.columns:
        if c == "genomic_Disease_Risk_global_mode":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            row[c] = float(df[c].median())
        else:
            mode_vals = df[c].mode(dropna=True)
            row[c] = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
    return row


def build_aligned_input_df(user_values: dict, model=None, reference_df: pd.DataFrame = None):
    """Create a 1-row input dataframe aligned to model training feature names/order."""
    row = dict(user_values)

    expected_features = None
    try:
        if model is not None and hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        elif (
            model is not None
            and hasattr(model, "named_steps")
            and "prep" in model.named_steps
            and hasattr(model.named_steps["prep"], "feature_names_in_")
        ):
            expected_features = list(model.named_steps["prep"].feature_names_in_)
    except Exception:
        expected_features = None

    if not expected_features:
        return pd.DataFrame([row])

    aligned = {}
    for col in expected_features:
        if col in row:
            aligned[col] = row[col]
            continue

        if reference_df is not None and col in reference_df.columns:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                aligned[col] = float(reference_df[col].median())
            else:
                mode_vals = reference_df[col].mode(dropna=True)
                aligned[col] = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
        else:
            aligned[col] = 0.0

    return pd.DataFrame([aligned], columns=expected_features)


@st.cache_data(show_spinner=False)
def compute_local_shap(_model, background_df: pd.DataFrame, input_row_df: pd.DataFrame, max_background: int = 80):
    """Compute SHAP contribution values for one input row with robust fallbacks."""
    if not HAS_SHAP:
        return None, "SHAP package is not available in this environment."
    if _model is None or background_df is None or input_row_df is None:
        return None, "Model/data unavailable for SHAP computation."
    if len(background_df) == 0:
        return None, "Background data is empty for SHAP computation."

    bg = background_df.sample(n=min(max_background, len(background_df)), random_state=42)
    try:
        if hasattr(_model, "named_steps") and "prep" in _model.named_steps and "model" in _model.named_steps:
            prep = _model.named_steps["prep"]
            est = _model.named_steps["model"]

            bg_t = prep.transform(bg)
            in_t = prep.transform(input_row_df)
            if hasattr(bg_t, "toarray"):
                bg_t = bg_t.toarray()
            if hasattr(in_t, "toarray"):
                in_t = in_t.toarray()

            feature_names = prep.get_feature_names_out()

            if hasattr(est, "coef_"):
                explainer = shap.LinearExplainer(est, bg_t)
                shap_values = explainer(in_t)
            elif hasattr(est, "feature_importances_"):
                explainer = shap.TreeExplainer(est)
                shap_values = explainer(in_t)
            else:
                explainer = shap.Explainer(est.predict, bg_t)
                shap_values = explainer(in_t)

            if isinstance(shap_values, list):
                arr = np.array(shap_values[0])
            else:
                arr = np.array(getattr(shap_values, "values", shap_values))

            if arr.ndim == 2:
                contrib = arr[0]
            elif arr.ndim == 3:
                pred_class_idx = 0
                if hasattr(est, "predict") and hasattr(est, "classes_"):
                    pred_cls = est.predict(in_t)[0]
                    try:
                        pred_class_idx = int(np.where(est.classes_ == pred_cls)[0][0])
                    except Exception:
                        pred_class_idx = 0
                contrib = arr[0, :, pred_class_idx]
            else:
                return None, f"Unexpected SHAP value shape: {arr.shape}"
        else:
            # Fallback when model is not a pipeline with named preprocessor/model steps.
            explainer = shap.Explainer(_model.predict, bg)
            shap_values = explainer(input_row_df)
            arr = np.array(shap_values.values)
            feature_names = input_row_df.columns
            if arr.ndim == 2:
                contrib = arr[0]
            elif arr.ndim == 3:
                contrib = arr[0].mean(axis=-1)
            else:
                return None, f"Unexpected SHAP value shape: {arr.shape}"

        out = pd.DataFrame(
            {
                "feature": feature_names,
                "contribution": contrib,
            }
        )
        out["abs_contribution"] = out["contribution"].abs()
        out = out.sort_values("abs_contribution", ascending=False)
        return out, None
    except Exception as exc:
        return None, f"SHAP computation failed: {exc}"


def forecast_series_arima(series: pd.Series, horizon: int = 8):
    if HAS_ARIMA and len(series) >= 20:
        model = ARIMA(series.values, order=(1, 1, 1)).fit()
        fc = model.forecast(steps=horizon)
        return np.array(fc), "ARIMA(1,1,1)"

    # Fallback forecast when statsmodels is unavailable.
    rolling = series.rolling(window=min(6, len(series)), min_periods=1).mean()
    slope = (rolling.iloc[-1] - rolling.iloc[max(0, len(rolling) - 6)]) / max(1, min(6, len(rolling) - 1))
    start = rolling.iloc[-1]
    fc = np.array([start + slope * (i + 1) for i in range(horizon)])
    return fc, "Trend fallback"


st.set_page_config(page_title="Aquaculture Advanced Dashboard", page_icon="🐟", layout="wide")
st.title("Aquaculture Advanced Dashboard")
st.caption("Interactive predictions, explainability, forecasting, simulation, and policy guidance")

productivity_model = load_model(PRODUCTIVITY_MODEL_PATH)
sustainability_model = load_model(SUSTAINABILITY_MODEL_PATH)
feature_selector_model, feature_selector_load_error = load_feature_selector_model(FEATURE_SELECTOR_PATH)
feature_model = feature_selector_model
feature_model_loaded = feature_selector_model is not None

data_df = load_base_data()
metrics_df = safe_read_csv(PRODUCTIVITY_METRICS_PATH)
xai_df = safe_read_csv(XAI_FEATURE_PATH)
if xai_df is None:
    xai_df = safe_read_csv(GENOMIC_IMPORTANCE_PATH)

st.sidebar.header("Artifact Status")
st.sidebar.write(f"Productivity model: {'Loaded' if productivity_model is not None else 'Missing'}")
st.sidebar.write(f"Sustainability model: {'Loaded' if sustainability_model is not None else 'Pending from Shravya'}")
st.sidebar.write(f"Feature selector model: {'Loaded' if feature_model_loaded else 'Missing'}")
if feature_selector_load_error:
    st.sidebar.caption(f"Feature selector load status: {feature_selector_load_error}")
st.sidebar.write(f"Data file: {'Loaded' if data_df is not None else 'Missing'}")

with st.sidebar.expander("Path Debug"):
    st.write(f"Base dir: {BASE_DIR}")
    st.write(f"Productivity model path: {PRODUCTIVITY_MODEL_PATH}")
    st.write(f"Data path: {DATA_FILE}")

tabs = st.tabs(
    [
        "Predictions",
        "Explainability",
        "Forecasting",
        "Scenario Simulation",
        "Policy Recommendations",
    ]
)

with tabs[0]:
    st.subheader("Predictions")
    if data_df is None:
        st.error("`data/processed/final_dataset.csv` is missing. Add the processed dataset first.")
    else:
        st.info(
            "This tab builds one input profile and sends it to the trained productivity model. "
            "Profile Mode chooses the starting baseline, and sliders apply what-if changes. "
            "The output class (High/Medium/Low) is the model's risk prediction for that adjusted profile."
        )
        with st.expander("How to demonstrate this tab"):
            st.markdown(
                """
1. Start with **Typical (Median)** to show neutral baseline behavior.
2. Switch to **Representative High/Medium/Low** to show model sensitivity across historical profiles.
3. Change sliders one by one and explain how environmental stress shifts the predicted risk.
                """
            )

        mode = st.selectbox(
            "Profile Mode",
            [
                "Typical (Median)",
                "Representative High (Model-Predicted)",
                "Representative Medium (Model-Predicted)",
                "Representative Low (Model-Predicted)",
            ],
        )

        base = default_input_row(data_df)
        if productivity_model is not None:
            if mode == "Representative High (Model-Predicted)":
                prof = representative_profile(data_df, productivity_model, "High")
                if prof is not None:
                    base.update(prof)
            elif mode == "Representative Medium (Model-Predicted)":
                prof = representative_profile(data_df, productivity_model, "Medium")
                if prof is not None:
                    base.update(prof)
            elif mode == "Representative Low (Model-Predicted)":
                prof = representative_profile(data_df, productivity_model, "Low")
                if prof is not None:
                    base.update(prof)

        c1, c2, c3 = st.columns(3)
        temp_delta = c1.slider("Temperature Delta (C)", -5.0, 5.0, 0.0, 0.5)
        rain_delta = c2.slider("Rainfall Delta", -50.0, 50.0, 0.0, 5.0)
        salinity_delta = c3.slider("Salinity Delta", -5.0, 5.0, 0.0, 0.25)

        c4, c5, c6 = st.columns(3)
        humidity_delta = c4.slider("Humidity Delta", -20.0, 20.0, 0.0, 1.0)
        water_temp_delta = c5.slider("Water Temp Delta (C)", -5.0, 5.0, 0.0, 0.5)
        ph_delta = c6.slider("pH Delta", -2.0, 2.0, 0.0, 0.1)

        for key, delta in [
            ("temperature_celsius", temp_delta),
            ("precip_mm", rain_delta),
            ("water_Salinity (ppt)", salinity_delta),
            ("humidity", humidity_delta),
            ("water_WaterTemp (C)", water_temp_delta),
            ("water_pH", ph_delta),
        ]:
            if key in base:
                base[key] = float(base[key]) + float(delta)

        input_df = build_aligned_input_df(base, productivity_model, data_df)

        feature_input_df = None
        if feature_model_loaded:
            feature_input_df = build_aligned_input_df(base, feature_model, data_df)
            # Best-effort numeric coercion for non-pipeline models that require numeric arrays.
            for col in feature_input_df.columns:
                if not pd.api.types.is_numeric_dtype(feature_input_df[col]):
                    if data_df is not None and col in data_df.columns:
                        known_vals = sorted(data_df[col].dropna().astype(str).unique().tolist())
                        val_to_num = {v: i for i, v in enumerate(known_vals)}
                        current_val = str(feature_input_df.at[0, col])
                        feature_input_df.at[0, col] = val_to_num.get(current_val, 0)
                    else:
                        feature_input_df.at[0, col] = 0
            feature_input_df = feature_input_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        local_shap_df = None
        if productivity_model is not None:
            try:
                pred = productivity_model.predict(input_df)[0]
                st.success(f"Primary model output: {class_label_from_prediction(pred, productivity_model)}")

                if HAS_SHAP:
                    show_local_shap = st.checkbox("Show SHAP explanation for this prediction", value=True)
                    if show_local_shap:
                        background_features = model_feature_frame(data_df)
                        local_shap_df, shap_err = compute_local_shap(
                            productivity_model,
                            background_features,
                            input_df,
                        )
                        if local_shap_df is not None:
                            top_local = local_shap_df.head(10).copy()
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.barplot(data=top_local, y="feature", x="contribution", ax=ax)
                            ax.axvline(0.0, color="black", linewidth=1)
                            ax.set_title("Local SHAP Contribution (Current Prediction)")
                            st.pyplot(fig)
                            st.dataframe(top_local[["feature", "contribution", "abs_contribution"]], width="stretch")
                        else:
                            st.info(shap_err)
                else:
                    st.info("SHAP is not installed in this runtime; using permutation importance in Explainability tab.")
            except Exception as exc:
                st.warning(f"Prediction could not run on current profile: {exc}")
        else:
            st.warning("`models/productivity_model.pkl` not found.")

if feature_model_loaded and feature_input_df is not None:
    try:
        prediction = feature_model.predict(feature_input_df)

        st.subheader("Model Prediction (Feature Selection)")

        st.info("This model uses selected important features to predict production category efficiently.")

        st.success(f"Production Category: {prediction[0]}")

        # Show selected input features
        with st.expander("🔍 View Input Features Used"):
            st.write("Actual Input Values (Readable Format):")

            input_table = pd.DataFrame([base]).T
            input_table.columns = ["Values"]
            st.dataframe(input_table)

        # Show feature names
        st.caption("Model is trained using top selected features (dimensionality reduction applied).")

    except Exception as exc:
        st.warning(f"Feature model prediction could not run: {exc}")

        if sustainability_model is None:
            st.info("Sustainability model pending from Shravya.")
        if not feature_model_loaded:
            st.info("Feature selector pending from Janhavi.")

with tabs[1]:
    st.subheader("Explainability")
    st.info(
        "This tab explains why the model behaves as it does. "
        "It shows top influential features and benchmark evidence for model quality vs efficiency."
    )
    with st.expander("How to demonstrate this tab"):
        st.markdown(
            """
1. Show top feature importance bars and explain which factors drive risk most.
2. Show the model comparison proof image to justify final model selection.
3. Reference the metrics table for quantitative evidence.
            """
        )

    if xai_df is not None and {"feature", "importance_mean"}.issubset(xai_df.columns):
        top = xai_df.sort_values("importance_mean", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top, y="feature", x="importance_mean", ax=ax)
        ax.set_title("Top Feature Importance (Permutation)")
        st.pyplot(fig)
    else:
        st.info("XAI feature file not available yet (`khushi_weak_model_xai_features.csv` or `genomic_feature_importance.csv`).")

    if HAS_SHAP and productivity_model is not None and data_df is not None:
        st.markdown("**SHAP Runtime Status**")
        sample_row = model_feature_frame(data_df).head(1)
        shap_probe_df, shap_probe_err = compute_local_shap(productivity_model, model_feature_frame(data_df), sample_row)
        if shap_probe_df is not None:
            st.success("SHAP explanation pipeline is active for live predictions.")
        else:
            st.warning(shap_probe_err)
    elif not HAS_SHAP:
        st.warning("SHAP package not available. Install `shap` to enable local explanations.")

    if MODEL_COMPARISON_PLOT_PATH.exists():
        st.image(str(MODEL_COMPARISON_PLOT_PATH), caption="Model Comparison Proof Plot")
    else:
        st.info("Model comparison plot not available yet.")

    if metrics_df is not None:
        st.markdown("**Current productivity metrics table**")
        st.dataframe(metrics_df, width="stretch")

with tabs[2]:
    st.subheader("Time-Series Forecasting")
    st.info(
        "This tab projects future production trend from historical yearly production. "
        "It is an independent forecasting prototype (ARIMA or trend fallback)."
    )
    st.caption("Optional/experimental module until full team merge")
    with st.expander("How to demonstrate this tab"):
        st.markdown(
            """
1. Point to historical line and then forecast line extension.
2. Change forecast horizon and show how long-term projection updates.
3. Explain this supports planning, not final disease-class output.
            """
        )

    if data_df is None or "year" not in data_df.columns or "production" not in data_df.columns:
        st.warning("Forecasting needs `year` and `production` columns in final dataset.")
    else:
        yearly = data_df.groupby("year", as_index=False)["production"].mean().sort_values("year")
        horizon = st.slider("Forecast Horizon (years)", 3, 15, 8, 1)
        fc, method = forecast_series_arima(yearly["production"], horizon=horizon)

        last_year = int(yearly["year"].max())
        future_years = np.arange(last_year + 1, last_year + 1 + horizon)
        fc_df = pd.DataFrame({"year": future_years, "forecast_production": fc})

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(yearly["year"], yearly["production"], marker="o", label="Historical")
        ax.plot(fc_df["year"], fc_df["forecast_production"], marker="o", linestyle="--", label=f"Forecast ({method})")
        ax.set_title("Production Trend Forecast")
        ax.set_xlabel("Year")
        ax.set_ylabel("Production")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(fc_df, width="stretch")
        st.info("LSTM module can be added after full team merge; ARIMA/trend prototype is active now.")

with tabs[3]:
    st.subheader("Scenario Simulation")
    st.info(
        "This tab runs baseline vs scenario comparison. "
        "You alter climate/water variables and observe how the prediction changes."
    )
    with st.expander("How to demonstrate this tab"):
        st.markdown(
            """
1. Keep all deltas at 0 and note baseline output.
2. Increase temperature/salinity or reduce rainfall and show scenario output.
3. Explain this is policy-testing: what happens if conditions worsen or improve.
            """
        )

    if data_df is None:
        st.warning("Simulation requires final dataset.")
    else:
        base = default_input_row(data_df)

        t_plus = st.slider("Temperature Change (C)", -4.0, 4.0, 2.0, 0.5)
        p_plus = st.slider("Rainfall Change", -100.0, 100.0, 0.0, 10.0)
        s_plus = st.slider("Salinity Change", -6.0, 6.0, 0.0, 0.5)

        baseline_df = build_aligned_input_df(base, productivity_model, data_df)
        scenario = dict(base)
        if "temperature_celsius" in scenario:
            scenario["temperature_celsius"] = float(scenario["temperature_celsius"]) + t_plus
        if "precip_mm" in scenario:
            scenario["precip_mm"] = float(scenario["precip_mm"]) + p_plus
        if "water_Salinity (ppt)" in scenario:
            scenario["water_Salinity (ppt)"] = float(scenario["water_Salinity (ppt)"]) + s_plus
        scenario_df = build_aligned_input_df(scenario, productivity_model, data_df)

        if productivity_model is not None:
            try:
                base_pred = class_label_from_prediction(productivity_model.predict(baseline_df)[0], productivity_model)
                scen_pred = class_label_from_prediction(productivity_model.predict(scenario_df)[0], productivity_model)
                st.write(f"Baseline output: **{base_pred}**")
                st.write(f"Scenario output: **{scen_pred}**")
            except Exception as exc:
                st.warning(f"Simulation prediction failed: {exc}")
        else:
            st.warning("Productivity model not found for simulation.")

with tabs[4]:
    st.subheader("Policy Recommendations (Rule-Based Interim)")
    st.info(
        "This tab converts model/data insights into operational recommendations. "
        "Current version is rule-based and will be upgraded after full team model merge."
    )
    st.caption("Safe interim policy layer; can be replaced by learned policy after team merge")
    with st.expander("How to demonstrate this tab"):
        st.markdown(
            """
1. Show each recommendation and map it to a measurable variable (pH, salinity, temperature).
2. Connect recommendations back to feature-importance and scenario outputs.
3. Explain this is decision-support for farm operations.
            """
        )

    recs = []
    if data_df is not None:
        if "water_pH" in data_df.columns:
            recs.append("Maintain water pH near historical median range for lower risk variability.")
        if "water_Salinity (ppt)" in data_df.columns:
            recs.append("Avoid abrupt salinity shifts; apply gradual adjustments with daily monitoring.")
        if "temperature_celsius" in data_df.columns:
            recs.append("For heat spikes, increase aeration and schedule adaptive feeding windows.")
        if "humidity" in data_df.columns:
            recs.append("Track humidity with water temperature jointly to preempt stress conditions.")

    if not recs:
        recs = ["Dataset/model integration pending. Policy recommendations will appear after data load."]

    for idx, rec in enumerate(recs, start=1):
        st.write(f"{idx}. {rec}")

    if sustainability_model is None:
        st.info("Policy layer will be strengthened once Shravya's sustainability model is available.")
    if feature_selector_model is None:
        st.info("Policy layer will include genomic adjustments once Janhavi's selector is available.")

st.markdown("---")
st.caption("Aquaculture XAI Project - Integration Dashboard")
