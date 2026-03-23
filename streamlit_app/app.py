"""Final integrated Streamlit dashboard for the Aquaculture XAI project."""

from pathlib import Path
import importlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

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
SUSTAINABILITY_METRICS_PATH = RESULTS_DIR / "sustainability_metrics.csv"
GENOMIC_IMPORTANCE_PATH = RESULTS_DIR / "genomic_feature_importance.csv"

KHUSHI_PLOT_PATH = RESULTS_DIR / "khushi_final_model_comparison.png"
SUSTAINABILITY_COMPARISON_PATH = RESULTS_DIR / "sustainability_model_comparison.png"
SUSTAINABILITY_EFFICIENCY_PATH = RESULTS_DIR / "sustainability_inference_training_ram.png"
JANHAVI_COMPARISON_PATH = RESULTS_DIR / "janhavi_model_comparison.png"
JANHAVI_EFFICIENCY_PATH = RESULTS_DIR / "janhavi_efficiency_plot.png"


def safe_read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_resource
def load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_data
def load_data():
    return safe_read_csv(DATA_FILE)


def model_feature_frame(df: pd.DataFrame):
    drop_cols = {
        "genomic_Disease_Risk_global_mode",
        "disease_risk_target",
        "disease_risk_score",
        "Production_Category",
    }
    return df[[c for c in df.columns if c not in drop_cols]].copy()


def default_input_row(df: pd.DataFrame):
    row = {}
    for col in model_feature_frame(df).columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(df[col].median())
        else:
            mode_vals = df[col].mode(dropna=True)
            row[col] = mode_vals.iloc[0] if len(mode_vals) else "Unknown"

    # Keep default context aligned to India-focused interpretation when available.
    if "country" in row:
        countries = df["country"].dropna().astype(str).unique().tolist()
        if "India" in countries:
            row["country"] = "India"

    return row


def get_expected_features(model_obj, bundle=None):
    if bundle and isinstance(bundle, dict) and "feature_columns" in bundle:
        return list(bundle["feature_columns"])

    if (
        bundle
        and isinstance(bundle, dict)
        and "preprocessor" in bundle
        and hasattr(bundle["preprocessor"], "feature_names_in_")
    ):
        return list(bundle["preprocessor"].feature_names_in_)

    if hasattr(model_obj, "feature_names_in_"):
        return list(model_obj.feature_names_in_)

    if (
        hasattr(model_obj, "named_steps")
        and "prep" in model_obj.named_steps
        and hasattr(model_obj.named_steps["prep"], "feature_names_in_")
    ):
        return list(model_obj.named_steps["prep"].feature_names_in_)

    return None


def build_aligned_input_df(user_values: dict, model_obj=None, reference_df: pd.DataFrame = None, bundle=None):
    expected = get_expected_features(model_obj, bundle=bundle)
    if not expected:
        return pd.DataFrame([dict(user_values)])

    aligned = {}
    for col in expected:
        if col in user_values:
            aligned[col] = user_values[col]
            continue
        if reference_df is not None and col in reference_df.columns:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                aligned[col] = float(reference_df[col].median())
            else:
                mode_vals = reference_df[col].mode(dropna=True)
                aligned[col] = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
        else:
            aligned[col] = 0.0

    return pd.DataFrame([aligned], columns=expected)


def class_label_from_prediction(pred_value, model_obj=None):
    if isinstance(pred_value, (int, np.integer)) and model_obj is not None:
        try:
            est = model_obj.named_steps["model"] if hasattr(model_obj, "named_steps") and "model" in model_obj.named_steps else model_obj
            if hasattr(est, "classes_"):
                classes = list(est.classes_)
                idx = int(pred_value)
                if 0 <= idx < len(classes):
                    return str(classes[idx])
        except Exception:
            pass

    if isinstance(pred_value, (int, np.integer)):
        mapping = {0: "High", 1: "Low", 2: "Medium"}
        return mapping.get(int(pred_value), f"Class {pred_value}")
    return str(pred_value)


def predict_sustainability(bundle_or_model, input_df: pd.DataFrame):
    if bundle_or_model is None:
        return None, None, "Sustainability model unavailable"

    model_obj = bundle_or_model
    label_encoder = None
    preprocessor = None

    if isinstance(bundle_or_model, dict):
        model_obj = bundle_or_model.get("model")
        label_encoder = bundle_or_model.get("label_encoder")
        preprocessor = bundle_or_model.get("preprocessor")

    if model_obj is None:
        return None, None, "Invalid sustainability bundle"

    try:
        model_input = input_df
        if preprocessor is not None and not (
            hasattr(model_obj, "named_steps") and "prep" in model_obj.named_steps
        ):
            model_input = preprocessor.transform(input_df)

        if hasattr(model_input, "toarray"):
            model_input = model_input.toarray()

        try:
            pred = model_obj.predict(model_input, verbose=0)
        except TypeError:
            pred = model_obj.predict(model_input)

        pred_value = pred[0] if isinstance(pred, (list, np.ndarray, pd.Series)) else pred

        if isinstance(pred_value, np.ndarray):
            if pred_value.ndim == 0:
                pred_idx = int(pred_value)
            elif pred_value.ndim == 1 and len(pred_value) > 1:
                pred_idx = int(np.argmax(pred_value))
            else:
                pred_idx = int(np.ravel(pred_value)[0])
        elif isinstance(pred_value, list):
            pred_idx = int(np.argmax(pred_value)) if len(pred_value) > 1 else int(pred_value[0])
        else:
            pred_idx = int(pred_value) if isinstance(pred_value, (int, np.integer, np.floating)) else pred_value

        label = pred_idx
        if label_encoder is not None and isinstance(pred_idx, (int, np.integer)):
            try:
                label = label_encoder.inverse_transform([pred_idx])[0]
            except Exception:
                label = pred_idx

        confidence = None
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(model_input)
            if isinstance(proba, np.ndarray) and proba.ndim == 2:
                confidence = float(np.max(proba[0]))
        elif isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.shape[1] > 1:
            confidence = float(np.max(pred[0]))

        return str(label), confidence, None
    except Exception as exc:
        return None, None, f"Sustainability prediction failed: {exc}"


def prepare_feature_selector_input(feature_model, input_df: pd.DataFrame, reference_df: pd.DataFrame):
    work = input_df.copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            continue
        if reference_df is not None and col in reference_df.columns:
            cat_values = sorted(reference_df[col].dropna().astype(str).unique().tolist())
            mapping = {v: i for i, v in enumerate(cat_values)}
            work[col] = work[col].astype(str).map(mapping).fillna(0)
        else:
            work[col] = 0

    return work.apply(pd.to_numeric, errors="coerce").fillna(0.0)


@st.cache_data(show_spinner=False)
def compute_local_shap(_model_obj, background_df: pd.DataFrame, input_df: pd.DataFrame, max_background: int = 80):
    if not HAS_SHAP:
        return None, "SHAP package not available"
    if _model_obj is None or background_df is None or input_df is None:
        return None, "Model/data missing for SHAP"
    if len(background_df) == 0:
        return None, "Background data empty"

    bg = background_df.sample(n=min(max_background, len(background_df)), random_state=42)
    try:
        if hasattr(_model_obj, "named_steps") and "prep" in _model_obj.named_steps and "model" in _model_obj.named_steps:
            prep = _model_obj.named_steps["prep"]
            est = _model_obj.named_steps["model"]

            bg_t = prep.transform(bg)
            in_t = prep.transform(input_df)
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

            arr = np.array(getattr(shap_values, "values", shap_values))
            if arr.ndim == 3:
                contrib = arr[0, :, 0]
            else:
                contrib = arr[0]
        else:
            explainer = shap.Explainer(_model_obj.predict, bg)
            shap_values = explainer(input_df)
            arr = np.array(shap_values.values)
            feature_names = input_df.columns
            contrib = arr[0] if arr.ndim == 2 else arr[0].mean(axis=-1)

        out = pd.DataFrame({"feature": feature_names, "contribution": contrib})
        out["abs_contribution"] = out["contribution"].abs()
        return out.sort_values("abs_contribution", ascending=False), None
    except Exception as exc:
        return None, f"SHAP computation failed: {exc}"


def forecast_series_arima(series: pd.Series, horizon: int = 8):
    if HAS_ARIMA and len(series) >= 20:
        model = ARIMA(series.values, order=(1, 1, 1)).fit()
        return np.array(model.forecast(steps=horizon)), "ARIMA(1,1,1)"

    rolling = series.rolling(window=min(6, len(series)), min_periods=1).mean()
    slope = (rolling.iloc[-1] - rolling.iloc[max(0, len(rolling) - 6)]) / max(1, min(6, len(rolling) - 1))
    start = rolling.iloc[-1]
    return np.array([start + slope * (i + 1) for i in range(horizon)]), "Trend fallback"


def existing_files(paths):
    return [p for p in paths if p.exists()]


def pretty_feature_name(name):
    text = str(name)
    if text.startswith("num__"):
        text = text.replace("num__", "", 1)
    if text.startswith("cat__"):
        text = text.replace("cat__", "", 1)

    if text.startswith("country_"):
        return f"Country = {text.replace('country_', '', 1)} (one-hot)"

    return text.replace("_", " ")


def normalize_label(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if "high" in text:
        return "high"
    if "medium" in text:
        return "medium"
    if "low" in text:
        return "low"
    return text


def productivity_meaning(label):
    lbl = normalize_label(label)
    if lbl == "high":
        return "High risk class: stress conditions are likely elevated and immediate mitigation is recommended."
    if lbl == "medium":
        return "Medium risk class: conditions are moderate; monitor trend-sensitive variables closely."
    if lbl == "low":
        return "Low risk class: current profile is comparatively stable against modeled disease-pressure factors."
    return f"Predicted class is {label}."


def sustainability_meaning(label):
    lbl = normalize_label(label)
    if lbl == "high":
        return "High sustainability class: profile aligns with stronger resource and environmental balance under the model."
    if lbl == "medium":
        return "Medium sustainability class: baseline is acceptable but can improve through water-quality optimization."
    if lbl == "low":
        return "Low sustainability class: long-term efficiency and ecological stability may degrade without intervention."
    return f"Predicted class is {label}."


def describe_risk_shift(before_label, after_label):
    order = {"low": 0, "medium": 1, "high": 2}
    b = normalize_label(before_label)
    a = normalize_label(after_label)
    if b not in order or a not in order:
        return "Risk-shift direction is not ordinal for these labels."
    delta = order[a] - order[b]
    if delta > 0:
        return "Scenario indicates risk escalation versus baseline."
    if delta < 0:
        return "Scenario indicates risk reduction versus baseline."
    return "Scenario indicates no class-level risk shift versus baseline."


st.set_page_config(page_title="Aquaculture Final Dashboard", page_icon="AQ", layout="wide")
st.title("Aquaculture Final Integrated Dashboard")
st.caption("Productivity + Sustainability + Genomic integration with explainability and forecasting")

productivity_model = load_pickle(PRODUCTIVITY_MODEL_PATH)
sustainability_bundle = load_pickle(SUSTAINABILITY_MODEL_PATH)
feature_selector_model = load_pickle(FEATURE_SELECTOR_PATH)

data_df = load_data()
productivity_metrics_df = safe_read_csv(PRODUCTIVITY_METRICS_PATH)
sustainability_metrics_df = safe_read_csv(SUSTAINABILITY_METRICS_PATH)
genomic_importance_df = safe_read_csv(GENOMIC_IMPORTANCE_PATH)

sustainability_shap_images = existing_files(
    [
        RESULTS_DIR / "sustainability_shap_best_mlp.png",
        RESULTS_DIR / "sustainability_shap_1d_cnn.png",
        RESULTS_DIR / "sustainability_shap_best_1d_cnn.png",
        RESULTS_DIR / "sustainability_shap_gru.png",
        RESULTS_DIR / "sustainability_shap_best_gru.png",
        RESULTS_DIR / "sustainability_shap_lstm.png",
        RESULTS_DIR / "sustainability_shap_best_lstm.png",
        RESULTS_DIR / "sustainability_shap_cnn_lstm.png",
        RESULTS_DIR / "sustainability_shap_best_cnn_lstm.png",
        RESULTS_DIR / "sustainability_shap_mlp.png",
    ]
)

st.sidebar.header("Integration Status")
st.sidebar.write(f"Productivity model: {'Loaded' if productivity_model is not None else 'Missing'}")
st.sidebar.write(f"Sustainability model: {'Loaded' if sustainability_bundle is not None else 'Missing'}")
st.sidebar.write(f"Genomic model: {'Loaded' if feature_selector_model is not None else 'Missing'}")
st.sidebar.write(f"Processed dataset: {'Loaded' if data_df is not None else 'Missing'}")
st.sidebar.write(f"SHAP runtime: {'Available' if HAS_SHAP else 'Not available'}")
st.sidebar.write(f"ARIMA runtime: {'Available' if HAS_ARIMA else 'Fallback mode'}")

with st.sidebar.expander("Paths"):
    st.write(f"Productivity model path: {PRODUCTIVITY_MODEL_PATH}")
    st.write(f"Sustainability model path: {SUSTAINABILITY_MODEL_PATH}")
    st.write(f"Genomic model path: {FEATURE_SELECTOR_PATH}")
    st.write(f"Data path: {DATA_FILE}")

if data_df is None:
    st.error("Processed dataset missing. Run preprocessing first.")
    st.stop()

base = default_input_row(data_df)

tabs = st.tabs(["Predictions", "Explainability", "Forecasting", "Scenario Simulation", "Policy Recommendations"])

with tabs[0]:
    st.subheader("Predictions")
    st.info(
        "This tab performs real-time multi-model inference from one shared farm profile. "
        "Use it to understand immediate model outcomes and confidence under controlled parameter changes."
    )

    with st.expander("How to read this tab"):
        st.markdown(
            """
        1. Productivity output estimates the disease-pressure class from environment plus genomic context.
        2. Sustainability output estimates long-term operational balance and ecological robustness.
        3. Genomic output provides the selected model class from feature-driven genomic signals.
        4. SHAP chart explains why the productivity model gave the current prediction by ranking feature contributions.
        """
        )

    mode = st.selectbox(
        "Profile Mode",
        [
            "Typical (Median)",
            "High-Risk Environmental Stress",
            "Low-Risk Environmental Stress",
        ],
    )

    profile = dict(base)
    if mode == "High-Risk Environmental Stress":
        for key, delta in [
            ("temperature_celsius", 2.0),
            ("water_WaterTemp (C)", 2.0),
            ("water_Salinity (ppt)", 1.5),
            ("precip_mm", -20.0),
        ]:
            if key in profile:
                profile[key] = float(profile[key]) + delta
    elif mode == "Low-Risk Environmental Stress":
        for key, delta in [
            ("temperature_celsius", -1.0),
            ("water_WaterTemp (C)", -1.0),
            ("water_Salinity (ppt)", -0.8),
            ("precip_mm", 15.0),
        ]:
            if key in profile:
                profile[key] = float(profile[key]) + delta

    c1, c2, c3 = st.columns(3)
    temp_delta = c1.slider("Temperature Delta", -5.0, 5.0, 0.0, 0.5)
    rain_delta = c2.slider("Rainfall Delta", -50.0, 50.0, 0.0, 5.0)
    salinity_delta = c3.slider("Salinity Delta", -5.0, 5.0, 0.0, 0.25)

    for key, delta in [
        ("temperature_celsius", temp_delta),
        ("precip_mm", rain_delta),
        ("water_Salinity (ppt)", salinity_delta),
    ]:
        if key in profile:
            profile[key] = float(profile[key]) + float(delta)

    productivity_input = build_aligned_input_df(profile, productivity_model, data_df)
    sustainability_input = build_aligned_input_df(profile, sustainability_bundle if not isinstance(sustainability_bundle, dict) else sustainability_bundle.get("model"), data_df, bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None)
    genomic_input = build_aligned_input_df(profile, feature_selector_model, data_df)

    pcol, scol, gcol = st.columns(3)

    p_label = None
    if productivity_model is not None:
        try:
            p_pred = productivity_model.predict(productivity_input)[0]
            p_label = class_label_from_prediction(p_pred, productivity_model)
            pcol.metric("Productivity/Disease Output", p_label)
            pcol.caption(productivity_meaning(p_label))
        except Exception as exc:
            pcol.warning(f"Productivity prediction failed: {exc}")
    else:
        pcol.info("Productivity model missing")

    s_label, s_conf, s_err = predict_sustainability(sustainability_bundle, sustainability_input)
    if s_err:
        scol.info(s_err)
    else:
        scol.metric("Sustainability Class", s_label)
        scol.caption(sustainability_meaning(s_label))
        if s_conf is not None:
            scol.caption(f"Confidence: {s_conf:.3f}")

    g_pred_label = None
    if feature_selector_model is not None:
        try:
            genomic_ready = prepare_feature_selector_input(feature_selector_model, genomic_input, data_df)
            g_pred = feature_selector_model.predict(genomic_ready)[0]
            g_pred_label = str(g_pred)
            gcol.metric("Genomic Model Output", g_pred_label)
            gcol.caption("Genomic classification inferred from selected high-impact sequence features.")
        except Exception as exc:
            gcol.warning(f"Genomic prediction failed: {exc}")
    else:
        gcol.info("Genomic model missing")

    if HAS_SHAP and productivity_model is not None:
        show_local_shap = st.checkbox("Show local SHAP for productivity", value=True)
        if show_local_shap:
            shap_df, shap_err = compute_local_shap(productivity_model, model_feature_frame(data_df), productivity_input)
            if shap_df is not None:
                top = shap_df.head(10)
                fig, ax = plt.subplots(figsize=(9, 4.5))
                sns.barplot(data=top, y="feature", x="contribution", ax=ax)
                ax.axvline(0.0, color="black", linewidth=1)
                ax.set_title("Local SHAP Contributions")
                st.pyplot(fig)
                st.dataframe(top[["feature", "contribution", "abs_contribution"]], width="stretch")
            else:
                st.info(shap_err)

    st.markdown("### Inference Interpretation")
    if p_label is not None:
        st.write(f"Productivity inference: {productivity_meaning(p_label)}")
    if s_label is not None and not s_err:
        st.write(f"Sustainability inference: {sustainability_meaning(s_label)}")
    if g_pred_label is not None:
        st.write(
            "Genomic inference: class reflects the selector model response to the current feature signature; "
            "track this together with productivity class rather than in isolation."
        )

with tabs[1]:
    st.subheader("Explainability")
    st.info(
        "This tab validates model behavior with benchmark metrics and evidence artifacts. "
        "It supports model trust, selection justification, and technical defense in evaluation."
    )

    with st.expander("What SHAP, feature importance, and benchmark plots mean"):
        st.markdown(
            """
        1. SHAP: signed contribution of each feature to one prediction; higher absolute value means stronger local influence.
        2. Feature importance: average global influence over many samples, useful for ranking key drivers.
        3. Model comparison plot: relative performance among candidate models for final selection.
        4. Efficiency plot: trade-off between quality and operational cost (train time, inference time, RAM).
        """
        )

    st.markdown("### Productivity")
    if productivity_metrics_df is not None:
        st.dataframe(productivity_metrics_df, width="stretch")
    if KHUSHI_PLOT_PATH.exists():
        st.image(str(KHUSHI_PLOT_PATH), caption="Productivity model comparison")

    st.markdown("### Sustainability")
    if sustainability_metrics_df is not None:
        st.dataframe(sustainability_metrics_df, width="stretch")
    for path in existing_files([SUSTAINABILITY_COMPARISON_PATH, SUSTAINABILITY_EFFICIENCY_PATH]):
        st.image(str(path), caption=path.name)
    if sustainability_shap_images:
        st.caption("Sustainability SHAP outputs")
        for path in sustainability_shap_images:
            st.image(str(path), caption=path.name)

    st.markdown("### Genomic")
    if genomic_importance_df is not None:
        st.dataframe(genomic_importance_df, width="stretch")
    for path in existing_files([JANHAVI_COMPARISON_PATH, JANHAVI_EFFICIENCY_PATH]):
        st.image(str(path), caption=path.name)

with tabs[2]:
    st.subheader("Time-Series Forecasting")
    st.info(
        "This tab projects future production trajectory from historical annual production. "
        "ARIMA is used when available; otherwise a trend-based fallback is applied."
    )

    with st.expander("How forecasting is computed"):
        st.markdown(
            """
        1. ARIMA(1,1,1):
           AR term (p=1) captures lag dependence,
           differencing (d=1) removes trend level,
           MA term (q=1) models residual shock carryover.
        2. If ARIMA is unavailable, rolling-trend extrapolation is used for continuity.
        3. Forecast is decision-support, not ground truth; it should be interpreted with scenario and policy tabs.
        """
        )

    if "year" not in data_df.columns or "production" not in data_df.columns:
        st.warning("Forecasting requires year and production columns.")
    else:
        yearly = data_df.groupby("year", as_index=False)["production"].mean().sort_values("year")
        horizon = st.slider("Forecast horizon (years)", 3, 15, 8, 1)
        fc, method = forecast_series_arima(yearly["production"], horizon)
        last_year = int(yearly["year"].max())
        future_years = np.arange(last_year + 1, last_year + 1 + horizon)
        fc_df = pd.DataFrame({"year": future_years, "forecast_production": fc, "method": method})

        fig, ax = plt.subplots(figsize=(10.5, 5))
        ax.plot(yearly["year"], yearly["production"], marker="o", label="Historical")
        ax.plot(fc_df["year"], fc_df["forecast_production"], marker="o", linestyle="--", label=f"Forecast ({method})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Production")
        ax.set_title("Production Forecast")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(fc_df, width="stretch")

        hist_last = float(yearly["production"].iloc[-1])
        fc_last = float(fc_df["forecast_production"].iloc[-1])
        delta = fc_last - hist_last
        sign = "increase" if delta >= 0 else "decrease"
        st.write(
            f"Forecast inference: projected end-horizon production shows a {sign} of {abs(delta):.2f} "
            f"from the latest observed yearly production (method: {method})."
        )

with tabs[3]:
    st.subheader("Scenario Simulation")
    st.info(
        "This tab performs what-if experimentation by perturbing environmental drivers and comparing baseline vs scenario outputs."
    )

    with st.expander("How to explain scenario outcomes"):
        st.markdown(
            """
        1. Baseline uses median environmental conditions from the processed dataset.
        2. Scenario applies user-defined deltas to temperature, rainfall, and salinity.
        3. If output class changes, the direction indicates either risk escalation or mitigation potential.
        """
        )

    sim1, sim2, sim3 = st.columns(3)
    t_plus = sim1.slider("Scenario Temperature Change", -4.0, 4.0, 2.0, 0.5)
    p_plus = sim2.slider("Scenario Rainfall Change", -100.0, 100.0, 0.0, 10.0)
    s_plus = sim3.slider("Scenario Salinity Change", -6.0, 6.0, 0.0, 0.5)

    baseline = dict(base)
    scenario = dict(base)

    if "temperature_celsius" in scenario:
        scenario["temperature_celsius"] = float(scenario["temperature_celsius"]) + t_plus
    if "precip_mm" in scenario:
        scenario["precip_mm"] = float(scenario["precip_mm"]) + p_plus
    if "water_Salinity (ppt)" in scenario:
        scenario["water_Salinity (ppt)"] = float(scenario["water_Salinity (ppt)"]) + s_plus

    base_prod_df = build_aligned_input_df(baseline, productivity_model, data_df)
    scen_prod_df = build_aligned_input_df(scenario, productivity_model, data_df)

    base_sus_df = build_aligned_input_df(baseline, sustainability_bundle if not isinstance(sustainability_bundle, dict) else sustainability_bundle.get("model"), data_df, bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None)
    scen_sus_df = build_aligned_input_df(scenario, sustainability_bundle if not isinstance(sustainability_bundle, dict) else sustainability_bundle.get("model"), data_df, bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None)

    if productivity_model is not None:
        try:
            b_pred = class_label_from_prediction(productivity_model.predict(base_prod_df)[0], productivity_model)
            s_pred = class_label_from_prediction(productivity_model.predict(scen_prod_df)[0], productivity_model)
            st.write(f"Productivity baseline: **{b_pred}**")
            st.write(f"Productivity scenario: **{s_pred}**")
            st.write(describe_risk_shift(b_pred, s_pred))
        except Exception as exc:
            st.warning(f"Productivity simulation failed: {exc}")

    b_sus, _, b_err = predict_sustainability(sustainability_bundle, base_sus_df)
    s_sus, _, s_err = predict_sustainability(sustainability_bundle, scen_sus_df)
    if b_err or s_err:
        st.info(b_err or s_err)
    else:
        st.write(f"Sustainability baseline: **{b_sus}**")
        st.write(f"Sustainability scenario: **{s_sus}**")
        st.write(describe_risk_shift(b_sus, s_sus))

with tabs[4]:
    st.subheader("Policy Recommendations")
    st.info(
        "This tab translates model outcomes into operational guidance. "
        "Recommendations are currently rule-based and aligned with integrated model signals."
    )

    with st.expander("How to explain policy recommendations"):
        st.markdown(
            """
        1. Productivity and sustainability classes trigger risk-prioritized actions.
        2. Environmental parameter checks add preventive controls.
        3. Genomic importance contributes monitoring priorities.
        4. This layer is interpretable now and can be upgraded later to a learned policy optimizer.
        """
        )

    policy_input = build_aligned_input_df(base, productivity_model, data_df)
    prod_label = None
    sus_label = None

    if productivity_model is not None:
        try:
            prod_label = class_label_from_prediction(productivity_model.predict(policy_input)[0], productivity_model)
        except Exception:
            prod_label = None

    sus_df = build_aligned_input_df(base, sustainability_bundle if not isinstance(sustainability_bundle, dict) else sustainability_bundle.get("model"), data_df, bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None)
    sus_label, _, _ = predict_sustainability(sustainability_bundle, sus_df)

    recs = []
    if prod_label and str(prod_label).lower() == "high":
        recs.append("Prioritize stress mitigation: reduce abrupt temperature and salinity shifts.")
    if sus_label and str(sus_label).lower() == "low":
        recs.append("Increase water-quality monitoring frequency and tighten aeration control windows.")

    if "water_pH" in data_df.columns:
        recs.append("Maintain water pH close to historical median to reduce volatility in predicted risk.")
    if "water_Salinity (ppt)" in data_df.columns:
        recs.append("Avoid sudden salinity changes; apply incremental adjustments with daily checks.")
    if "temperature_celsius" in data_df.columns:
        recs.append("Prepare a heat-spike protocol: aeration boost and adaptive feed timing.")

    if genomic_importance_df is not None and len(genomic_importance_df) > 0:
        top_feature = str(genomic_importance_df.iloc[0][genomic_importance_df.columns[0]])
        recs.append(f"Track top genomic driver ({top_feature}) in routine model-driven audits.")

    for idx, rec in enumerate(recs, start=1):
        st.write(f"{idx}. {rec}")

    st.markdown("### System Readiness")
    readiness = pd.DataFrame(
        [
            {"Component": "Productivity model", "Status": "Ready" if productivity_model is not None else "Missing"},
            {"Component": "Sustainability model", "Status": "Ready" if sustainability_bundle is not None else "Missing"},
            {"Component": "Genomic model", "Status": "Ready" if feature_selector_model is not None else "Missing"},
            {"Component": "SHAP local explainability", "Status": "Ready" if HAS_SHAP else "Unavailable"},
            {"Component": "ARIMA forecasting", "Status": "Ready" if HAS_ARIMA else "Fallback active"},
        ]
    )
    st.dataframe(readiness, width="stretch")

st.markdown("---")
st.caption("Aquaculture XAI Project - Fully Integrated Dashboard")
