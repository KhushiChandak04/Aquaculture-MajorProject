"""Aquaculture Intelligence Platform - clean, responsive, research-grade dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from statsmodels.tsa.arima.model import ARIMA

    HAS_ARIMA = True
except Exception:
    ARIMA = None
    HAS_ARIMA = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = BASE_DIR / "data" / "processed" / "final_dataset.csv"

PRODUCTIVITY_MODEL_PATH = MODELS_DIR / "productivity_model.pkl"
SUSTAINABILITY_MODEL_PATH = MODELS_DIR / "sustainability_model.pkl"
FEATURE_SELECTOR_PATH = MODELS_DIR / "feature_selector.pkl"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
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
def load_data() -> pd.DataFrame | None:
    return safe_read_csv(DATA_FILE)


def first_existing(cols: list[str], df: pd.DataFrame) -> str | None:
    for col in cols:
        if col in df.columns:
            return col
    return None


def normalize_label(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if "high" in text:
        return "high"
    if "medium" in text:
        return "medium"
    if "low" in text:
        return "low"
    return text


def default_profile(df: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(df[col].median())
        else:
            mode_vals = df[col].dropna().astype(str).mode()
            row[col] = mode_vals.iloc[0] if len(mode_vals) else "Unknown"

    if "country" in row:
        countries = df["country"].dropna().astype(str).unique().tolist()
        if "India" in countries:
            row["country"] = "India"

    return row


def expected_features(model_obj, bundle: dict | None = None) -> list[str] | None:
    if bundle and isinstance(bundle, dict) and "feature_columns" in bundle:
        return list(bundle["feature_columns"])

    if hasattr(model_obj, "feature_names_in_"):
        return list(model_obj.feature_names_in_)

    if hasattr(model_obj, "named_steps"):
        prep = model_obj.named_steps.get("prep")
        if prep is not None and hasattr(prep, "feature_names_in_"):
            return list(prep.feature_names_in_)

    return None


def build_aligned_input(profile: dict[str, Any], model_obj, reference_df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    features = expected_features(model_obj, bundle=bundle)
    if not features:
        return pd.DataFrame([profile])

    aligned: dict[str, Any] = {}
    for col in features:
        if col in profile:
            aligned[col] = profile[col]
        elif col in reference_df.columns:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                aligned[col] = float(reference_df[col].median())
            else:
                mode_vals = reference_df[col].dropna().astype(str).mode()
                aligned[col] = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
        else:
            aligned[col] = 0.0

    return pd.DataFrame([aligned], columns=features)


def class_label_from_prediction(pred_value, model_obj=None) -> str:
    if isinstance(pred_value, (int, np.integer)) and model_obj is not None:
        try:
            est = model_obj.named_steps.get("model") if hasattr(model_obj, "named_steps") else model_obj
            if hasattr(est, "classes_"):
                classes = list(est.classes_)
                idx = int(pred_value)
                if 0 <= idx < len(classes):
                    return str(classes[idx])
        except Exception:
            pass

    return str(pred_value)


def to_numeric_features(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        if col in reference_df.columns:
            vals = sorted(reference_df[col].dropna().astype(str).unique().tolist())
            mapper = {v: i for i, v in enumerate(vals)}
            out[col] = out[col].astype(str).map(mapper).fillna(0)
        else:
            out[col] = 0
    return out.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def apply_feature_selector(input_df: pd.DataFrame, selector_model, reference_df: pd.DataFrame) -> pd.DataFrame:
    if selector_model is None:
        return input_df

    try:
        if hasattr(selector_model, "transform"):
            transformed = selector_model.transform(to_numeric_features(input_df, reference_df))
            if isinstance(transformed, pd.DataFrame):
                return transformed
            if isinstance(transformed, np.ndarray):
                return pd.DataFrame(transformed)
    except Exception:
        pass

    return input_df


def predict_productivity(model_obj, input_df: pd.DataFrame) -> tuple[str | None, float | None]:
    if model_obj is None:
        return None, None
    try:
        pred = model_obj.predict(input_df)[0]
        label = class_label_from_prediction(pred, model_obj)
        confidence = None
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(input_df)
            if isinstance(proba, np.ndarray) and proba.ndim == 2:
                confidence = float(np.max(proba[0]))
        return label, confidence
    except Exception:
        return None, None


def predict_sustainability(bundle_or_model, input_df: pd.DataFrame) -> tuple[str | None, float | None]:
    if bundle_or_model is None:
        return None, None

    model_obj = bundle_or_model
    label_encoder = None
    preprocessor = None

    if isinstance(bundle_or_model, dict):
        model_obj = bundle_or_model.get("model")
        label_encoder = bundle_or_model.get("label_encoder")
        preprocessor = bundle_or_model.get("preprocessor")

    if model_obj is None:
        return None, None

    try:
        model_input = input_df
        if preprocessor is not None and not (hasattr(model_obj, "named_steps") and "prep" in model_obj.named_steps):
            model_input = preprocessor.transform(input_df)

        if hasattr(model_input, "toarray"):
            model_input = model_input.toarray()

        try:
            pred = model_obj.predict(model_input, verbose=0)
        except TypeError:
            pred = model_obj.predict(model_input)

        pred_value = pred[0] if isinstance(pred, (np.ndarray, list, pd.Series)) else pred
        if isinstance(pred_value, np.ndarray):
            if pred_value.ndim == 1 and len(pred_value) > 1:
                pred_idx = int(np.argmax(pred_value))
            else:
                pred_idx = int(np.ravel(pred_value)[0])
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

        return str(label), confidence
    except Exception:
        return None, None


def predict_genomic_signal(selector_model, input_df: pd.DataFrame, reference_df: pd.DataFrame) -> str | None:
    if selector_model is None:
        return None
    try:
        numeric_df = to_numeric_features(input_df, reference_df)
        pred = selector_model.predict(numeric_df)[0]
        return str(pred)
    except Exception:
        return None


def productivity_score_from_outputs(prod_label: str | None, prod_conf: float | None, sensitivity_shift: float = 0.0) -> float:
    base = {"high": 85.0, "medium": 62.0, "low": 40.0}.get(normalize_label(prod_label), 50.0)
    if prod_conf is not None:
        base = base * 0.75 + float(prod_conf) * 25.0
    # Blend a bounded sensitivity adjustment so slider motion produces visible change
    # even when class labels remain unchanged.
    base = base - min(25.0, max(0.0, float(sensitivity_shift) * 0.25))
    return float(min(100.0, max(0.0, base)))


def sustainability_score_from_outputs(sus_label: str | None, sus_conf: float | None, sensitivity_shift: float = 0.0) -> float:
    base = {"high": 88.0, "medium": 66.0, "low": 42.0}.get(normalize_label(sus_label), 55.0)
    if sus_conf is not None:
        base = base * 0.78 + float(sus_conf) * 22.0
    base = base - min(20.0, max(0.0, float(sensitivity_shift) * 0.18))
    return float(min(100.0, max(0.0, base)))


def status_color(label: str | None) -> tuple[str, str]:
    n = normalize_label(label)
    if n == "high":
        return ("#1f7a3f", "#eaf7ef")
    if n == "medium":
        return ("#8a6d1d", "#fff6df")
    if n == "low":
        return ("#9f2222", "#fdeaea")
    return ("#37506b", "#edf3fb")


def render_label_chip(label: str | None):
    text = str(label) if label is not None else "N/A"
    fg, bg = status_color(label)
    st.markdown(
        f"""
<div style="display:inline-block;padding:0.22rem 0.62rem;border-radius:999px;background:{bg};color:{fg};border:1px solid {fg};font-weight:700;font-size:0.8rem;">{text}</div>
""",
        unsafe_allow_html=True,
    )


def explain_with_shap(model_obj, reference_df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame | None:
    if not HAS_SHAP or model_obj is None or not hasattr(model_obj, "named_steps"):
        return None

    try:
        prep = model_obj.named_steps.get("prep")
        est = model_obj.named_steps.get("model")
        if prep is None or est is None:
            return None

        bg = prep.transform(reference_df.sample(n=min(80, len(reference_df)), random_state=42))
        one = prep.transform(input_df)
        if hasattr(bg, "toarray"):
            bg = bg.toarray()
        if hasattr(one, "toarray"):
            one = one.toarray()

        exp = shap.Explainer(est, bg)
        values = exp(one)
        arr = np.array(values.values)
        if arr.ndim == 3:
            contrib = arr[0, :, 0]
        else:
            contrib = arr[0]

        out = pd.DataFrame({"feature": prep.get_feature_names_out(), "impact": contrib})
        out["impact_abs"] = out["impact"].abs()
        out = out.sort_values("impact_abs", ascending=False).head(10).reset_index(drop=True)
        out["feature"] = out["feature"].astype(str).str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False).str.replace("_", " ", regex=False)
        return out
    except Exception:
        return None


def explain_with_fallback(model_obj, reference_df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame | None:
    if model_obj is None or not hasattr(model_obj, "named_steps"):
        return None

    try:
        prep = model_obj.named_steps.get("prep")
        est = model_obj.named_steps.get("model")
        if prep is None or est is None:
            return None

        base_t = prep.transform(reference_df)
        one_t = prep.transform(input_df)
        if hasattr(base_t, "toarray"):
            base_t = base_t.toarray()
        if hasattr(one_t, "toarray"):
            one_t = one_t.toarray()

        delta = np.ravel(one_t[0] - np.mean(base_t, axis=0))
        if hasattr(est, "feature_importances_"):
            weights = np.array(est.feature_importances_, dtype=float)
            impact = delta * weights
        elif hasattr(est, "coef_"):
            coef = np.array(est.coef_, dtype=float)
            weights = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(np.ravel(coef))
            impact = delta * weights
        else:
            return None

        out = pd.DataFrame({"feature": prep.get_feature_names_out(), "impact": impact})
        out["impact_abs"] = out["impact"].abs()
        out = out.sort_values("impact_abs", ascending=False).head(10).reset_index(drop=True)
        out["feature"] = out["feature"].astype(str).str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False).str.replace("_", " ", regex=False)
        return out
    except Exception:
        return None


def global_feature_importance(model_obj, top_n: int = 10) -> pd.DataFrame | None:
    if model_obj is None or not hasattr(model_obj, "named_steps"):
        return None
    try:
        prep = model_obj.named_steps.get("prep")
        est = model_obj.named_steps.get("model")
        if prep is None or est is None or not hasattr(est, "feature_importances_"):
            return None
        names = prep.get_feature_names_out()
        vals = np.array(est.feature_importances_, dtype=float)
        out = pd.DataFrame({"feature": names, "impact_abs": vals})
        out = out.sort_values("impact_abs", ascending=False).head(top_n).reset_index(drop=True)
        out["impact"] = out["impact_abs"]
        out["feature"] = out["feature"].astype(str).str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False).str.replace("_", " ", regex=False)
        return out
    except Exception:
        return None


def build_forecast(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    if "year" not in df.columns or "production" not in df.columns:
        return None, None

    hist = df.groupby("year", as_index=False)["production"].mean().sort_values("year")
    if len(hist) < 3:
        return None, None

    if HAS_ARIMA and len(hist) >= 20:
        model = ARIMA(hist["production"].values, order=(1, 1, 1)).fit()
        fc_vals = np.array(model.forecast(steps=horizon), dtype=float)
    else:
        tail = hist.tail(min(6, len(hist)))
        x = np.arange(len(tail), dtype=float)
        y = tail["production"].to_numpy(dtype=float)
        slope = np.polyfit(x, y, deg=1)[0]
        start = float(hist["production"].iloc[-1])
        fc_vals = np.array([start + slope * (i + 1) for i in range(horizon)], dtype=float)

    years = np.arange(int(hist["year"].max()) + 1, int(hist["year"].max()) + 1 + horizon)
    fc = pd.DataFrame({"year": years, "production": fc_vals})
    return hist, fc


def policy_recommendations(prod_label: str | None, sus_label: str | None, profile: dict[str, Any]) -> list[tuple[str, str]]:
    recs: list[tuple[str, str]] = []

    p = normalize_label(prod_label)
    s = normalize_label(sus_label)

    if p == "low":
        recs.append(("warning", "Productivity output is low. Review feed timing and water quality stability."))
    elif p == "medium":
        recs.append(("warning", "Productivity output is moderate. Improve consistency in water management windows."))
    else:
        recs.append(("success", "Productivity output is strong. Maintain current operating practices."))

    if s == "low":
        recs.append(("warning", "Sustainability level is low. Reduce stressors and tighten environmental monitoring."))
    elif s == "medium":
        recs.append(("warning", "Sustainability level is medium. Use preventive checks to avoid future decline."))
    else:
        recs.append(("success", "Sustainability level is high. Keep existing control strategy active."))

    ammo_col = first_existing(["water_Ammonia (mg/L)", "ammonia", "water_ammonia"], pd.DataFrame([profile]))
    if ammo_col is not None:
        try:
            if float(profile.get(ammo_col, 0.0)) > 0.5:
                recs.append(("warning", "Reduce ammonia concentration through feed-waste control and water exchange planning."))
        except Exception:
            pass

    do_col = first_existing(["water_DissolvedOxygen (mg/L)", "dissolved_oxygen", "water_DO"], pd.DataFrame([profile]))
    if do_col is not None:
        try:
            if float(profile.get(do_col, 99.0)) < 4.0:
                recs.append(("warning", "Improve oxygen levels with aeration and lower nighttime stress load."))
        except Exception:
            pass

    return recs


def load_results_graphs() -> list[Path]:
    return sorted([p for p in RESULTS_DIR.glob("*.png") if p.is_file()])


st.set_page_config(page_title="Aquaculture Intelligence Platform", page_icon="AQ", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top:1rem;padding-bottom:1.25rem;max-width:1200px;}
.stApp, html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#f4f8fc;}
h1,h2,h3,h4,h5,h6,p,div,label,li,span {color:#102a43;}
[data-testid="stSidebar"] {background:#f6fbff;border-right:1px solid #d9e7f5;}
.header-card {background:linear-gradient(140deg,#ffffff 0%,#eef5fd 100%);border:1px solid #d2e2f2;border-radius:14px;padding:1rem 1.1rem;margin-bottom:0.8rem;}
.section-card {background:#ffffff;border:1px solid #d9e7f5;border-radius:12px;padding:0.85rem 0.95rem;margin-bottom:0.7rem;}
.small-note {color:#365a7c;font-size:0.9rem;}
[data-testid="stHorizontalBlock"] {gap:0.9rem;}
div[data-testid="metric-container"] {background:#ffffff;border:1px solid #d9e7f5;border-radius:10px;padding:0.45rem 0.6rem;}
hr {border-color:#d9e7f5;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header-card">
  <h2 style="margin-bottom:0.2rem;">Aquaculture Intelligence Platform</h2>
  <div style="font-size:1rem;color:#1f4f79;margin-bottom:0.3rem;">Explainable AI for Productivity, Sustainability, and Risk Intelligence</div>
  <div class="small-note">AI-powered decision support for aquaculture farms using climate, water, genomic, and production-linked signals</div>
</div>
""",
    unsafe_allow_html=True,
)

productivity_model = load_pickle(PRODUCTIVITY_MODEL_PATH)
sustainability_bundle = load_pickle(SUSTAINABILITY_MODEL_PATH)
feature_selector_model = load_pickle(FEATURE_SELECTOR_PATH)

data_df = load_data()
results_graphs = load_results_graphs()

st.sidebar.header("System Status")
st.sidebar.write(f"Productivity model: {'Ready' if productivity_model is not None else 'Missing'}")
st.sidebar.write(f"Sustainability model: {'Ready' if sustainability_bundle is not None else 'Missing'}")
st.sidebar.write(f"Feature selector: {'Ready' if feature_selector_model is not None else 'Missing'}")
st.sidebar.write(f"Processed dataset: {'Ready' if data_df is not None else 'Missing'}")

if data_df is None:
    st.error("Processed dataset is missing. Run preprocessing first.")
    st.stop()

base_profile = default_profile(data_df)

if "latest_profile" not in st.session_state:
    st.session_state["latest_profile"] = dict(base_profile)
if "latest_outputs" not in st.session_state:
    st.session_state["latest_outputs"] = {
        "productivity_label": None,
        "productivity_score": None,
        "productivity_conf": None,
        "sustainability_label": None,
        "sustainability_conf": None,
        "genomic_signal": None,
    }

tabs = st.tabs([
    "Overview",
    "Predictions",
    "Explainability",
    "Forecasting",
    "Scenario Simulation",
    "Policy Recommendations",
])

with tabs[0]:
    st.subheader("Overview")
    st.caption("Use the tabs to move from prediction to explainability, forecasting, and policy actions.")
    st.divider()
    left, right = st.columns([1.2, 1.0])

    with left:
        st.markdown(
            """
<div class="section-card">
<b>What this platform does</b><br>
It helps aquaculture teams evaluate productivity and sustainability outcomes from climate, water, genomic, and farm context inputs.
<br><br>
<b>Why it is useful</b><br>
It brings critical farm signals into one screen so teams can respond early and plan operations more confidently.
<br><br>
<b>What inputs it uses</b><br>
Climate, water quality, genomic indicators, and production-linked context from the processed project dataset.
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        k1, k2 = st.columns(2)
        k3, k4 = st.columns(2)
        k1.metric("Productivity optimization", "Active")
        k2.metric("Sustainability scoring", "Active")
        k3.metric("Genomic-aware insights", "Active")
        k4.metric("Decision support", "Interactive")

with tabs[1]:
    st.subheader("Predictions")
    st.caption("Adjust farm inputs on the left and review model outputs on the right.")
    st.divider()
    controls, outputs = st.columns([1.08, 1.0])

    with controls:
        st.markdown("### Inputs")
        species = "Fish"
        st.caption("Species profile: Generic Fish")
        age_months = st.slider("Age (months)", 1, 36, 8, key="pred_age")
        temperature = st.slider("Temperature (deg C)", 10.0, 40.0, 28.0, 0.5, key="pred_temp")
        ph = st.slider("pH", 5.0, 9.5, 7.5, 0.1, key="pred_ph")
        dissolved_oxygen = st.slider("Dissolved Oxygen (mg/L)", 0.0, 12.0, 5.5, 0.1, key="pred_do")
        ammonia = st.slider("Ammonia (mg/L)", 0.0, 2.0, 0.25, 0.05, key="pred_amm")
        rainfall = st.slider("Rainfall / climate input (mm)", -100.0, 300.0, 30.0, 5.0, key="pred_rain")
        genomic_marker = st.text_input("Optional genomic markers", value="", key="pred_genomic")

        countries = sorted(data_df["country"].dropna().astype(str).unique().tolist()) if "country" in data_df.columns else ["India"]
        country = st.selectbox("Country context", countries, index=(countries.index("India") if "India" in countries else 0), key="pred_country")

        st.divider()
        st.markdown("#### Live Input Snapshot")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Temp", f"{temperature:.1f}")
        l2.metric("pH", f"{ph:.1f}")
        l3.metric("DO", f"{dissolved_oxygen:.1f}")
        l4.metric("NH3", f"{ammonia:.2f}")

        live_shift = (
            abs(float(temperature) - 28.0) * 1.4
            + abs(float(ph) - 7.5) * 8.0
            + abs(float(dissolved_oxygen) - 5.5) * 2.2
            + abs(float(ammonia) - 0.25) * 20.0
            + abs(float(rainfall) - 30.0) * 0.12
        )
        st.metric("Input shift index", f"{live_shift:.1f}")

    profile = dict(base_profile)
    profile["country"] = country

    species_col = first_existing(["species", "farm_species"], data_df)
    if species_col:
        profile[species_col] = species

    age_col = first_existing(["age", "age_months", "fish_age"], data_df)
    if age_col:
        profile[age_col] = float(age_months)

    for col in ["temperature_celsius", "water_WaterTemp (C)"]:
        if col in profile:
            profile[col] = float(temperature)
    if "water_pH" in profile:
        profile["water_pH"] = float(ph)
    for col in ["water_DissolvedOxygen (mg/L)", "dissolved_oxygen", "water_DO"]:
        if col in profile:
            profile[col] = float(dissolved_oxygen)
    for col in ["water_Ammonia (mg/L)", "ammonia", "water_ammonia"]:
        if col in profile:
            profile[col] = float(ammonia)
    if "precip_mm" in profile:
        profile["precip_mm"] = float(rainfall)
    if "genomic_Mutation_Flag_global_mean" in profile:
        profile["genomic_Mutation_Flag_global_mean"] = 1.0 if genomic_marker.strip() else 0.0

    st.session_state["latest_profile"] = dict(profile)

    prod_input = build_aligned_input(profile, productivity_model, data_df)
    prod_input_selected = apply_feature_selector(prod_input, feature_selector_model, data_df)

    sust_target = sustainability_bundle.get("model") if isinstance(sustainability_bundle, dict) else sustainability_bundle
    sust_input = build_aligned_input(profile, sust_target, data_df, bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None)

    prod_label, prod_conf = predict_productivity(productivity_model, prod_input_selected)
    sus_label, sus_conf = predict_sustainability(sustainability_bundle, sust_input)
    genomic_signal = predict_genomic_signal(feature_selector_model, prod_input, data_df)

    prod_score = productivity_score_from_outputs(prod_label, prod_conf, sensitivity_shift=live_shift)
    sus_score = sustainability_score_from_outputs(sus_label, sus_conf, sensitivity_shift=live_shift)

    st.session_state["latest_outputs"] = {
        "productivity_label": prod_label,
        "productivity_score": prod_score,
        "productivity_conf": prod_conf,
        "sustainability_label": sus_label,
        "sustainability_score": sus_score,
        "sustainability_conf": sus_conf,
        "genomic_signal": genomic_signal,
    }

    with outputs:
        st.markdown("### Outputs")
        o1, o2, o3 = st.columns(3)
        o1.metric("Productivity Score", f"{prod_score:.1f}/100")
        with o1:
            render_label_chip(prod_label)
        if prod_conf is not None:
            o1.caption(f"Confidence: {prod_conf:.3f}")

        o2.metric("Sustainability Level", str(sus_label) if sus_label is not None else "N/A")
        with o2:
            render_label_chip(sus_label)
        if sus_conf is not None:
            o2.caption(f"Confidence: {sus_conf:.3f}")
        o2.caption(f"Sustainability score: {sus_score:.1f}/100")

        o3.metric("Genomic Signal", str(genomic_signal) if genomic_signal is not None else "N/A")
        o3.caption("Derived from feature selector logic")

        st.divider()
        st.markdown("#### Decision Note")
        if prod_score >= 70:
            st.success("Current profile supports strong productivity conditions.")
        elif prod_score >= 45:
            st.warning("Productivity is moderate. Fine-tune water and climate controls.")
        else:
            st.error("Productivity is constrained. Immediate optimization is recommended.")

with tabs[2]:
    st.subheader("Explainability")
    st.caption("Feature impacts for the current profile are shown below.")
    st.divider()
    active_profile = st.session_state.get("latest_profile", dict(base_profile))
    one_input = build_aligned_input(active_profile, productivity_model, data_df)
    ref_input = build_aligned_input(base_profile, productivity_model, data_df)

    st.markdown("### SHAP Analysis")
    exp_df = explain_with_shap(productivity_model, ref_input, one_input)
    used_shap = exp_df is not None
    if exp_df is None:
        exp_df = explain_with_fallback(productivity_model, ref_input, one_input)

    if exp_df is not None and len(exp_df) > 0 and float(exp_df["impact_abs"].max()) <= 1e-12:
        exp_df = global_feature_importance(productivity_model, top_n=10)
        if exp_df is not None:
            st.info("Local change is very small for current inputs. Showing global feature importance for clarity.")

    if exp_df is not None and len(exp_df) > 0:
        if used_shap:
            st.success("SHAP values are active for this prediction.")
        else:
            st.warning("SHAP not available for this model path. Showing model-based fallback importance.")

        chart_df = exp_df[["feature", "impact_abs"]].set_index("feature")
        st.bar_chart(chart_df)

        top = exp_df.iloc[0]
        direction = "increased" if float(top["impact"]) > 0 else "reduced"
        st.info(f"{top['feature']} most strongly {direction} the current productivity output.")

        top3 = exp_df.head(3)
        lines = []
        for _, row in top3.iterrows():
            action = "increased" if float(row["impact"]) > 0 else "reduced"
            lines.append(f"- {row['feature']} {action} the productivity output")
        st.markdown("\n".join(lines))
    else:
        st.warning("Explainability information is unavailable for the current model format.")

    if results_graphs:
        st.divider()
        st.markdown("### Results Graphs")
        cols = st.columns(2)
        for i, pth in enumerate(results_graphs):
            with cols[i % 2]:
                st.image(str(pth), caption=pth.name, use_container_width=True)

with tabs[3]:
    st.subheader("Forecasting")
    st.caption("Compare historical production with projected trend for the selected horizon.")
    st.divider()
    horizon = st.slider("Forecast horizon (years)", 3, 15, 8, 1, key="forecast_h")
    hist_df, fc_df = build_forecast(data_df, horizon)

    if hist_df is None or fc_df is None:
        st.warning("Forecasting requires year and production columns with sufficient historical data.")
    else:
        display = pd.concat([
            hist_df.assign(series="Historical"),
            fc_df.assign(series="Projected"),
        ], ignore_index=True)
        st.line_chart(display.pivot(index="year", columns="series", values="production"))

        latest = float(hist_df["production"].iloc[-1])
        future = float(fc_df["production"].iloc[-1])
        delta = future - latest
        f1, f2, f3 = st.columns(3)
        f1.metric("Selected horizon", f"{horizon} years")
        f2.metric("Latest production", f"{latest:.2f}")
        f3.metric("Projected end value", f"{future:.2f}", delta=f"{delta:+.2f}")

with tabs[4]:
    st.subheader("Scenario Simulation")
    st.caption("Apply scenario adjustments and compare baseline vs changed outcomes.")
    st.divider()
    baseline = dict(st.session_state.get("latest_profile", dict(base_profile)))
    scenario = dict(baseline)

    c1, c2, c3 = st.columns(3)
    d_temp = c1.slider("Temperature adjustment", -5.0, 5.0, 0.0, 0.5, key="sc_temp")
    d_rain = c2.slider("Rainfall adjustment", -80.0, 80.0, 0.0, 5.0, key="sc_rain")
    d_water = c3.slider("Water quality adjustment", -2.0, 2.0, 0.0, 0.1, key="sc_water")

    scenario_shift = abs(d_temp) * 1.5 + abs(d_rain) * 0.08 + abs(d_water) * 12.0

    if "temperature_celsius" in scenario:
        scenario["temperature_celsius"] = float(scenario["temperature_celsius"]) + float(d_temp)
    if "water_WaterTemp (C)" in scenario:
        scenario["water_WaterTemp (C)"] = float(scenario["water_WaterTemp (C)"]) + float(d_temp)
    if "precip_mm" in scenario:
        scenario["precip_mm"] = float(scenario["precip_mm"]) + float(d_rain)
    if "water_pH" in scenario:
        scenario["water_pH"] = float(scenario["water_pH"]) + float(d_water * 0.2)
    if "water_Salinity (ppt)" in scenario:
        scenario["water_Salinity (ppt)"] = float(scenario["water_Salinity (ppt)"]) + float(d_water * 0.8)

    base_prod = build_aligned_input(baseline, productivity_model, data_df)
    scen_prod = build_aligned_input(scenario, productivity_model, data_df)
    base_prod = apply_feature_selector(base_prod, feature_selector_model, data_df)
    scen_prod = apply_feature_selector(scen_prod, feature_selector_model, data_df)

    sust_target = sustainability_bundle.get("model") if isinstance(sustainability_bundle, dict) else sustainability_bundle
    base_sus = build_aligned_input(baseline, sust_target, data_df, bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None)
    scen_sus = build_aligned_input(scenario, sust_target, data_df, bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None)

    b_prod_label, b_prod_conf = predict_productivity(productivity_model, base_prod)
    s_prod_label, s_prod_conf = predict_productivity(productivity_model, scen_prod)
    b_sus_label, b_sus_conf = predict_sustainability(sustainability_bundle, base_sus)
    s_sus_label, s_sus_conf = predict_sustainability(sustainability_bundle, scen_sus)

    b_score = productivity_score_from_outputs(b_prod_label, b_prod_conf, sensitivity_shift=0.0)
    s_score = productivity_score_from_outputs(s_prod_label, s_prod_conf, sensitivity_shift=scenario_shift)

    b_sus_score = sustainability_score_from_outputs(b_sus_label, b_sus_conf, sensitivity_shift=0.0)
    s_sus_score = sustainability_score_from_outputs(s_sus_label, s_sus_conf, sensitivity_shift=scenario_shift)

    st.markdown("### Baseline vs Scenario")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Baseline productivity", f"{b_score:.1f}/100")
        render_label_chip(b_prod_label)
        st.metric("Baseline sustainability", str(b_sus_label) if b_sus_label else "N/A")
        render_label_chip(b_sus_label)
        st.caption(f"Sustainability score: {b_sus_score:.1f}/100")

    with r2:
        st.metric("Scenario productivity", f"{s_score:.1f}/100", delta=f"{(s_score - b_score):+.1f}")
        render_label_chip(s_prod_label)
        st.metric("Scenario sustainability", str(s_sus_label) if s_sus_label else "N/A")
        render_label_chip(s_sus_label)
        st.caption(f"Sustainability score: {s_sus_score:.1f}/100")

    d1, d2, d3 = st.columns(3)
    d1.metric("Productivity confidence delta", "N/A" if (b_prod_conf is None or s_prod_conf is None) else f"{(s_prod_conf - b_prod_conf):+.3f}")
    d2.metric("Sustainability confidence delta", "N/A" if (b_sus_conf is None or s_sus_conf is None) else f"{(s_sus_conf - b_sus_conf):+.3f}")
    d3.metric("Scenario shift index", f"{scenario_shift:.1f}")

with tabs[5]:
    st.subheader("Policy Recommendations")
    st.caption("Actionable recommendations generated from current prediction outcomes.")
    st.divider()
    latest_profile = st.session_state.get("latest_profile", dict(base_profile))
    latest = st.session_state.get("latest_outputs", {})

    recs = policy_recommendations(
        latest.get("productivity_label"),
        latest.get("sustainability_label"),
        latest_profile,
    )

    for level, text in recs:
        if level == "warning":
            st.warning(text)
        else:
            st.success(text)

    s1, s2, s3 = st.columns(3)
    s1.metric("Productivity score", f"{float(latest.get('productivity_score') or 0.0):.1f}/100")
    s2.metric("Sustainability", str(latest.get("sustainability_label") or "N/A"))
    s3.metric("Genomic signal", str(latest.get("genomic_signal") or "N/A"))
    st.caption(f"Sustainability score: {float(latest.get('sustainability_score') or 0.0):.1f}/100")

st.markdown("---")
st.caption("Built using Explainable AI for Aquaculture Intelligence")
