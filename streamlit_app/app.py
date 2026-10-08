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
XAI_EVIDENCE_REPORT_PATH = RESULTS_DIR / "xai_evidence_report.md"
RESULT_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PRIMARY_SPIRAL_FILES = {
    "productivity_spiral_all_models.png",
    "sustainability_spiral_net_metrics.png",
    "genomic_spiral_all_models.png",
}

PRODUCTIVITY_MODEL_PATH = MODELS_DIR / "productivity_model.pkl"
SUSTAINABILITY_MODEL_PATH = MODELS_DIR / "sustainability_model.pkl"
FEATURE_SELECTOR_PATH = MODELS_DIR / "feature_selector.pkl"
DEPLOYMENT_URL = "https://khushichandak04-aquaculture-majorprojec-streamlit-appapp-gsbdkq.streamlit.app/"
WATER_FEATURES = [
    "water_Salinity (ppt)",
    "water_pH",
    "water_SecchiDepth (m)",
    "water_WaterDepth (m)",
    "water_WaterTemp (C)",
    "water_AirTemp (C)",
]
WATER_LABELS = {
    "water_Salinity (ppt)": "Salinity (scaled)",
    "water_pH": "pH (scaled)",
    "water_SecchiDepth (m)": "Secchi depth (scaled)",
    "water_WaterDepth (m)": "Water depth (scaled)",
    "water_WaterTemp (C)": "Water temperature (scaled)",
    "water_AirTemp (C)": "Air temperature (scaled)",
}


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def file_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return 0, 0


@st.cache_resource
def load_pickle(path: Path, modified_ns: int, size_bytes: int):
    del modified_ns, size_bytes
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_data
def load_data(modified_ns: int, size_bytes: int) -> pd.DataFrame | None:
    del modified_ns, size_bytes
    return safe_read_csv(DATA_FILE)


@st.cache_data
def load_markdown_file(path: Path, modified_ns: int, size_bytes: int) -> str | None:
    del modified_ns, size_bytes
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


@st.cache_data
def load_text_file(path: Path, modified_ns: int, size_bytes: int) -> str | None:
    del modified_ns, size_bytes
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


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
        elif col == "decade" and "year" in profile:
            aligned[col] = (float(profile["year"]) // 10) * 10
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


def model_response_delta(model_or_bundle, base_input: pd.DataFrame, scenario_input: pd.DataFrame) -> float | None:
    """Return a model-derived response delta from transformed feature movement."""
    if model_or_bundle is None:
        return None

    model_obj = model_or_bundle
    preprocessor = None
    estimator = None

    if isinstance(model_or_bundle, dict):
        model_obj = model_or_bundle.get("model")
        preprocessor = model_or_bundle.get("preprocessor")

    if model_obj is None:
        return None

    try:
        if hasattr(model_obj, "named_steps"):
            preprocessor = model_obj.named_steps.get("prep", preprocessor)
            estimator = model_obj.named_steps.get("model")
        else:
            estimator = model_obj

        xb = base_input
        xs = scenario_input
        if preprocessor is not None:
            xb = preprocessor.transform(base_input)
            xs = preprocessor.transform(scenario_input)

        if hasattr(xb, "toarray"):
            xb = xb.toarray()
        if hasattr(xs, "toarray"):
            xs = xs.toarray()

        xb = np.asarray(xb, dtype=float)
        xs = np.asarray(xs, dtype=float)
        if xb.ndim == 2:
            xb = xb[0]
        if xs.ndim == 2:
            xs = xs[0]

        delta = np.ravel(xs - xb)
        if not np.any(delta):
            return 0.0

        if estimator is not None and hasattr(estimator, "feature_importances_"):
            w = np.asarray(estimator.feature_importances_, dtype=float)
            m = min(len(delta), len(w))
            return float(np.sum(np.abs(delta[:m]) * np.abs(w[:m])))

        if estimator is not None and hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            w = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(np.ravel(coef))
            m = min(len(delta), len(w))
            return float(np.sum(np.abs(delta[:m]) * np.abs(w[:m])))

        return float(np.sum(np.abs(delta)))
    except Exception:
        return None


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


def aggregate_feature_impacts(feature_names: list[str], impacts: np.ndarray, top_n: int = 10, base_features: list[str] | None = None) -> pd.DataFrame:
    """Aggregate expanded transformed features back to their original base feature names."""

    base_features = list(base_features or [])

    def base_name(name: str) -> str:
        text = str(name)
        raw = text
        if "__" in raw:
            raw = raw.split("__", 1)[1]

        if base_features:
            if raw in base_features:
                return raw
            # Prefer longest match to support feature names containing underscores.
            matches = [bf for bf in base_features if raw.startswith(f"{bf}_")]
            if matches:
                return sorted(matches, key=len, reverse=True)[0]

        # Heuristic fallback when original feature list is unavailable.
        return raw.split("_", 1)[0] if "_" in raw else raw

    df = pd.DataFrame({"feature": [base_name(n) for n in feature_names], "impact": np.asarray(impacts, dtype=float)})
    grouped = df.groupby("feature", as_index=False).agg(impact=("impact", "sum"))
    grouped["impact_abs"] = grouped["impact"].abs()
    grouped = grouped.sort_values("impact_abs", ascending=False).head(top_n).reset_index(drop=True)
    grouped["feature"] = grouped["feature"].astype(str).str.replace("_", " ", regex=False)
    return grouped


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

        return aggregate_feature_impacts(
            list(prep.get_feature_names_out()),
            np.asarray(contrib, dtype=float),
            top_n=10,
            base_features=list(getattr(prep, "feature_names_in_", [])),
        )
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

        return aggregate_feature_impacts(
            list(prep.get_feature_names_out()),
            np.asarray(impact, dtype=float),
            top_n=10,
            base_features=list(getattr(prep, "feature_names_in_", [])),
        )
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
        return aggregate_feature_impacts(
            list(names),
            np.asarray(vals, dtype=float),
            top_n=top_n,
            base_features=list(getattr(prep, "feature_names_in_", [])),
        )
    except Exception:
        return None


def build_aligned_reference_frame(
    model_obj,
    reference_df: pd.DataFrame,
    bundle: dict | None = None,
    max_rows: int = 240,
) -> pd.DataFrame:
    features = expected_features(model_obj, bundle=bundle)
    if not features:
        out = reference_df.copy()
    else:
        aligned: dict[str, Any] = {}
        for col in features:
            if col in reference_df.columns:
                aligned[col] = reference_df[col]
            elif col == "decade" and "year" in reference_df.columns:
                aligned[col] = (pd.to_numeric(reference_df["year"], errors="coerce") // 10) * 10
            else:
                aligned[col] = 0.0
        out = pd.DataFrame(aligned, columns=features)

    out = out.dropna(axis=0, how="all")
    if len(out) == 0:
        return pd.DataFrame(columns=list(features or reference_df.columns))
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=42)
    return out.reset_index(drop=True)


def explain_pipeline_track(
    model_obj,
    reference_df: pd.DataFrame,
    input_df: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame | None, str]:
    exp_df = explain_with_shap(model_obj, reference_df, input_df)
    mode = "shap" if exp_df is not None and len(exp_df) > 0 else "none"

    if exp_df is None or len(exp_df) == 0:
        exp_df = explain_with_fallback(model_obj, reference_df, input_df)
        mode = "fallback" if exp_df is not None and len(exp_df) > 0 else "none"

    if exp_df is not None and len(exp_df) > 0 and float(exp_df["impact_abs"].max()) <= 1e-5:
        gdf = global_feature_importance(model_obj, top_n=top_n)
        if gdf is not None and len(gdf) > 0:
            exp_df = gdf
            mode = "global"

    if exp_df is not None and len(exp_df) > 0:
        exp_df = exp_df.head(top_n).reset_index(drop=True)

    return exp_df, mode


def resolve_estimator(model_obj):
    if model_obj is None:
        return None
    if hasattr(model_obj, "named_steps"):
        return model_obj.named_steps.get("model", model_obj)
    return model_obj


def friendly_model_name(model_obj) -> str:
    est = resolve_estimator(model_obj)
    if est is None:
        return "Unknown"

    name = est.__class__.__name__
    mapping = {
        "XGBClassifier": "XGBoost",
        "LGBMClassifier": "LightGBM",
        "Sequential": "MLP",
        "HistGradientBoostingClassifier": "HistGB",
        "ExtraTreesClassifier": "ExtraTrees",
        "RandomForestClassifier": "RandomForest",
        "LogisticRegression": "LogisticRegression",
    }
    return mapping.get(name, name)


def explain_sustainability_track(
    bundle_or_model,
    reference_df: pd.DataFrame,
    input_df: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame | None, str]:
    model_obj = bundle_or_model
    preprocessor = None

    if isinstance(bundle_or_model, dict):
        model_obj = bundle_or_model.get("model")
        preprocessor = bundle_or_model.get("preprocessor")

    if model_obj is None:
        return None, "none"

    # If the model is a sklearn pipeline, reuse the standard SHAP/fallback path.
    if hasattr(model_obj, "named_steps"):
        return explain_pipeline_track(model_obj, reference_df, input_df, top_n=top_n)

    # Keras bundle path (current sustainability best-model artifact).
    if preprocessor is None:
        return None, "none"

    try:
        bg_raw = reference_df.sample(n=min(140, len(reference_df)), random_state=42)
        bg_t = preprocessor.transform(bg_raw)
        one_t = preprocessor.transform(input_df)

        if hasattr(bg_t, "toarray"):
            bg_t = bg_t.toarray()
        if hasattr(one_t, "toarray"):
            one_t = one_t.toarray()

        bg_t = np.asarray(bg_t, dtype=float)
        one_t = np.asarray(one_t, dtype=float)
        if bg_t.ndim == 1:
            bg_t = bg_t.reshape(1, -1)
        if one_t.ndim == 1:
            one_t = one_t.reshape(1, -1)
    except Exception:
        return None, "none"

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(one_t.shape[1])]

    base_features = list(getattr(preprocessor, "feature_names_in_", []))

    def predict_fn(x):
        x_arr = np.asarray(x, dtype=float)
        try:
            return model_obj.predict(x_arr, verbose=0)
        except TypeError:
            return model_obj.predict(x_arr)

    if HAS_SHAP:
        try:
            # Primary path: direct SHAP explainer for tabular MLP outputs.
            explainer = shap.Explainer(predict_fn, bg_t)
            values = explainer(one_t)
            arr = np.asarray(values.values, dtype=float)

            pred = np.asarray(predict_fn(one_t), dtype=float)
            cls_idx = int(np.argmax(pred[0])) if pred.ndim == 2 and pred.shape[1] > 1 else 0

            if arr.ndim == 3:
                cls_idx = int(min(max(cls_idx, 0), arr.shape[2] - 1))
                impact = arr[0, :, cls_idx]
            elif arr.ndim == 2:
                impact = arr[0]
            else:
                impact = np.ravel(arr)

            m = min(len(impact), len(feature_names), one_t.shape[1])
            exp_df = aggregate_feature_impacts(
                feature_names[:m],
                np.asarray(impact[:m], dtype=float),
                top_n=top_n,
                base_features=base_features,
            )
            return exp_df, "shap"
        except Exception:
            pass

    try:
        delta = np.ravel(one_t[0] - np.mean(bg_t, axis=0))
        weights = None

        if hasattr(model_obj, "get_weights"):
            ws = model_obj.get_weights()
            if ws:
                first = np.asarray(ws[0], dtype=float)
                if first.ndim == 2:
                    weights = np.mean(np.abs(first), axis=1)

        if weights is not None:
            m = min(len(delta), len(weights), len(feature_names), one_t.shape[1])
            impact = delta[:m] * weights[:m]
        else:
            m = min(len(delta), len(feature_names), one_t.shape[1])
            impact = delta[:m]

        exp_df = aggregate_feature_impacts(
            feature_names[:m],
            np.asarray(impact, dtype=float),
            top_n=top_n,
            base_features=base_features,
        )
        return exp_df, "fallback"
    except Exception:
        return None, "none"


def build_profile_frame(profile: dict[str, Any], reference_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for col in cols:
        if col in profile:
            row[col] = profile[col]
        elif col in reference_df.columns:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                row[col] = float(reference_df[col].median())
            else:
                mode_vals = reference_df[col].dropna().astype(str).mode()
                row[col] = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
        else:
            row[col] = 0.0
    return pd.DataFrame([row], columns=cols)


def direct_impact_dataframe(feature_names: list[str], impacts: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"feature": [str(x) for x in feature_names], "impact": np.asarray(impacts, dtype=float)})
    df["impact_abs"] = df["impact"].abs()
    df = df.sort_values("impact_abs", ascending=False).head(top_n).reset_index(drop=True)
    df["feature"] = df["feature"].str.replace("_", " ", regex=False)
    return df


def explain_genomic_track(
    model_obj,
    reference_df: pd.DataFrame,
    input_df: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame | None, str]:
    if model_obj is None or reference_df is None or input_df is None or reference_df.empty or input_df.empty:
        return None, "none"

    try:
        ref_num = to_numeric_features(reference_df.copy(), reference_df)
        one_num = to_numeric_features(input_df.copy(), reference_df)

        # Keep genomic explainability strictly aligned to the trained model schema.
        model_features = list(getattr(model_obj, "feature_names_in_", []))
        if model_features:
            for col in model_features:
                if col not in ref_num.columns:
                    ref_num[col] = 0.0
                if col not in one_num.columns:
                    one_num[col] = 0.0
            ref_num = ref_num[model_features]
            one_num = one_num[model_features]
    except Exception:
        return None, "none"

    def resolve_class_index() -> int:
        cls_idx = 0
        try:
            pred_label = model_obj.predict(one_num)[0]
            classes = list(getattr(model_obj, "classes_", []))
            if pred_label in classes:
                cls_idx = classes.index(pred_label)
            elif hasattr(model_obj, "predict_proba"):
                cls_idx = int(np.argmax(model_obj.predict_proba(one_num)[0]))
        except Exception:
            cls_idx = 0
        return int(max(cls_idx, 0))

    if HAS_SHAP:
        try:
            # Prefer Tree SHAP for HistGB and related tree classifiers.
            tree_exp = shap.TreeExplainer(model_obj)
            values = tree_exp.shap_values(one_num)

            if isinstance(values, list):
                cls_idx = resolve_class_index()
                cls_idx = int(min(max(cls_idx, 0), len(values) - 1))
                impact = np.ravel(np.asarray(values[cls_idx], dtype=float))[0 : one_num.shape[1]]
            else:
                arr = np.asarray(values, dtype=float)
                if arr.ndim == 3:
                    cls_idx = resolve_class_index()
                    cls_idx = int(min(max(cls_idx, 0), arr.shape[2] - 1))
                    impact = arr[0, :, cls_idx]
                elif arr.ndim == 2:
                    impact = arr[0]
                else:
                    impact = np.ravel(arr)[0 : one_num.shape[1]]

            m = min(len(impact), one_num.shape[1])
            return (
                direct_impact_dataframe(list(one_num.columns[:m]), np.asarray(impact[:m], dtype=float), top_n=top_n),
                "shap",
            )
        except Exception:
            try:
                bg = ref_num.sample(n=min(120, len(ref_num)), random_state=42)
                predict_fn = model_obj.predict_proba if hasattr(model_obj, "predict_proba") else model_obj.predict
                explainer = shap.Explainer(predict_fn, bg)
                values = explainer(one_num)
                arr = np.asarray(values.values, dtype=float)

                if arr.ndim == 3:
                    cls_idx = resolve_class_index()
                    cls_idx = int(min(max(cls_idx, 0), arr.shape[2] - 1))
                    impact = arr[0, :, cls_idx]
                elif arr.ndim == 2:
                    impact = arr[0]
                else:
                    impact = np.ravel(arr)[0 : one_num.shape[1]]

                m = min(len(impact), one_num.shape[1])
                return (
                    direct_impact_dataframe(
                        list(one_num.columns[:m]),
                        np.asarray(impact[:m], dtype=float),
                        top_n=top_n,
                    ),
                    "shap",
                )
            except Exception:
                pass

    try:
        delta = np.ravel(one_num.iloc[0].to_numpy(dtype=float) - ref_num.mean(axis=0).to_numpy(dtype=float))
        if hasattr(model_obj, "feature_importances_"):
            w = np.asarray(model_obj.feature_importances_, dtype=float)
            m = min(len(delta), len(w), one_num.shape[1])
            impact = delta[:m] * w[:m]
        elif hasattr(model_obj, "coef_"):
            coef = np.asarray(model_obj.coef_, dtype=float)
            w = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(np.ravel(coef))
            m = min(len(delta), len(w), one_num.shape[1])
            impact = delta[:m] * w[:m]
        else:
            m = min(len(delta), one_num.shape[1])
            impact = delta[:m]

        return direct_impact_dataframe(list(one_num.columns[:m]), np.asarray(impact, dtype=float), top_n=top_n), "fallback"
    except Exception:
        return None, "none"


def render_explainability_results(exp_df: pd.DataFrame | None, mode: str, unavailable_msg: str) -> None:
    if exp_df is None or len(exp_df) == 0:
        st.warning(unavailable_msg)
        return

    if mode == "shap":
        st.success("SHAP values are active for this model.")
    elif mode == "global":
        st.info("Local signal is very small for current inputs. Showing global feature influence.")
    else:
        st.warning("SHAP path unavailable for this model path. Showing model-based fallback influence.")

    chart_df = exp_df[["feature", "impact_abs"]].set_index("feature")
    st.bar_chart(chart_df)

    show_df = exp_df[["feature", "impact", "impact_abs"]].copy()
    show_df["impact"] = show_df["impact"].map(lambda x: f"{float(x):+.6f}")
    show_df["impact_abs"] = show_df["impact_abs"].map(lambda x: f"{float(x):.6f}")
    st.dataframe(show_df, width="stretch", hide_index=True)

    top = exp_df.iloc[0]
    if mode == "global":
        st.info(f"Most influential parameter (global): {top['feature']} (importance={float(top['impact_abs']):.6f}).")
    else:
        direction = "increased" if float(top["impact"]) > 0 else "reduced"
        st.info(
            f"Most influential parameter: {top['feature']} ({direction} the current output; |impact|={float(top['impact_abs']):.6f})."
        )


def render_architecture_diagram(input_text: str, processing_text: str, model_text: str, output_text: str) -> None:
    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0.3rem 0 0.8rem 0;">
  <div style="padding:0.5rem 0.7rem;border:1px solid #c8d9ea;border-radius:10px;background:#f9fcff;min-width:180px;"><b>Input</b><br>{input_text}</div>
  <div style="font-size:1.2rem;color:#4b6b88;">→</div>
  <div style="padding:0.5rem 0.7rem;border:1px solid #c8d9ea;border-radius:10px;background:#f9fcff;min-width:180px;"><b>Processing</b><br>{processing_text}</div>
  <div style="font-size:1.2rem;color:#4b6b88;">→</div>
  <div style="padding:0.5rem 0.7rem;border:1px solid #c8d9ea;border-radius:10px;background:#f9fcff;min-width:180px;"><b>Model</b><br>{model_text}</div>
  <div style="font-size:1.2rem;color:#4b6b88;">→</div>
  <div style="padding:0.5rem 0.7rem;border:1px solid #c8d9ea;border-radius:10px;background:#f9fcff;min-width:180px;"><b>Output</b><br>{output_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )


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

    recs.append(("info", "Official water measurements are period-level covariates and are not country-specific."))

    return recs


def load_results_graphs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted([p for p in RESULTS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in RESULT_IMAGE_EXTS])


def filter_primary_spiral_graphs(graphs: list[Path]) -> list[Path]:
    filtered: list[Path] = []
    for p in graphs:
        name = p.name.lower()
        if "spiral" not in name or name in PRIMARY_SPIRAL_FILES:
            filtered.append(p)
    return filtered


def count_total_result_graphs() -> int:
    if not RESULTS_DIR.exists():
        return 0
    return sum(1 for p in RESULTS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in RESULT_IMAGE_EXTS)


def load_result_artifacts() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    exts = {".csv", ".md", ".txt"}
    return sorted([p for p in RESULTS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def group_result_graphs(graphs: list[Path]) -> dict[str, list[Path]]:
    groups = {
        "Spiral Net Metrics": [],
        "Train vs Test Curves": [],
        "Productivity Visuals": [],
        "Sustainability Visuals": [],
        "Genomic Visuals": [],
        "Other Visuals": [],
    }

    for p in graphs:
        name = p.name.lower()
        if "spiral" in name:
            groups["Spiral Net Metrics"].append(p)
        elif "train_vs_test" in name or "train_val" in name or "train_trial" in name:
            if name == "all_tracks_train_vs_test_accuracy.png":
                groups["Train vs Test Curves"].append(p)
            continue
        elif name.startswith("khushi_") or "productivity" in name:
            groups["Productivity Visuals"].append(p)
        elif name.startswith("sustainability"):
            groups["Sustainability Visuals"].append(p)
        elif name.startswith("janhavi_") or "genomic" in name:
            groups["Genomic Visuals"].append(p)
        else:
            groups["Other Visuals"].append(p)

    return {k: v for k, v in groups.items() if v}


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
    <div style="font-size:1rem;color:#1f4f79;margin-bottom:0.3rem;">Explainable decision support for productivity and sustainability</div>
    <div class="small-note">Current prediction inputs: country, year, and six time-bucket water-quality measurements</div>
</div>
""",
    unsafe_allow_html=True,
)

st.link_button("Open deployed app", DEPLOYMENT_URL)

productivity_model = load_pickle(PRODUCTIVITY_MODEL_PATH, *file_signature(PRODUCTIVITY_MODEL_PATH))
sustainability_bundle = load_pickle(SUSTAINABILITY_MODEL_PATH, *file_signature(SUSTAINABILITY_MODEL_PATH))
feature_selector_model = load_pickle(FEATURE_SELECTOR_PATH, *file_signature(FEATURE_SELECTOR_PATH))

data_df = load_data(*file_signature(DATA_FILE))
results_graphs = filter_primary_spiral_graphs(load_results_graphs())
total_result_graphs = len(results_graphs)
result_artifacts = load_result_artifacts()
xai_evidence_md = load_markdown_file(XAI_EVIDENCE_REPORT_PATH, *file_signature(XAI_EVIDENCE_REPORT_PATH))
psg_metrics_df = safe_read_csv(RESULTS_DIR / "psg_combined_metrics.csv")

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
    "Results Gallery",
    "Architecture",
    "Forecasting",
    "Scenario Simulation",
    "Policy Recommendations",
])

with tabs[0]:
    st.subheader("Overview")
    st.caption("Current models, cleaned dataset, and generated result metrics.")
    st.divider()
    left, right = st.columns([1.2, 1.0])

    with left:
        st.markdown(
            """
<div class="section-card">
<b>Current modeling table</b><br>
11,657 production records with country, year, and six period-level water-quality features.
<br><br>
<b>Climate and genomic data</b><br>
Climate years do not overlap the production period. Genomic samples have no production-compatible join key, so both are excluded from farm-level predictions.
<br><br>
<b>Water data limitation</b><br>
Water measurements are joined by period and are not country-specific in the official source.
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        k1, k2 = st.columns(2)
        k3, k4 = st.columns(2)
        k1.metric("Productivity model", friendly_model_name(productivity_model))
        k2.metric("Sustainability model", friendly_model_name(sustainability_bundle))
        k3.metric("Dataset records", f"{len(data_df):,}")
        k4.metric("Dataset columns", str(len(data_df.columns)))
        if psg_metrics_df is not None and len(psg_metrics_df):
            st.metric("PSG held-out accuracy", f"{float(psg_metrics_df.iloc[0]['accuracy']):.4f}")
        st.caption("Genomic samples are analyzed separately; there is no source key for farm-level genomic predictions.")

with tabs[1]:
    st.subheader("Predictions")
    st.caption("Adjust country, year, and source-scaled water features used by the current model.")
    st.divider()
    controls, outputs = st.columns([1.08, 1.0])

    with controls:
        st.markdown("### Inputs")
        countries = sorted(data_df["country"].dropna().astype(str).unique().tolist()) if "country" in data_df.columns else ["India"]
        country = st.selectbox("Country context", countries, index=(countries.index("India") if "India" in countries else 0), key="pred_country")
        year_min = int(pd.to_numeric(data_df["year"], errors="coerce").min())
        year_max = int(pd.to_numeric(data_df["year"], errors="coerce").max())
        year_default = int(pd.to_numeric(data_df["year"], errors="coerce").median())
        year = st.slider("Production year", year_min, year_max, year_default, key="pred_year")

        st.divider()
        st.markdown("#### Water features")
        st.caption("Values use the standardized scale stored by preprocessing; the source has no country key.")
        water_profile = {}
        shift_terms = []
        for column in WATER_FEATURES:
            values = pd.to_numeric(data_df[column], errors="coerce").dropna()
            low = float(values.min())
            high = float(values.max())
            center = float(values.median())
            spread = float(values.std(ddof=0))
            if low == high:
                low -= 0.5
                high += 0.5
            step = max((high - low) / 100.0, 0.01)
            value = st.slider(
                WATER_LABELS[column],
                min_value=low,
                max_value=high,
                value=min(max(center, low), high),
                step=step,
                key=f"pred_{column}",
            )
            water_profile[column] = float(value)
            shift_terms.append(abs(float(value) - center) / spread if spread > 0 else 0.0)
        live_shift = float(np.mean(shift_terms)) if shift_terms else 0.0
        st.metric("Mean feature shift (standard deviations)", f"{live_shift:.2f}")

    profile = dict(base_profile)
    profile["country"] = country
    profile["year"] = int(year)
    profile.update(water_profile)

    st.session_state["latest_profile"] = dict(profile)

    prod_input = build_aligned_input(profile, productivity_model, data_df)
    prod_input_selected = apply_feature_selector(prod_input, feature_selector_model, data_df)
    genomic_input = build_aligned_input(profile, feature_selector_model, data_df)

    sust_target = sustainability_bundle.get("model") if isinstance(sustainability_bundle, dict) else sustainability_bundle
    sust_input = build_aligned_input(profile, sust_target, data_df, bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None)

    prod_label, prod_conf = predict_productivity(productivity_model, prod_input_selected)
    sus_label, sus_conf = predict_sustainability(sustainability_bundle, sust_input)
    genomic_signal = predict_genomic_signal(feature_selector_model, genomic_input, data_df)

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

        o3.metric("Auxiliary class signal", str(genomic_signal) if genomic_signal is not None else "N/A")
        o3.caption("Context model only; no genomic-to-farm key exists in the source data.")

        st.divider()
        st.markdown("#### Decision Note")
        if prod_score >= 70:
            st.success("Current profile supports strong productivity conditions.")
        elif prod_score >= 45:
            st.warning("Productivity is moderate. Review the production context and period-level water features.")
        else:
            st.error("Productivity is constrained. Immediate optimization is recommended.")

with tabs[2]:
    st.subheader("Explainability")
    st.caption("Simple model-level SHAP view: one tab per best model, plus graph coverage status.")
    st.divider()

    st.markdown("### SHAP Analysis")
    active_profile = st.session_state.get("latest_profile", dict(base_profile))
    top_n = 10

    prod_input = build_aligned_input(active_profile, productivity_model, data_df)
    prod_ref = build_aligned_reference_frame(productivity_model, data_df)
    prod_exp, prod_mode = explain_pipeline_track(productivity_model, prod_ref, prod_input, top_n=top_n)

    sust_target = sustainability_bundle.get("model") if isinstance(sustainability_bundle, dict) else sustainability_bundle
    sust_input = build_aligned_input(
        active_profile,
        sust_target,
        data_df,
        bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None,
    )
    sust_ref = build_aligned_reference_frame(
        sust_target,
        data_df,
        bundle=sustainability_bundle if isinstance(sustainability_bundle, dict) else None,
    )
    sust_exp, sust_mode = explain_sustainability_track(sustainability_bundle, sust_ref, sust_input, top_n=top_n)

    gen_ref = build_aligned_reference_frame(feature_selector_model, data_df)
    gen_input = build_aligned_input(active_profile, feature_selector_model, data_df)
    gen_exp, gen_mode = explain_genomic_track(feature_selector_model, gen_ref, gen_input, top_n=top_n)

    prod_name = friendly_model_name(productivity_model)
    sust_name = friendly_model_name(sust_target)
    gen_name = friendly_model_name(feature_selector_model)

    track_tabs = st.tabs([
        f"Productivity ({prod_name})",
        f"Sustainability ({sust_name})",
        f"Genomic ({gen_name})",
    ])

    with track_tabs[0]:
        render_explainability_results(
            prod_exp,
            prod_mode,
            "Productivity explainability is unavailable for the current model format.",
        )

    with track_tabs[1]:
        render_explainability_results(
            sust_exp,
            sust_mode,
            "Sustainability explainability is unavailable for the current model format.",
        )

    with track_tabs[2]:
        render_explainability_results(
            gen_exp,
            gen_mode,
            "Genomic explainability is unavailable for the current model format.",
        )

    if results_graphs:
        st.divider()
        st.markdown("### Model Visual Evidence")
        st.caption("Coverage and grouped model charts from the latest notebook runs.")

        loaded_count = len(results_graphs)
        coverage_line = f"Loaded {loaded_count}/{total_result_graphs} graphs from results/."
        if loaded_count == total_result_graphs:
            st.caption(coverage_line)
        else:
            st.warning(coverage_line + " Some result graphs are not currently loaded into the UI.")

        grouped = group_result_graphs(results_graphs)
        section_names = list(grouped.keys())
        section_tabs = st.tabs(section_names)

        for tab, section in zip(section_tabs, section_names):
            with tab:
                images = grouped.get(section, [])
                cols = st.columns(2)
                for i, pth in enumerate(images):
                    with cols[i % 2]:
                        st.image(str(pth), caption=pth.name, use_container_width=True)
    else:
        st.divider()
        st.markdown("### Model Visual Evidence")
        st.caption(f"Loaded 0/{total_result_graphs} graphs from results/.")
        st.warning("No result image files were found in results/. Add image outputs to display visual evidence.")

with tabs[3]:
    st.subheader("Results Gallery")
    st.caption("Current model evidence first; archived figures and reports remain available below.")
    st.divider()

    g1, g2 = st.columns(2)
    g1.metric("Result images", f"{len(results_graphs)}")
    g2.metric("Result tables/docs", f"{len(result_artifacts)}")

    gallery_tabs = st.tabs(["Current Charts", "All Charts", "Reports and Data"])
    current_chart_names = [
        "psg_track_weight_split.png",
        "productivity_feature_shap_importance.png",
        "model_comparison_visualization.png",
        "all_tracks_train_vs_test_accuracy.png",
        "productivity_spiral_all_models.png",
        "sustainability_spiral_net_metrics.png",
        "genomic_spiral_all_models.png",
    ]
    current_charts = [RESULTS_DIR / name for name in current_chart_names if (RESULTS_DIR / name).exists()]

    with gallery_tabs[0]:
        if current_charts:
            for start in range(0, len(current_charts), 2):
                cols = st.columns(2)
                for col, image_path in zip(cols, current_charts[start : start + 2]):
                    with col:
                        st.image(str(image_path), caption=image_path.stem.replace("_", " ").title(), use_container_width=True)
        else:
            st.info("Current charts have not been generated yet. Run the full rebuild.")

    with gallery_tabs[1]:
        groups = group_result_graphs(load_results_graphs())
        if groups:
            group_name = st.selectbox("Chart group", list(groups), key="gallery_chart_group")
            selected_images = groups[group_name]
            for start in range(0, len(selected_images), 2):
                cols = st.columns(2)
                for col, image_path in zip(cols, selected_images[start : start + 2]):
                    with col:
                        st.image(str(image_path), caption=image_path.name, use_container_width=True)
        else:
            st.info("No result images are available.")

    with gallery_tabs[2]:
        if result_artifacts:
            preferred = [
                "final_project_results_summary.md",
                "paper_results_tables.md",
                "psg_combined_metrics.csv",
                "psg_track_contributions.csv",
                "productivity_feature_shap_importance.csv",
                "validation_audit.md",
                "water_join_metadata.json",
                "reconstruction_readiness.json",
            ]
            ordered_artifacts = sorted(
                result_artifacts,
                key=lambda path: (preferred.index(path.name) if path.name in preferred else len(preferred), path.name.lower()),
            )
            selected_artifact = st.selectbox(
                "Report or data file",
                ordered_artifacts,
                format_func=lambda path: path.name,
                key="gallery_artifact",
            )
            suffix = selected_artifact.suffix.lower()
            if suffix == ".csv":
                artifact_df = safe_read_csv(selected_artifact)
                if artifact_df is not None:
                    st.caption(f"{len(artifact_df):,} rows · {len(artifact_df.columns)} columns")
                    st.dataframe(artifact_df, width="stretch", hide_index=True)
            elif suffix == ".md":
                text = load_markdown_file(selected_artifact, *file_signature(selected_artifact))
                if text:
                    st.markdown(text)
            else:
                text = load_text_file(selected_artifact, *file_signature(selected_artifact))
                if text:
                    st.code(text, language="json" if suffix == ".json" else "text")
        else:
            st.info("No result reports or data files are available.")

with tabs[4]:
    st.subheader("Architecture")
    st.caption("Model types and input features are read from the current saved artifacts.")
    productivity_features_now = expected_features(productivity_model) or []
    sustainability_features_now = sustainability_bundle.get("feature_columns", []) if isinstance(sustainability_bundle, dict) else expected_features(sustainability_bundle) or []
    genomic_features_now = expected_features(feature_selector_model) or []
    architecture_rows = [
        {"Track": "Productivity", "Model": prod_name, "Inputs": ", ".join(productivity_features_now), "Target": "Production tertiles from log1p(production)"},
        {"Track": "Sustainability", "Model": sust_name, "Inputs": ", ".join(sustainability_features_now), "Target": "Production tertiles"},
        {"Track": "Auxiliary context classifier", "Model": gen_name, "Inputs": ", ".join(genomic_features_now), "Target": "Production tertiles; not genomic sample disease risk"},
    ]
    st.dataframe(pd.DataFrame(architecture_rows), width="stretch", hide_index=True)
    st.info(
        "The production table contains country, year, and six time-bucket water measurements. "
        "Climate is excluded because the source years do not overlap production; raw genomic samples "
        "are clustered separately and cannot be joined to farm records with the available keys."
    )

with tabs[5]:
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

with tabs[6]:
    st.subheader("Scenario Simulation")
    st.caption("Perturb the six time-bucket water features used by the current models.")
    st.divider()
    baseline = dict(st.session_state.get("latest_profile", dict(base_profile)))
    scenario = dict(baseline)

    scenario_changes = {}
    controls = st.columns(3)
    for index, feature in enumerate(WATER_FEATURES):
        center = float(baseline.get(feature, data_df[feature].median()))
        spread = float(data_df[feature].std(ddof=0))
        step = max(spread / 10.0, 0.01)
        change = controls[index % len(controls)].slider(
            f"{WATER_LABELS[feature]} adjustment",
            min_value=-2.0 * spread if spread > 0 else -0.5,
            max_value=2.0 * spread if spread > 0 else 0.5,
            value=0.0,
            step=step,
            key=f"scenario_{feature}",
        )
        scenario[feature] = center + float(change)
        scenario_changes[feature] = float(change)

    spread_by_feature = data_df[WATER_FEATURES].std(ddof=0).replace(0, 1.0)
    scenario_shift = float(np.mean([abs(scenario_changes[col]) / spread_by_feature[col] for col in WATER_FEATURES]))

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

    prod_model_delta = model_response_delta(productivity_model, base_prod, scen_prod)
    sus_model_delta = model_response_delta(sustainability_bundle, base_sus, scen_sus)

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
    if b_prod_conf is not None and s_prod_conf is not None:
        prod_delta = float(s_prod_conf - b_prod_conf)
        if abs(prod_delta) < 1e-12 and prod_model_delta is not None:
            prod_delta = float(prod_model_delta)
    else:
        # Fallback to normalized score shift so slider changes always reflect here.
        prod_delta = float(prod_model_delta) if prod_model_delta is not None else float((s_score - b_score) / 100.0)

    if b_sus_conf is not None and s_sus_conf is not None:
        sus_delta = float(s_sus_conf - b_sus_conf)
        if abs(sus_delta) < 1e-12 and sus_model_delta is not None:
            sus_delta = float(sus_model_delta)
    else:
        sus_delta = float(sus_model_delta) if sus_model_delta is not None else float((s_sus_score - b_sus_score) / 100.0)

    d1.metric("Productivity confidence delta", f"{prod_delta:+.6f}")
    d2.metric("Sustainability confidence delta", f"{sus_delta:+.6f}")
    d3.metric("Scenario shift index", f"{scenario_shift:.1f}")

with tabs[7]:
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
        elif level == "info":
            st.info(text)
        else:
            st.success(text)

    s1, s2, s3 = st.columns(3)
    s1.metric("Productivity score", f"{float(latest.get('productivity_score') or 0.0):.1f}/100")
    s2.metric("Sustainability", str(latest.get("sustainability_label") or "N/A"))
    s3.metric("Auxiliary class signal", str(latest.get("genomic_signal") or "N/A"))
    st.caption(f"Sustainability score: {float(latest.get('sustainability_score') or 0.0):.1f}/100")

st.markdown("---")
st.caption("Built using Explainable AI for Aquaculture Intelligence")
