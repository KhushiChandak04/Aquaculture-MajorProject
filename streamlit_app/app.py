"""
Aquaculture XAI — Streamlit Dashboard
======================================
Owner: Khushi (final integration)

This app loads all three trained models and provides:
  1. Productivity prediction (regression)
  2. Sustainability classification
  3. Genomic feature importance + SHAP explanations

HOW TO RUN:
  cd streamlit_app
  streamlit run app.py

PREREQUISITES:
  All three .pkl files must exist in ../models/
    - productivity_model.pkl   (Khushi)
    - sustainability_model.pkl (Shravya)
    - feature_selector.pkl     (Janhavi)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Aquaculture XAI Dashboard",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 Aquaculture XAI Dashboard")
st.markdown("Explainable AI for Aquaculture Productivity, Sustainability & Genomics")

# ============================================================
# LOAD MODELS
# ============================================================
MODELS_DIR = os.path.join("..", "models")

@st.cache_resource
def load_models():
    """Load all three models. Returns None for any missing model."""
    models = {}

    prod_path = os.path.join(MODELS_DIR, "productivity_model.pkl")
    sust_path = os.path.join(MODELS_DIR, "sustainability_model.pkl")
    feat_path = os.path.join(MODELS_DIR, "feature_selector.pkl")

    models["productivity"] = joblib.load(prod_path) if os.path.exists(prod_path) else None
    models["sustainability"] = joblib.load(sust_path) if os.path.exists(sust_path) else None
    models["feature_selector"] = joblib.load(feat_path) if os.path.exists(feat_path) else None

    return models


# ============================================================
# SIDEBAR — MODEL STATUS
# ============================================================
st.sidebar.header("Model Status")

try:
    models = load_models()
    for name, model in models.items():
        status = "✅ Loaded" if model is not None else "❌ Not found"
        st.sidebar.write(f"**{name}:** {status}")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    models = {"productivity": None, "sustainability": None, "feature_selector": None}


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Productivity Prediction",
    "🌿 Sustainability Classification",
    "🧬 Genomic Feature Importance"
])

# --- TAB 1: Productivity ---
with tab1:
    st.header("Productivity Prediction (Regression)")
    if models.get("productivity") is None:
        st.warning("⚠️ productivity_model.pkl not found in models/. Khushi needs to upload it.")
    else:
        st.success("Model loaded! Add input widgets below for prediction.")
        # TODO: Add st.number_input() widgets for user input features
        # TODO: model.predict() and display result
        # TODO: SHAP waterfall plot for individual prediction

# --- TAB 2: Sustainability ---
with tab2:
    st.header("Sustainability Classification")
    if models.get("sustainability") is None:
        st.warning("⚠️ sustainability_model.pkl not found in models/. Shravya needs to upload it.")
    else:
        st.success("Model loaded! Add input widgets below for prediction.")
        # TODO: Add st.number_input() widgets for user input features
        # TODO: model.predict() and display class
        # TODO: SHAP summary plot

# --- TAB 3: Genomics ---
with tab3:
    st.header("Genomic Feature Importance & XAI")
    if models.get("feature_selector") is None:
        st.warning("⚠️ feature_selector.pkl not found in models/. Janhavi needs to upload it.")
    else:
        st.success("Feature selector loaded!")
        # TODO: Display selected features
        # TODO: Load genomic_feature_importance.csv from results/
        # TODO: Plot feature importance bar chart


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Aquaculture XAI Project — Khushi, Shravya, Janhavi")
