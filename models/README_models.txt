═══════════════════════════════════════════════════════════════
  MODELS  —  Place trained .pkl model files here
═══════════════════════════════════════════════════════════════

Expected files:

  productivity_model.pkl     ← Khushi   (from 02_productivity_khushi.ipynb)
  sustainability_model.pkl   ← Shravya  (from 03_sustainability_shravya.ipynb)
  feature_selector.pkl       ← Janhavi  (from 04_genomic_xai_janhavi.ipynb)

RULES:
  • Do NOT rename these files — the Streamlit app loads them by exact name
  • Save models using: joblib.dump(model, "filename.pkl")
  • Colab users: download .pkl and upload here manually

═══════════════════════════════════════════════════════════════
