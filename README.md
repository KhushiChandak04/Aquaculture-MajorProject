# 🐟 Aquaculture XAI Project

> **Explainable AI for Aquaculture: Productivity Prediction, Sustainability Classification & Genomic Feature Modeling**

---

## 📌 Team

| Member      | Role                                                                              | Primary Notebook                  |
| ----------- | --------------------------------------------------------------------------------- | --------------------------------- |
| **Khushi**  | Productivity modeling (Regression) + Shared preprocessing + Streamlit integration | `02_productivity_khushi.ipynb`    |
| **Shravya** | Sustainability modeling (Classification)                                          | `03_sustainability_shravya.ipynb` |
| **Janhavi** | Genomic feature modeling + XAI                                                    | `04_genomic_xai_janhavi.ipynb`    |

---

## 📁 Project Structure

```
aquaculture-xai/
│
├── data/
│   ├── raw/
│   │   ├── production.csv
│   │   ├── water_quality.csv
│   │   ├── climate.csv
│   │   └── genomic.csv
│   │
│   └── processed/
│       └── final_dataset.csv
│
├── notebooks/
│   ├── 01_shared_preprocessing.ipynb
│   ├── 02_productivity_khushi.ipynb
│   ├── 03_sustainability_shravya.ipynb
│   └── 04_genomic_xai_janhavi.ipynb
│
├── models/
│   ├── productivity_model.pkl
│   ├── sustainability_model.pkl
│   └── feature_selector.pkl
│
├── results/
│   ├── productivity_metrics.csv
│   ├── sustainability_metrics.csv
│   └── genomic_feature_importance.csv
│
├── streamlit_app/
│   ├── app.py
│   └── README_streamlit.txt
│
├── requirements.txt
└── README.md
```

---

## 🧠 Responsibility Map

### Shared Preprocessing (Khushi only)

* Input: all files in `data/raw/`
* Output: `data/processed/final_dataset.csv`
* Rule: No manual edits by others

---

### Khushi — Productivity (Regression)

* Output: `models/productivity_model.pkl`
* Metrics: MAE, RMSE, R²

---

### Shravya — Sustainability (Classification)

* Output: `models/sustainability_model.pkl`
* Metrics: Accuracy, Precision, Recall, F1

---

### Janhavi — Genomic + XAI

* Output: `models/feature_selector.pkl`
* Metrics: Feature importance + SHAP analysis

---

## 🔗 Model Integration Flow

```
Genomic Model → Trait Prediction
Environment Model → Production Influence
Final Model → Combined Prediction
```

Streamlit pipeline:

1. User input
2. Genomic feature selection
3. Productivity prediction
4. Sustainability classification
5. SHAP explanation

---

## 🚀 Streamlit App

```bash
pip install -r requirements.txt
cd streamlit_app
streamlit run app.py
```

---

## 📋 Tech Stack

* pandas, numpy
* scikit-learn, xgboost
* shap (Explainable AI)
* matplotlib, seaborn
* streamlit
* tensorflow (optional)
* joblib

---

## 🧩 Advanced Enhancements (For 8.5–9/10 Project)

### 1. Interactive Dashboard (High Impact)

* Display predictions
* SHAP explanations
* Feature importance visualization

### 2. Time-Series Forecasting

* Add LSTM / ARIMA models
* Predict future production trends

### 3. Scenario Simulation

* Example: temperature +2°C impact on production
* “What-if” analysis for sustainability

### 4. Policy Recommendation Layer (Optional)

* Suggest optimal environmental conditions
* Provide decision-support insights

---

## 🔄 Workflow (Colab Users)

1. Download `final_dataset.csv`
2. Train model in Colab
3. Save:

```python
joblib.dump(model, "model.pkl")
metrics.to_csv("metrics.csv")
```

4. Upload to:

* `models/`
* `results/`

---

## ✅ Rules

* Independent work per member
* Same dataset for all
* Fixed filenames (no renaming)
* No duplication
* Clean merge only

---

## 🛠 Setup

```bash
git clone https://github.com/YOUR_USERNAME/Aquaculture-MajorProject.git
cd Aquaculture-MajorProject

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## 📌 Final Note

This project integrates:

* Genetic modeling (Dataset 4)
* Environmental analysis (Datasets 2 & 3)
* Production prediction (Dataset 1)

with Explainable AI to provide interpretable insights into aquaculture productivity.

---

*Built by Khushi, Shravya & Janhavi*
