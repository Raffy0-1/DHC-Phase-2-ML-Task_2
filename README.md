# DHC-Phase-2-ML-Task_2
Production-ready ML pipeline for customer churn prediction using scikit-learn Pipeline API, GridSearchCV, and joblib
# Telco Customer Churn — End-to-End ML Pipeline

> A production-ready machine learning pipeline for predicting customer churn,
> built with scikit-learn's Pipeline API, GridSearchCV hyperparameter tuning,
> and joblib model export.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## 📌 Project Overview

Customer churn is one of the most costly problems in subscription businesses.
This project builds a **full ML pipeline** — from raw messy data to a saved,
loadable model — using professional engineering practices throughout.

The pipeline takes raw customer data and outputs a **churn probability score**,
enabling retention teams to proactively target high-risk customers.

```
Raw Customer Data  →  ColumnTransformer  →  RandomForestClassifier  →  Churn Probability
                      (Scale + Encode)        (Tuned via GridSearch)
```

---

## 📊 Dataset

| Property         | Value                                      |
|------------------|--------------------------------------------|
| Source           | [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) |
| Customers        | 7,043                                      |
| Features         | 20 (after dropping `customerID`)           |
| Target           | `Churn` — binary (1 = churned, 0 = stayed) |
| Class Split      | 73.5% No Churn / 26.5% Churn              |
| Challenge        | Imbalanced classes, mixed feature types    |

---

## 🔍 Key EDA Findings

Three features dominate churn signal:

| Feature          | Insight                                                       |
|------------------|---------------------------------------------------------------|
| **Contract type**    | Month-to-month → 42.7% churn vs 2.8% for two-year contracts  |
| **Tenure**           | Churners leave early — peak risk in first few months          |
| **Monthly charges**  | Positive correlation (0.19) with churn — higher bills = more risk |

> **Hidden data quality issue found:** `TotalCharges` was stored as `object` dtype
> due to 11 blank strings (new customers with zero tenure). Fixed using
> `pd.to_numeric(..., errors='coerce')` then filled with `0` based on domain knowledge.

---

## ⚙️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     sklearn Pipeline                     │
│                                                         │
│  Step 1: ColumnTransformer                              │
│  ├── StandardScaler    → tenure, MonthlyCharges,        │
│  │                        TotalCharges                  │
│  └── OneHotEncoder     → 16 categorical columns         │
│            ↓                                            │
│  Step 2: RandomForestClassifier                         │
│           max_depth=10, min_samples_split=10,           │
│           n_estimators=300, class_weight='balanced'     │
└─────────────────────────────────────────────────────────┘
```

**Why Pipeline over manual preprocessing?**

- Preprocessing is fitted **only on training data** — no data leakage
- One object to `joblib.dump()` — one object to `joblib.load()`
- Raw data in, predictions out — no manual steps at inference time

---

## 📈 Model Results

| Model               | Baseline AUC | Tuned AUC  | Churn Recall (Baseline) | Churn Recall (Tuned) |
|---------------------|:------------:|:----------:|:-----------------------:|:--------------------:|
| Logistic Regression | 0.8415       | 0.8408     | 0.78                    | ~0.78                |
| **Random Forest**   | 0.8222       | **0.8415** | 0.48                    | **0.74**             |

**Winner: Tuned Random Forest**

### Why did default Random Forest underperform?

With `max_depth=None` (default), trees grew deep enough to memorize majority-class
patterns — effectively learning to predict "No Churn" most of the time.
GridSearchCV found that constraining `max_depth=10` and `min_samples_split=10`
forced generalization, lifting churn recall from **0.48 → 0.74**.

### Why AUC over Accuracy?

A model predicting "No Churn" for every customer scores **73.5% accuracy**
while catching **zero** actual churners. AUC-ROC and Recall on the minority
class are the correct metrics when the cost of missing a churner exceeds
the cost of a false alarm.

---

## 🔮 Live Prediction Example

```python
import joblib, pandas as pd

pipeline = joblib.load('pipeline_random_forest_tuned.joblib')

new_customer = pd.DataFrame([{
    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 2, 'PhoneService': 'Yes',
    'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.0, 'TotalCharges': 170.0
}])

prob = pipeline.predict_proba(new_customer)[0][1]
print(f"Churn probability: {prob:.1%}")  # → 84.3%
```

> **84.3% churn probability** — fully explained by EDA: short tenure (2 months),
> month-to-month contract, fiber optic service, and above-average monthly charges.
> Every feature points in the same direction.

---

## 🗂️ Repository Structure

```
telco-churn-pipeline/
│
├── telco_churn_pipeline.ipynb     ← Main notebook (all phases)
├── requirements.txt               ← Python dependencies
├── .gitignore                     ← Excludes models, data, cache
├── README.md                      ← You are here
│
└── assets/                        ← Generated output images
    ├── eda_plots.png
    └── roc_curve.png
```

> **Note:** `.joblib` model files and the raw dataset are excluded from the repo
> via `.gitignore` (file size). The notebook downloads the dataset automatically
> on first run and saves everything to Google Drive.

---

## 🚀 Getting Started

### Run on Google Colab (Recommended)

1. Open the notebook in Colab
2. Run **Cell 1** — mounts your Google Drive and creates the project folder
3. Run all cells top to bottom — dataset downloads automatically on first run
4. All outputs (models, plots) save to `/MyDrive/telco_churn_pipeline/`

> On runtime restart, re-running cells is safe — the dataset and trained models
> are loaded from Drive if they already exist, skipping expensive recomputation.

### Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/telco-churn-pipeline.git
cd telco-churn-pipeline
pip install -r requirements.txt
jupyter notebook telco_churn_pipeline.ipynb
```

---

## 🛠️ Tech Stack

| Tool              | Purpose                                      |
|-------------------|----------------------------------------------|
| `scikit-learn`    | Pipeline, ColumnTransformer, models, GridSearchCV |
| `pandas`          | Data loading, cleaning, EDA                  |
| `numpy`           | Numerical operations                         |
| `matplotlib`      | EDA plots, ROC curve                         |
| `seaborn`         | Correlation heatmap                          |
| `joblib`          | Model serialization / export                 |
| Google Colab      | GPU-enabled runtime environment              |
| Google Drive      | Persistent storage across sessions           |

---

## 💡 Skills Demonstrated

- **ML Pipeline construction** — `sklearn.pipeline.Pipeline` + `ColumnTransformer`
- **Imbalanced classification** — class weighting, stratified splits, correct metrics
- **Hyperparameter tuning** — `GridSearchCV` with `StratifiedKFold` cross-validation
- **Data leakage prevention** — pipeline-enforced fit/transform separation
- **Hidden data quality issues** — detecting and resolving non-NaN missing values
- **Model export & reusability** — `joblib` serialization with verified reload
- **EDA-driven modeling** — visual insights informing preprocessing decisions
- **Production inference** — raw input → prediction with zero manual preprocessing

---

## 🔭 Possible Extensions

| Extension                  | Skill Added                       |
|----------------------------|-----------------------------------|
| SHAP explainability        | Per-prediction feature attribution |
| FastAPI `/predict` endpoint | REST API deployment               |
| MLflow experiment tracking | Experiment management             |
| SMOTE oversampling         | Advanced imbalance handling       |
| Streamlit dashboard        | Interactive demo for stakeholders |

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.

---

<p align="center">
  Built with intention — every decision documented, every result explainable.
</p>
