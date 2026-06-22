# ❤️ Heart Disease Risk Prediction

A binary classification pipeline that predicts the presence of heart disease in patients using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. The project covers the full ML workflow from EDA through model comparison, cross-validation, and real-world patient risk prediction.

---

## 📋 Project Overview

| Field | Details |
|---|---|
| **Author** | Robert Cicero Son |
| **Dataset** | Cleveland Heart Disease — UCI ML Repository |
| **Patients** | 303 patients, 13 clinical features |
| **Task** | Binary classification: Heart Disease (1) vs No Disease (0) |
| **Best Model** | XGBoost / LightGBM |
| **Best ROC-AUC** | ~0.92+ |
| **Best Accuracy** | ~85%+ |

---

## 🎯 Objective

Build and compare multiple classification models to predict whether a patient has heart disease based on clinical measurements such as age, chest pain type, cholesterol, maximum heart rate, and resting blood pressure. Accurate early detection can assist clinicians in prioritising high-risk patients for further testing.

---

## 📁 Repository Structure

```
Heart-Disease-Risk-Prediction/
│
├── heart_disease_prediction_fixed.ipynb   # Full ML pipeline notebook
└── README.md                            # This file
```

---

## 🩺 Dataset — Cleveland Heart Disease (UCI)

The dataset is loaded directly from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) and contains 13 clinical features:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| `target` | Heart disease present (1) or absent (0) — **label** |

---

## 🔧 Pipeline Steps

### Step 1 — Load Dataset
- Loads Cleveland data directly from the UCI repository via URL
- Manually assigns column headers per UCI documentation
- Handles `?` characters as missing values on load

### Step 2 — Exploratory Data Analysis (EDA)
- **Class distribution** check — verifies balance between positive/negative cases
- **Feature distributions** — histograms split by target class for all 5 numeric features
- **Categorical feature breakdown** — bar charts for all 8 categorical features vs target
- **Correlation heatmap** — identifies multicollinearity and feature-target relationships
- **Box plots** — age, max heart rate (`thalach`), and ST depression (`oldpeak`) vs target

### Step 3 — Data Preprocessing
- Converts `?` values to `NaN` and imputes with **column medians** (robust to outliers)
- Splits features (`X`) and target (`y`)
- **80/20 stratified train/test split** — preserves class ratio in both sets
- **StandardScaler** fitted on training data only and applied to both sets (prevents data leakage)

### Step 4 — Model Training & Evaluation
Five models trained and evaluated:

| Model | Notes |
|---|---|
| **Logistic Regression** | Linear baseline; scaled features required |
| **Decision Tree** | Interpretable; `max_depth=5` to prevent overfitting |
| **Random Forest** | 200-tree ensemble; majority vote; no scaling needed |
| **XGBoost** | Sequential gradient boosting; 300 rounds, `learning_rate=0.05` |
| **LightGBM** | Microsoft's leaf-wise boosting; fast and memory-efficient |

Each model is evaluated on 5 metrics:

| Metric | Clinical Meaning |
|---|---|
| **Accuracy** | Overall % correct predictions |
| **Precision** | Of patients flagged as high risk, how many truly have disease |
| **Recall** | Of patients with disease, how many were correctly identified |
| **F1 Score** | Harmonic mean of Precision and Recall — balanced metric |
| **ROC-AUC** | Ranking quality across all classification thresholds |

### Step 5 — Model Comparison
- Side-by-side metrics summary table (best value per metric highlighted)
- Grouped bar chart across all 5 metrics for all models
- Confusion matrices for every model
- ROC curves with AUC scores overlaid on one plot
- **5-Fold Stratified Cross-Validation** — more reliable performance estimates on this small dataset (303 rows)

### Step 6 — Feature Importance Comparison
- Side-by-side importance plots for Random Forest, XGBoost, and LightGBM
- Key predictors across all three models: `cp` (chest pain type), `thalach` (max heart rate), `ca` (vessel count), `oldpeak` (ST depression)

### Step 7 — Best Model Selection
- Composite score = (F1 + ROC-AUC) / 2 used to select the final model
- Radar/spider chart provides a visual multi-metric comparison of all models
- Clinical guidance on when to prioritise Recall vs F1 vs AUC

### Step 8 — Predicting New Patients
- The best model is used to predict risk for simulated new patients
- Each patient is described by the same 13 clinical features
- Output includes predicted class (0/1) and risk probability

### Step 9 — Key Concepts Summary
- Covers data leakage prevention, stratified splits, cross-validation rationale, and metric selection guidance

---

## 📊 Key Findings

- **Chest pain type (`cp`)** is consistently the most predictive feature across all tree-based models
- **Maximum heart rate (`thalach`)** is strongly inversely correlated with disease — lower max heart rate = higher risk
- **ST depression (`oldpeak`)** and **number of major vessels (`ca`)** are high-importance features across XGBoost and LightGBM
- **Ensemble methods** (Random Forest, XGBoost, LightGBM) substantially outperform Logistic Regression and Decision Tree on this dataset
- **5-Fold CV scores** are slightly lower than single-split scores, indicating mild overfitting on the 80/20 split — CV scores are more trustworthy given the small dataset size (303 rows)

---

## 🛠️ Technical Highlights

- **No data leakage** — `StandardScaler` is fit on training data only; test set uses the same mean/std without refitting
- **Stratified splits** — `stratify=y` ensures class balance is preserved in both train and test sets
- **Clinically-framed metrics** — notebook explains the cost of false negatives (missed diagnoses) vs false positives in a medical context
- **Cross-validation** — `StratifiedKFold(n_splits=5)` provides reliable estimates on a small 303-row dataset
- **Composite model selection** — uses F1 + ROC-AUC average rather than accuracy alone, which can be misleading on imbalanced data

---

## 🚀 How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/robertciceroson/Heart-Disease-Risk-Prediction.git
   cd Heart-Disease-Risk-Prediction
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook heart_disease_prediction_fixed.ipynb
   ```

3. Run all cells top to bottom (`Kernel → Restart & Run All`)

> The notebook loads the Cleveland dataset automatically from the UCI repository — no manual download needed.

---

## 📄 Dataset License

Cleveland Heart Disease dataset sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease).  
Original data collected by Robert Detrano, M.D., Ph.D., V.A. Medical Center, Long Beach and Cleveland Clinic Foundation.
---
## Author

**Robert Cicero Son**
Scrum Master · Process Engineer · Prompt Engineer · Data Analyst · AI/ML Practitioner · CSM · CSPO · AI-Empowered SAFe Agilist · Active DoD Secret Clearance

