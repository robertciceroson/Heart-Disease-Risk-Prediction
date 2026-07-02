import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered",
)

# ── Load data & train model ───────────────────────────────────────────────────
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

@st.cache_resource(show_spinner=False)
def load_and_train():
    df = pd.read_csv(UCI_URL, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    medians = X.median()
    X = X.fillna(medians)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train_s, y_train)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return model, scaler, medians, list(X.columns), acc, auc


# ── Header ────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Prediction")
st.markdown(
    "Enter a patient's clinical measurements to estimate their risk of heart disease. "
    "Trained on the **Cleveland Heart Disease dataset** (UCI) using **XGBoost**."
)
st.warning(
    "⚠️ **This tool is for educational and portfolio demonstration purposes only. "
    "It is not a medical device and should not be used for clinical decision-making.**",
    icon="🩺",
)
st.markdown("---")

with st.spinner("Loading model…"):
    model, scaler, medians, feature_cols, acc, auc = load_and_train()

st.caption(f"Model ready · Test Accuracy: **{acc*100:.1f}%** · ROC-AUC: **{auc:.2f}**")
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("🩺 Patient Clinical Features")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 20, 80, 54)

    sex = st.selectbox("Sex", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    )
    cp_val = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 570, 246)

    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs_val = 1 if fbs == "Yes" else 0

    restecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
    )
    restecg_val = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)

with col2:
    thalach = st.slider("Max Heart Rate Achieved (bpm)", 70, 205, 149)

    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    exang_val = 1 if exang == "Yes" else 0

    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"],
    )
    slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope) + 1

    ca = st.slider("Major Vessels Coloured by Fluoroscopy (0–3)", 0, 3, 0)

    thal = st.selectbox(
        "Thalassemia",
        ["Normal", "Fixed Defect", "Reversible Defect"],
    )
    thal_val = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}[thal]

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([[
        age, sex_val, cp_val, trestbps, chol, fbs_val,
        restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val,
    ]], columns=feature_cols)

    input_scaled = scaler.transform(input_data)
    risk_proba   = float(model.predict_proba(input_scaled)[0][1])
    prediction   = int(model.predict(input_scaled)[0])

    st.subheader("📊 Risk Assessment")

    if prediction == 1:
        st.error(f"### 🔴 Elevated Risk Detected")
    else:
        st.success(f"### 🟢 Low Risk Detected")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Risk Probability", f"{risk_proba*100:.1f}%")
    col_b.metric("Prediction", "Heart Disease" if prediction == 1 else "No Disease")
    col_c.metric("Model ROC-AUC", f"{auc:.2f}")

    # Risk gauge (simple horizontal bar)
    fig_g, ax_g = plt.subplots(figsize=(6, 0.8))
    ax_g.barh(["Risk"], [risk_proba],       color="#E53935", height=0.5)
    ax_g.barh(["Risk"], [1 - risk_proba],   color="#E0E0E0", height=0.5, left=risk_proba)
    ax_g.axvline(0.5, color="white", linewidth=2, linestyle="--")
    ax_g.set_xlim(0, 1)
    ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_g.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_g.set_yticks([])
    ax_g.set_title(f"Risk Score: {risk_proba*100:.1f}%", fontsize=11)
    ax_g.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig_g)

    # Key risk factor callouts
    st.markdown("---")
    st.markdown("#### 🔑 Key Risk Signals in This Patient")
    signals = []
    if cp_val == 3:
        signals.append("⚠️ **Asymptomatic chest pain** — highest-risk chest pain type")
    if thalach < 120:
        signals.append(f"⚠️ **Low max heart rate ({thalach} bpm)** — inversely correlated with disease")
    if oldpeak > 2.0:
        signals.append(f"⚠️ **High ST depression ({oldpeak})** — strong positive predictor")
    if ca > 0:
        signals.append(f"⚠️ **{ca} major vessel(s) coloured** — significant disease indicator")
    if thal_val == 7:
        signals.append("⚠️ **Reversible thalassemia defect** — elevated risk marker")
    if exang_val == 1:
        signals.append("⚠️ **Exercise-induced angina** — associated with higher risk")

    if signals:
        for s in signals:
            st.markdown(s)
    else:
        st.markdown("✅ No major individual risk flags detected in this profile.")

    # Feature importance
    with st.expander("📊 What features matter most to this model?"):
        importance   = model.feature_importances_
        feature_labels = [
            "Age", "Sex", "Chest Pain", "Rest BP", "Cholesterol", "Fasting BS",
            "Rest ECG", "Max HR", "Exer Angina", "ST Depression", "ST Slope",
            "Vessels", "Thalassemia",
        ]
        sorted_idx = np.argsort(importance)
        fig_i, ax_i = plt.subplots(figsize=(6, 4))
        ax_i.barh(
            [feature_labels[i] for i in sorted_idx],
            importance[sorted_idx],
            color="#E53935",
        )
        ax_i.set_xlabel("Feature Importance")
        ax_i.set_title("XGBoost Feature Importances")
        st.pyplot(fig_i)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "XGBoost trained on Cleveland Heart Disease dataset (UCI ML Repository) · "
    f"Test Accuracy: {acc*100:.1f}% · ROC-AUC: {auc:.2f} · "
    "For educational use only · "
    "[GitHub repo](https://github.com/robertciceroson/Heart-Disease-Risk-Prediction)"
)
