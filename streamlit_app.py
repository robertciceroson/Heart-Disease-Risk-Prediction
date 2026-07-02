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
 
 
# ── Lifestyle risk modifier ───────────────────────────────────────────────────
def lifestyle_modifier(family_hx, smoking, alcohol, activity, diet):
    """Evidence-based additive adjustment to XGBoost clinical base score."""
    mod = 0.0
    # Family history — early-onset carries the strongest hereditary signal
    if "before age 55" in family_hx or "before age 65" in family_hx:
        mod += 0.15   # early-onset first-degree relative
    elif "age of onset unknown" in family_hx:
        mod += 0.10   # family history present but timing unclear
    # Smoking
    if smoking == "Current smoker":
        mod += 0.15
    elif smoking == "Former smoker (quit > 1 year ago)":
        mod += 0.05
    # Alcohol
    if alcohol == "Heavy (> 14 drinks / week)":
        mod += 0.08
    elif alcohol == "Moderate (7–14 drinks / week)":
        mod += 0.02
    # Physical activity
    if activity == "Sedentary (< 30 min / week)":
        mod += 0.10
    elif activity == "Light (30–90 min / week)":
        mod += 0.05
    elif activity == "Active (150–300 min / week)":
        mod -= 0.03
    elif activity == "Very active (> 300 min / week)":
        mod -= 0.05
    # Diet
    if diet == "Poor — high in processed foods, saturated fats & sodium":
        mod += 0.08
    elif diet == "Healthy — fiber-rich, vegetables & lean proteins":
        mod -= 0.05
    return mod
 
 
# ── Header ────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Prediction")
st.markdown(
    "Enter clinical measurements and lifestyle factors to estimate cardiovascular risk. "
    "Clinical features are scored by a **XGBoost** model trained on the Cleveland Heart Disease dataset (UCI); "
    "lifestyle factors apply evidence-based risk adjustments on top."
)
st.warning(
    "⚠️ **For educational and portfolio demonstration purposes only. "
    "Not a medical device — do not use for clinical decision-making.**",
    icon="🩺",
)
st.markdown("---")
 
with st.spinner("Loading model…"):
    model, scaler, medians, feature_cols, acc, auc = load_and_train()
st.caption(f"Model ready · Test Accuracy: **{acc*100:.1f}%** · ROC-AUC: **{auc:.2f}**")
st.markdown("---")
 
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Clinical measurements
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🩺 Section 1 — Clinical Measurements")
st.caption("These 13 features feed directly into the XGBoost model.")
 
col1, col2 = st.columns(2)
 
with col1:
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0
 
    age = st.slider("Age (years)", 20, 80, 54,
                    help="Risk of heart disease increases naturally with age. "
                         "Men tend to develop it earlier in life; women's risk rises "
                         "significantly after menopause (typically around age 55–65).")
    # Contextual age-sex note
    if sex_val == 1 and age >= 45:
        st.caption("ℹ️ Men aged 45+ enter an elevated cardiovascular risk window.")
    elif sex_val == 0 and age >= 55:
        st.caption("ℹ️ Women aged 55+ face significantly increased risk following menopause.")
 
    cp = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
        help="Asymptomatic chest pain is counterintuitively the highest-risk type in this dataset.",
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
    thalach = st.slider("Max Heart Rate Achieved (bpm)", 70, 205, 149,
                        help="Lower max heart rate is associated with higher disease risk.")
 
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    exang_val = 1 if exang == "Yes" else 0
 
    oldpeak = st.slider("ST Depression During Exercise", 0.0, 6.5, 1.0, step=0.1,
                        help="Higher ST depression indicates greater cardiac stress during exercise.")
 
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"],
    )
    slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope) + 1
 
    ca = st.slider("Major Vessels Visible on Fluoroscopy (0–3)", 0, 3, 0,
                   help="Number of major blood vessels coloured by fluoroscopy dye. More = higher risk.")
 
    thal = st.selectbox(
        "Thallium Stress Test Result",
        ["Normal", "Fixed Defect (permanent blockage)", "Reversible Defect (blood flow issue under stress)"],
        help="A nuclear imaging test that shows blood flow to the heart at rest and during stress. "
             "'Fixed defect' = permanently blocked area; 'Reversible' = reduces only under stress.",
    )
    thal_map = {
        "Normal": 3,
        "Fixed Defect (permanent blockage)": 6,
        "Reversible Defect (blood flow issue under stress)": 7,
    }
    thal_val = thal_map[thal]
 
st.markdown("---")
 
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Lifestyle & Family History
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🏃 Section 2 — Lifestyle & Family History")
st.caption(
    "These factors are not in the Cleveland dataset but are major cardiovascular risk drivers. "
    "They are applied as evidence-based adjustments to the clinical base score."
)
 
col3, col4 = st.columns(2)
 
with col3:
    family_hx = st.selectbox(
        "Family History of Heart Disease",
        [
            "No known family history",
            "Yes — male relative had heart attack / heart disease before age 55",
            "Yes — female relative had heart attack / heart disease before age 65",
            "Yes — family history present, age of onset unknown",
        ],
        help="Early-onset family history is the strongest hereditary risk signal. "
             "A first-degree relative (parent or sibling) with heart disease before age 55 (men) "
             "or 65 (women) indicates a significant genetic predisposition.",
    )
 
    smoking = st.selectbox(
        "Smoking / Tobacco Use",
        [
            "Never smoked",
            "Former smoker (quit > 1 year ago)",
            "Current smoker",
        ],
        help="Smoking is one of the leading causes of cardiovascular disease. "
             "Chemicals in tobacco damage blood vessels and accelerate artery narrowing.",
    )
 
    alcohol = st.selectbox(
        "Alcohol Consumption",
        [
            "None or rare (< 1 drink / week)",
            "Moderate (7–14 drinks / week)",
            "Heavy (> 14 drinks / week)",
        ],
        help="Heavy drinking raises blood pressure, contributes to heart failure, and increases stroke risk.",
    )
 
with col4:
    activity = st.selectbox(
        "Weekly Physical Activity",
        [
            "Sedentary (< 30 min / week)",
            "Light (30–90 min / week)",
            "Moderate (90–150 min / week) — meets minimum guidelines",
            "Active (150–300 min / week)",
            "Very active (> 300 min / week)",
        ],
        index=2,
        help="Leading health organisations recommend at least 150 minutes of moderate exercise "
             "(e.g. brisk walking) per week for cardiovascular health.",
    )
 
    diet = st.selectbox(
        "Typical Diet Quality",
        [
            "Poor — high in processed foods, saturated fats & sodium",
            "Average — mixed, some healthy choices",
            "Healthy — fiber-rich, vegetables & lean proteins",
        ],
        index=1,
        help="Diets high in saturated fats, trans fats, sodium, and processed foods elevate risk. "
             "Diets rich in fiber, vegetables, and lean proteins are cardioprotective.",
    )
 
st.markdown("---")
 
# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict My Heart Disease Risk", use_container_width=True, type="primary"):
 
    # Clinical XGBoost score
    input_data = pd.DataFrame([[
        age, sex_val, cp_val, trestbps, chol, fbs_val,
        restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val,
    ]], columns=feature_cols)
    input_scaled = scaler.transform(input_data)
    base_risk    = float(model.predict_proba(input_scaled)[0][1])
 
    # Lifestyle adjustment
    ls_mod       = lifestyle_modifier(family_hx, smoking, alcohol, activity, diet)
    combined_risk = float(np.clip(base_risk + ls_mod, 0.01, 0.99))
    prediction    = 1 if combined_risk >= 0.5 else 0
 
    st.subheader("📊 Risk Assessment")
 
    if prediction == 1:
        st.error("### 🔴 Elevated Risk Detected")
    else:
        st.success("### 🟢 Lower Risk Detected")
 
    # Three metrics
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Clinical Base Risk", f"{base_risk*100:.1f}%")
    ls_sign = f"+{ls_mod*100:.0f}%" if ls_mod >= 0 else f"{ls_mod*100:.0f}%"
    col_b.metric("Lifestyle Adjustment", ls_sign)
    col_c.metric("Combined Risk Score", f"{combined_risk*100:.1f}%")
 
    # Gauge bar
    fig_g, ax_g = plt.subplots(figsize=(6, 1.1))
    bar_color = "#E53935" if combined_risk >= 0.5 else "#43A047"
    ax_g.barh(["Risk"], [combined_risk],         color=bar_color,  height=0.5)
    ax_g.barh(["Risk"], [1 - combined_risk],     color="#E0E0E0",  height=0.5, left=combined_risk)
    ax_g.axvline(0.5, color="white", linewidth=2, linestyle="--")
    ax_g.set_xlim(0, 1)
    ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_g.set_xticklabels(["0%", "25%", "50% threshold", "75%", "100%"])
    ax_g.set_yticks([])
    ax_g.set_title(f"Combined Risk Score: {combined_risk*100:.1f}%", fontsize=11)
    ax_g.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig_g)
 
    # Risk signal callouts
    st.markdown("---")
    st.markdown("#### 🔑 Key Risk Signals")
 
    clinical_signals, lifestyle_signals = [], []
 
    # Clinical
    if cp_val == 3:
        clinical_signals.append("⚠️ **Asymptomatic chest pain** — highest-risk chest pain type in this model")
    if thalach < 120:
        clinical_signals.append(f"⚠️ **Low max heart rate ({thalach} bpm)** — inversely correlated with disease")
    if oldpeak > 2.0:
        clinical_signals.append(f"⚠️ **High ST depression ({oldpeak})** — strong cardiac stress indicator")
    if ca > 0:
        clinical_signals.append(f"⚠️ **{ca} major vessel(s) visible on fluoroscopy** — significant disease indicator")
    if thal_val == 7:
        clinical_signals.append("⚠️ **Reversible defect on thallium stress test** — elevated risk marker")
    if exang_val == 1:
        clinical_signals.append("⚠️ **Exercise-induced angina** — associated with higher disease risk")
 
    # Lifestyle
    if "before age 55" in family_hx or "before age 65" in family_hx:
        lifestyle_signals.append("⚠️ **Early-onset family history of heart disease** (+15% adjustment) — strongest hereditary risk signal")
    elif "age of onset unknown" in family_hx:
        lifestyle_signals.append("⚠️ **Family history of heart disease** (+10% adjustment) — genetic predisposition present")
    if smoking == "Current smoker":
        lifestyle_signals.append("⚠️ **Current smoker** (+15% adjustment) — leading modifiable risk factor")
    elif "Former" in smoking:
        lifestyle_signals.append("⚠️ **Former smoker** (+5% adjustment) — residual risk remains")
    if "Heavy" in alcohol:
        lifestyle_signals.append("⚠️ **Heavy alcohol use** (+8% adjustment) — raises blood pressure and strain")
    if "Sedentary" in activity:
        lifestyle_signals.append("⚠️ **Sedentary lifestyle** (+10% adjustment) — below recommended 150 min/week")
    if "Light" in activity and "30–90" in activity:
        lifestyle_signals.append("⚠️ **Low activity** (+5% adjustment) — below recommended 150 min/week")
    if "Poor" in diet:
        lifestyle_signals.append("⚠️ **Poor diet** (+8% adjustment) — high processed foods & saturated fats")
    if "Active (150" in activity:
        lifestyle_signals.append("✅ **Active lifestyle** (−3% adjustment) — meets cardiovascular guidelines")
    if "Very active" in activity:
        lifestyle_signals.append("✅ **Very active lifestyle** (−5% adjustment) — exceeds guidelines")
    if "Healthy" in diet:
        lifestyle_signals.append("✅ **Healthy diet** (−5% adjustment) — cardioprotective eating pattern")
 
    if clinical_signals:
        st.markdown("**Clinical factors:**")
        for s in clinical_signals:
            st.markdown(f"- {s}")
    if lifestyle_signals:
        st.markdown("**Lifestyle factors:**")
        for s in lifestyle_signals:
            st.markdown(f"- {s}")
    if not clinical_signals and not lifestyle_signals:
        st.markdown("✅ No major individual risk flags detected in this profile.")
 
    # Feature importance
    with st.expander("📊 What clinical features drive the XGBoost model?"):
        importance = model.feature_importances_
        feature_labels = [
            "Age", "Sex", "Chest Pain", "Rest BP", "Cholesterol", "Fasting BS",
            "Rest ECG", "Max HR", "Exer Angina", "ST Depression", "ST Slope",
            "Vessels", "Thallium Test",
        ]
        sorted_idx = np.argsort(importance)
        fig_i, ax_i = plt.subplots(figsize=(6, 4))
        ax_i.barh(
            [feature_labels[i] for i in sorted_idx],
            importance[sorted_idx],
            color="#E53935",
        )
        ax_i.set_xlabel("Feature Importance")
        ax_i.set_title("XGBoost Clinical Feature Importances")
        st.pyplot(fig_i)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Clinical model: XGBoost trained on Cleveland Heart Disease dataset (UCI ML Repository) · "
    f"Test Accuracy: {acc*100:.1f}% · ROC-AUC: {auc:.2f} · "
    "Lifestyle adjustments based on published cardiovascular risk literature · "
    "For educational use only · "
    "[GitHub repo](https://github.com/robertciceroson/Heart-Disease-Risk-Prediction)"
)
