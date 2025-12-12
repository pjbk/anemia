# Developted by* 
# Pankaj Bhowmik  
# Lecturer, Department of Computer Science and Engineering 
# Hajee Mohammad Danesh Science and Technology University  

import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ----------------- Load Model & Scaler -----------------
model = joblib.load('STEM_model.pkl')
scaler = joblib.load('scaler.pkl')

# ----------------- Page Setup -----------------
st.set_page_config(
    page_title="🩸 Anemia Risk Predictor",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("<h1 style='text-align: center;'>🩸 Anemia Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 16px;'>
Estimate <b>risk of anemia</b> based on hematological and demographic parameters.<br>
Enter patient data and click <b>Predict</b> to see the results.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- User Input ---
st.header("🧾 Patient Data")

col1, col2, col3 = st.columns(3)
gender = col1.selectbox("Gender", ["Male", "Female"])
age = col2.slider("Age", 1, 100, 30)
hb = col3.slider("Hemoglobin (Hb)", 3.0, 20.0, 13.0, step=0.1)

col4, col5, col6 = st.columns(3)
rbc = col4.slider("RBC (million cells/μL)", 2.0, 6.5, 4.7, step=0.1)
pcv = col5.slider("Packed Cell Volume (PCV)", 20.0, 60.0, 45.0, step=0.5)
mcv = col6.slider("Mean Corpuscular Volume (MCV)", 50.0, 120.0, 85.0, step=0.5)

col7, col8 = st.columns(2)
mch = col7.slider("Mean Corpuscular Hemoglobin (MCH)", 10.0, 40.0, 27.0, step=0.5)
mchc = col8.slider("Mean Corpuscular Hemoglobin Concentration (MCHC)", 20.0, 40.0, 32.0, step=0.5)

# ----------------- Prepare Input -----------------
gender_numeric = 0 if gender.lower() == 'male' else 1
input_data = np.array([[gender_numeric, age, hb, rbc, pcv, mcv, mch, mchc]])
input_scaled = scaler.transform(input_data)

# ----------------- Prediction -----------------
st.markdown("---")
if st.button("🔮 Predict Anemia Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("### 🎯 Prediction Result")
    risk_tag = "🔴 High Risk of Anemia" if prediction == 1 else "🟢 Low Risk of Anemia"

    st.metric("Risk Probability", f"{probability * 100:.2f}%", label_visibility="visible")
    
    if prediction == 1:
        st.error(f"{risk_tag}: Medical consultation is strongly recommended.")
    else:
        st.success(f"{risk_tag}: Keep maintaining a healthy lifestyle.")

    st.markdown("---")
    st.info("📌 This is a predictive tool. Always consult a licensed medical professional for diagnosis or treatment.")

    # ---------------- SHAP Force Plot + Explanation ----------------
    st.markdown("## 🔬 Model Explanation (SHAP Force Plot)")

    try:
        feature_names = ['gender', 'age', 'hb', 'rbc', 'pcv', 'mcv', 'mch', 'mchc']

        # Adjust estimator name if needed
        cat_model = model.named_estimators_['cat']
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(input_scaled)

        shap.initjs()
        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=input_scaled[0],
            feature_names=feature_names,
            matplotlib=False
        )
        components.html(
            f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
            height=130
        )

        # AI-generated Explanation
        st.markdown("## 🧠 Top Contributing Features")
        shap_dict = {
            name: (value, shap_val)
            for name, value, shap_val in zip(feature_names, input_scaled[0], shap_values[0])
        }

        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)

        st.markdown("**Top influencing features:**")
        for feature, (value, shap_val) in sorted_features[:5]:
            direction = "⬆️ increased" if shap_val > 0 else "⬇️ decreased"
            color = "#e74c3c" if shap_val > 0 else "#27ae60"
            st.markdown(f"<span style='color:{color}'>→ <b>{feature}</b> = {value:.2f} ({direction} risk)</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ SHAP explanation failed.")
        st.exception(e)

# ----------------- Footer -----------------
st.markdown("""
---
<div style='text-align: center; font-size: 15px;'>
© HSTU×MBSTU 2025 All Rights Reserved.
</div>
""", unsafe_allow_html=True)
