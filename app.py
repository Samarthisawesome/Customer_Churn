import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

if os.path.exists("churn_model.pkl"):
    import subprocess
    subprocess.run(["python", "churn_prediction.py"])

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL, SCALER & FEATURE COLUMNS
# ─────────────────────────────────────────
@st.cache_resource  # loads once, stays in memory — faster app
def load_artifacts():
    model   = pickle.load(open("churn_model.pkl", "rb"))
    scaler  = pickle.load(open("scaler.pkl", "rb"))
    columns = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, columns

model, scaler, feature_columns = load_artifacts()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title(" Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they are likely to churn.")
st.divider()

# ─────────────────────────────────────────
# INPUT FORM — 3 COLUMNS LAYOUT
# ─────────────────────────────────────────
st.subheader(" Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account Info**")
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

with col2:
    st.markdown("**Charges**")
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges   = st.slider("Total Charges ($)", 0.0, 9000.0, float(monthly_charges * tenure))

    st.markdown("**Demographics**")
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Has Partner", ["Yes", "No"])
    dependents      = st.selectbox("Has Dependents", ["Yes", "No"])

with col3:
    st.markdown("**Services**")
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support    = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies= st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.divider()

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
if st.button(" Predict Churn", use_container_width=True):

    # ── Step 1: Build a raw input dict (same structure as original CSV) ──
    input_dict = {
        "gender"           : gender,
        "SeniorCitizen"    : 1 if senior == "Yes" else 0,
        "Partner"          : partner,
        "Dependents"       : dependents,
        "tenure"           : tenure,
        "PhoneService"     : phone_service,
        "MultipleLines"    : multiple_lines,
        "InternetService"  : internet,
        "OnlineSecurity"   : online_security,
        "OnlineBackup"     : online_backup,
        "DeviceProtection" : device_protect,
        "TechSupport"      : tech_support,
        "StreamingTV"      : streaming_tv,
        "StreamingMovies"  : streaming_movies,
        "Contract"         : contract,
        "PaperlessBilling" : paperless,
        "PaymentMethod"    : payment_method,
        "MonthlyCharges"   : monthly_charges,
        "TotalCharges"     : total_charges
    }

    # ── Step 2: Convert to DataFrame ──
    input_df = pd.DataFrame([input_dict])

    # ── Step 3: One-hot encode (same as training) ──
    input_encoded = pd.get_dummies(input_df)

    # ── Step 4: Align columns with training data ──
    # The app input may have fewer columns than training (e.g. some categories
    # not selected). We add missing columns as 0 and drop any extras.
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # ── Step 5: Scale using the SAME scaler from training ──
    input_scaled = scaler.transform(input_encoded)

    # ── Step 6: Predict ──
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of churn

    # ─────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────
    st.subheader("Prediction Results")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error(" This customer is likely to CHURN")
        else:
            st.success("This customer is likely to STAY")

    with res_col2:
        st.metric(
            label="Churn Probability",
            value=f"{probability * 100:.1f}%"
        )

    with res_col3:
        risk = "High Risk" if probability > 0.7 else " Medium Risk" if probability > 0.4 else " Low Risk"
        st.metric(label="Risk Level", value=risk)

    # ── Probability bar ──
    st.markdown("**Churn Probability Breakdown**")
    prob_df = pd.DataFrame({
        "Outcome"     : ["Will Stay", "Will Churn"],
        "Probability" : [1 - probability, probability]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

    # ── Key factors ──
    st.markdown("** Key Risk Factors to Watch**")
    tips = []
    if contract == "Month-to-month":
        tips.append("Customer is on a **month-to-month contract** — highest churn risk contract type")
    if monthly_charges > 70:
        tips.append(f" Monthly charges are **${monthly_charges}** — high charges correlate with churn")
    if tenure < 12:
        tips.append(f" Customer tenure is only **{tenure} months** — newer customers churn more")
    if internet == "Fiber optic" and online_security == "No":
        tips.append(" Has **Fiber optic** but no **Online Security** — common churn profile")
    if payment_method == "Electronic check":
        tips.append(" Pays by **Electronic check** — this group has the highest churn rate")
    if not tips:
        tips.append(" No major churn risk factors detected for this customer")
    for tip in tips:
        st.markdown(f"- {tip}")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Model: Tuned Random Forest · Dataset: Telco Customer Churn")
