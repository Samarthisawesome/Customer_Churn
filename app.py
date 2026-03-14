import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ─────────────────────────────────────────
# TRAIN & SAVE MODEL (only if not already saved)
# ─────────────────────────────────────────
@st.cache_resource  # runs once, result stays in memory for the whole session
def train_model():
    # ── Load & Clean ──
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop("customerID", axis=1)

    # ── Encode ──
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, drop_first=True)

    # ── Split ──
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ── Tune & Train ──
    param_grid = {
        "n_estimators"      : [100, 200, 300],
        "max_depth"         : [None, 10, 20],
        "min_samples_split" : [2, 5],
        "min_samples_leaf"  : [1, 2],
        "max_features"      : ["sqrt", "log2"],
        "bootstrap"         : [True, False]
    }
    rf_search = RandomizedSearchCV(
        estimator           = RandomForestClassifier(class_weight="balanced", random_state=42),
        param_distributions = param_grid,
        n_iter              = 20,        # reduced for faster cloud startup
        scoring             = "roc_auc",
        cv                  = 3,
        random_state        = 42,
        n_jobs              = -1
    )
    rf_search.fit(X_train_scaled, y_train)
    best_model = rf_search.best_estimator_

    return best_model, scaler, list(X.columns)

# ─────────────────────────────────────────
# LOAD OR TRAIN
# ─────────────────────────────────────────
if os.path.exists("churn_model.pkl"):
    # Already trained locally — just load the files
    model          = pickle.load(open("churn_model.pkl", "rb"))
    scaler         = pickle.load(open("scaler.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
else:
    # On Streamlit Cloud — train directly inside the app
    with st.spinner("Training model for the first time... this takes ~1 minute"):
        model, scaler, feature_columns = train_model()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they are likely to churn.")
st.divider()

# ─────────────────────────────────────────
# INPUT FORM — 3 COLUMNS LAYOUT
# ─────────────────────────────────────────
st.subheader("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account Info**")
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
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
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

with col3:
    st.markdown("**Services**")
    phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet         = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup    = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.divider()

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
if st.button("Predict Churn", use_container_width=True):

    # ── Step 1: Build raw input dict ──
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

    # ── Step 3: One-hot encode ──
    input_encoded = pd.get_dummies(input_df)

    # ── Step 4: Align columns with training data ──
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # ── Step 5: Scale ──
    input_scaled = scaler.transform(input_encoded)

    # ── Step 6: Predict ──
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ─────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────
    st.subheader("Prediction Results")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("This customer is likely to CHURN")
        else:
            st.success("This customer is likely to STAY")

    with res_col2:
        st.metric(label="Churn Probability", value=f"{probability * 100:.1f}%")

    with res_col3:
        risk = "High Risk" if probability > 0.7 else "Medium Risk" if probability > 0.4 else "Low Risk"
        st.metric(label="Risk Level", value=risk)

    # ── Probability bar ──
    st.markdown("**Churn Probability Breakdown**")
    prob_df = pd.DataFrame({
        "Outcome"     : ["Will Stay", "Will Churn"],
        "Probability" : [1 - probability, probability]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

    # ── Key risk factors ──
    st.markdown("**Key Risk Factors to Watch**")
    tips = []
    if contract == "Month-to-month":
        tips.append("Customer is on a **month-to-month contract** — highest churn risk contract type")
    if monthly_charges > 70:
        tips.append(f"Monthly charges are **${monthly_charges}** — high charges correlate with churn")
    if tenure < 12:
        tips.append(f"Customer tenure is only **{tenure} months** — newer customers churn more")
    if internet == "Fiber optic" and online_security == "No":
        tips.append("Has **Fiber optic** but no **Online Security** — common churn profile")
    if payment_method == "Electronic check":
        tips.append("Pays by **Electronic check** — this group has the highest churn rate")
    if not tips:
        tips.append("No major churn risk factors detected for this customer")
    for tip in tips:
        st.markdown(f"- {tip}")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Model: Tuned Random Forest · Dataset: Telco Customer Churn")
