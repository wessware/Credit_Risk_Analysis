import streamlit as st
import pickle
import pandas as pd
import hashlib
import joblib
import numpy as np

# ------------------------- 
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Kinga Ensemble Credit Score",
    layout="wide"
)

st.title("🏆 Kinga ML Ensemble Predictor")
st.caption("Production Ensemble Model - kinga_ensemble_0.pkl")

st.divider()

# ------------------------- 
# MODEL LOADING - TOP LEVEL
# -------------------------
@st.cache_resource
def load_model():
    model_path = "pickled_models/kinga_ensemble_0.pkl"
    try:
        # Try joblib first (better for scikit ensemble)
        model = joblib.load(model_path)
        st.success("✅ Ensemble loaded (joblib)")
        return model
    except:
        # Fallback pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Ensemble loaded (pickle)")
        return model

model = load_model()

# Feature info (common credit risk features - adjust per model)
FEATURES = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Type_of_Loan", "Delayed_Dues", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_History_Age", "Payment_Behaviour", "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance"
]

st.info(f"Model expects {len(FEATURES)} features. Adjust form as needed.")

# ------------------------- 
# FORM - Same structure as SQL version
# -------------------------
with st.form("ensemble_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income & Debt")
        Annual_Income = st.number_input("Annual Income", min_value=1.0, value=50000.0)
        Monthly_Inhand_Salary = st.number_input("Monthly Salary", min_value=1.0, value=4000.0)
        Outstanding_Debt = st.number_input("Debt", min_value=0.0, value=10000.0)
    with col2:
        st.subheader("Loans")
        Total_EMI_per_month = st.number_input("Monthly EMI", min_value=0.0, value=500.0)
        Num_of_Loan = st.number_input("Loans", min_value=1, max_value=50, value=2)
        Num_of_Delayed_Payment = st.number_input("Delays", min_value=0, value=1)
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Credit")
        Credit_History_Age = st.number_input("History (months)", min_value=0, max_value=600, value=60)
        Credit_Utilization_Ratio = st.slider("Utilization", 0.0, 1.0, 0.3)
    with col4:
        st.subheader("Balance")
        Monthly_Balance = st.number_input("Monthly Balance", min_value=0.0, value=1000.0)

    submitted = st.form_submit_button("🎯 Predict Risk Score", use_container_width=True)

if submitted:
    # Prepare input matching model expected features
    input_data = {
        "Annual_Income": Annual_Income,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_of_Loan": Num_of_Loan,
        "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
        "Credit_History_Age": Credit_History_Age,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Monthly_Balance": Monthly_Balance,
        "Outstanding_Debt": Outstanding_Debt,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio
    }
    
    try:
        # Convert to DataFrame (standard for sklearn ensembles)
        df = pd.DataFrame([input_data])
        
        # Handle missing features (fill common defaults)
        for feature in FEATURES:
            if feature not in df.columns:
                df[feature] = 0.0  # or model default
        
        # Predict
        prediction = model.predict_proba(df)[0]  # Probability for risk score
        
        # FICO-style score (0-1 risk → 850-300 score)
        risk_prob = prediction[1] if len(prediction) > 1 else prediction[0]
        fico_score = int(850 - (risk_prob * 550))
        
        rating = "EXCELLENT" if fico_score >= 750 else "GOOD" if fico_score >= 700 else "FAIR" if fico_score >= 650 else "POOR" if fico_score >= 550 else "VERY POOR"
        
        st.divider()
        col_score, col_rating = st.columns([1, 2])
        with col_score:
            st.metric("Ensemble FICO Score", fico_score)
        with col_rating:
            st.markdown(f"**Risk Grade: {rating}**")
        
        color = {"EXCELLENT": "🟢", "GOOD": "🔵", "FAIR": "🟡", "POOR": "🟠", "VERY POOR": "🔴"}[rating]
        st.markdown(f"### {color} **Kinga Ensemble: {rating} Profile**")
        
        # Show input used
        st.subheader("Model Input")
        st.dataframe(df[FEATURES[:8]], use_container_width=True)  # First 8 for display
        
        # Proba breakdown if binary
        if len(prediction) > 1:
            st.subheader("Risk Breakdown")
            col1, col2 = st.columns(2)
            col1.metric("Safe", f"{prediction[0]:.1%}")
            col2.metric("Risky", f"{prediction[1]:.1%}")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Model may need exact features. Check pickled_models/kinga_ensemble_0.pkl requirements.")

st.caption("Powered by kinga_ensemble_0.pkl - Kingametric ML")
