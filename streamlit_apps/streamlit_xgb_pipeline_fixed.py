import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.kingametric_xgb_pipeline_fixed import KingaMetricXGB

st.set_page_config(
    page_title="Kinga Ensemble Credit Score",
    layout="wide"
)

st.title("Kinga ML Ensemble Predictor")
st.caption("Production Ensemble Model - kinga_ensemble_0.pkl")

st.divider()

@st.cache_resource
def load_model():
    model_path = "pickled_models/kingametric_xgb_lean_1.pkl"
    model = joblib.load(model_path)
    st.success(f"✅ KingaMetricXGB Pipeline loaded. Features: {getattr(model, 'feature_names', 'N/A')}, dim: {len(getattr(model, 'feature_names', [])) if hasattr(model, 'feature_names') else 'N/A'}")
    return model

model = load_model()

expected_features = getattr(model, 'feature_names', None)
expected_dim = len(expected_features) if expected_features else 30
st.info(f"Model loaded. Expected input dimension: {expected_dim}")

def preprocess_for_display(input_data):
    """Mimic pipeline for display only"""
    df = pd.DataFrame([input_data])
    df['normalized_dti'] = np.clip(df['Outstanding_Debt'] / (df['Annual_Income'] + 1), 0, 1)
    df['normalized_utilization'] = np.clip(df['Credit_Utilization_Ratio'], 0, 1)
    df['normalized_emi'] = np.clip(df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    df['normalized_delinquency'] = np.clip(df['Num_of_Delayed_Payment'] / (df['Num_of_Loan'] + 1), 0, 1)
    df['normalized_savings'] = np.clip(df['Monthly_Balance'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    df['normalized_credit_history'] = np.clip(df['Credit_History_Age'] / 840, 0, 1)
    df = model.add_interaction_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    st.info(f"Display preview shape: {df.shape}")
    return df

with st.form("xgb_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Income")
        Annual_Income = st.number_input("Annual Income", min_value=1.0, value=50000.0, format="%i")
        Monthly_Inhand_Salary = st.number_input("Monthly Salary", min_value=0.0, value=4000.0)
        Outstanding_Debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
        Monthly_Balance = st.number_input("Monthly Balance", min_value=0.0, value=500.0)
    with col2:
        st.subheader("Loans & Delinq")
        Num_of_Loan = st.number_input("Number of Loans", min_value=0, value=2)
        Total_EMI_per_month = st.number_input("Total EMI/month", min_value=0.0, value=500.0)
        Num_of_Delayed_Payment = st.number_input("Delayed Payments", min_value=0, value=1)
        Num_Credit_Inquiries = st.number_input("Credit Inquiries", min_value=0, value=2)
    with col3:
        st.subheader("Credit")
        Credit_Utilization_Ratio = st.slider("Utilization Ratio", 0.0, 1.0, 0.3)
        Changed_Credit_Limit = st.number_input("Credit Limit Changes", min_value=-10.0, value=0.0)
        Credit_History_Age = st.number_input("History Age (months)", min_value=0, value=60)
        Amount_invested_monthly = st.number_input("Monthly Investments", min_value=0.0, value=200.0)
    
    col_cat1, col_cat2, col_cat3, col_cat4 = st.columns(4)
    with col_cat1:
        Payment_of_Min_Amount = st.selectbox("Min Amount Payment", ["No", "Yes", "NM"], index=0)
    with col_cat2:
        Credit_Mix = st.selectbox("Credit Mix", ["Standard", "Good", "Poor"], index=0)
    with col_cat3:
        Borrower_Tier = st.selectbox("Borrower Tier", ["Prime", "Near_Prime", "Subprime"], index=0)
    with col_cat4:
        Payment_Behaviour = st.selectbox("Payment Behaviour", ["RMPO", "low", "moderate", "high"], index=0)
        
    col_extra1, col_extra2, col_extra3, col_extra4 = st.columns(4)
    with col_extra1:
        Num_Credit_Card = st.number_input("Credit Cards", min_value=0, value=2)
        Num_Bank_Accounts = st.number_input("Bank Accounts", min_value=0, value=3)
    with col_extra2:
        Interest_Rate = st.slider("Interest Rate", 0.0, 0.5, 0.12)
        Delayed_Dues = st.number_input("Delayed Dues", min_value=0.0, value=0.0)
        Delay_from_due_date = st.number_input("Delay from due date", min_value=-100.0, value=-1.0)
    with col_extra3:
        Type_of_Loan_options = ["Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan", "Mortgage Loan", "Student Loan", "Debt Consolidation"]
        Type_of_Loan = st.multiselect("Type of Loan", Type_of_Loan_options, default=["Personal Loan"])
    with col_extra4:
        Total_Payment_Made = st.number_input("Total Payment Made", min_value=0.0, value=1000.0)
    
    submitted = st.form_submit_button("Predict with XGB Pipeline", use_container_width=True)

if submitted:
    input_data = {
        "Annual_Income": Annual_Income,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_Bank_Accounts": Num_Bank_Accounts,
        "Num_Credit_Card": Num_Credit_Card,
        "Interest_Rate": Interest_Rate,
        "Num_of_Loan": Num_of_Loan,
        "Type_of_Loan": "; ".join(Type_of_Loan),
        "Delayed_Dues": Delayed_Dues,
        "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
        "Changed_Credit_Limit": Changed_Credit_Limit,
        "Num_Credit_Inquiries": Num_Credit_Inquiries,
        "Credit_Mix": Credit_Mix,
        "Outstanding_Debt": Outstanding_Debt,
        "Credit_History_Age": Credit_History_Age,
        "Delay_from_due_date": Delay_from_due_date,
        "Payment_Behaviour": Payment_Behaviour,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Amount_invested_monthly": Amount_invested_monthly,
        "Monthly_Balance": Monthly_Balance,
        "Payment_of_Min_Amount": Payment_of_Min_Amount,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
        "Total_Payment_Made": Total_Payment_Made,
        "Borrower_Tier": Borrower_Tier
    }
    
    try:
        df_input = pd.DataFrame([input_data])
        proba = model.predict_proba(df_input)[:, 1][0]
        pred_class = model.predict(df_input)[0]
        df_processed = preprocess_for_display(input_data)
        
        risk_prob = proba
        fico_score = int(850 - (risk_prob * 550))
        
        rating = "EXCELLENT" if fico_score >= 750 else "GOOD" if fico_score >= 700 else "FAIR" if fico_score >= 650 else "POOR" if fico_score >= 550 else "VERY POOR"
        
        st.divider()
        col_score, col_risk = st.columns([1, 2])
        with col_score:
            st.metric("XGB FICO Score", fico_score)
        with col_risk:
            st.markdown(f"**Risk Grade: {rating}**")
            st.metric("Risk Probability", f"{risk_prob:.1%}", delta=f"{'Default' if pred_class==1 else 'Safe'}")
        
        color = {"EXCELLENT": "🟢", "GOOD": "🔵", "FAIR": "🟡", "POOR": "🟠", "VERY POOR": "🔴"}[rating]
        st.markdown(f"### {color} **Kinga XGB Pipeline: {rating}**")
        
        st.subheader("Processed Features")
        st.dataframe(df_processed[list(expected_features)])
        
        st.subheader("Raw Inputs")
        st.json(input_data)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Model expects raw dataset features.")

st.caption("Powered by kingametric_xgb_lean_1.pkl")
