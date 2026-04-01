import streamlit as st
import pandas as pd
import joblib
import numpy as np
from kingametric_xgb_pipeline import KingaMetricXGB

# ------------------------- 
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Kinga XGB Credit Score",
    layout="wide"
)

st.title("🏆 Kinga XGB Pipeline Predictor")
st.caption("Powered by kingametric_xgb_1.pkl")

st.divider()

# ------------------------- 
# MODEL LOADING
# -------------------------
@st.cache_resource
def load_model():
    model_path = "pickled_models/kingametric_xgb_1.pkl"
    model = joblib.load(model_path)
    st.success("✅ KingaMetricXGB loaded")
    return model

model = load_model()

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    
    # Compute normalized features from SQL formulas
    df['normalized_dti'] = np.clip(df['Outstanding_Debt'] / (df['Annual_Income'] + 1), 0, 1)
    df['normalized_utilization'] = np.clip(df['Credit_Utilization_Ratio'], 0, 1)
    df['normalized_emi'] = np.clip(df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    df['normalized_delinquency'] = np.clip(df['Num_of_Delayed_Payment'] / (df['Num_of_Loan'] + 1), 0, 1)
    df['normalized_savings'] = np.clip(df['Monthly_Balance'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    
    # Model interaction features
    df = model.add_interaction_features(df)
    
    # Income bins - fix for single row
    df["Income_Q"] = pd.qcut(df["Annual_Income"], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop').astype(str)
    
    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Default missing raws
    defaults = {
        'Num_Bank_Accounts': 2.0, 'Num_Credit_Card': 2.0, 'Interest_Rate': 0.12,
        'Delayed_Dues': 0.0, 'Amount_invested_monthly': 100.0,
        'Num_Credit_Inquiries': 1.0, 'Changed_Credit_Limit': 0.0
    }
    for k, v in defaults.items():
        df[k] = df.get(k, v)
    
    return df

st.info("Enter raw features → Auto-computes normalized/interactions → XGB prediction")

# ------------------------- 
# FORM - Primary raw features
# ------------------------- 
with st.form("xgb_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Income")
        Annual_Income = st.number_input("Annual Income", min_value=1.0, value=50000.0)
        Monthly_Inhand_Salary = st.number_input("Monthly Salary", min_value=0.0, value=4000.0)
        Outstanding_Debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
        Monthly_Balance = st.number_input("Monthly Balance", min_value=0.0, value=500.0)
    with col2:
        st.subheader("Loans & Delinquency")
        Num_of_Loan = st.number_input("Num Loans", min_value=0, value=2)
        Total_EMI_per_month = st.number_input("Total EMI", min_value=0.0, value=500.0)
        Num_of_Delayed_Payment = st.number_input("Delayed Payments", min_value=0, value=1)
        Num_Credit_Inquiries = st.number_input("Credit Inquiries", min_value=0, value=2)
    with col3:
        st.subheader("Credit")
        Credit_Utilization_Ratio = st.slider("Utilization", 0.0, 1.0, 0.3)
        Changed_Credit_Limit = st.number_input("Limit Changes", min_value=-10.0, value=0.0)
        Credit_History_Age = st.number_input("History Age (mo)", min_value=0, value=60)
        Amount_invested_monthly = st.number_input("Investments", min_value=0.0, value=200.0)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        Payment_of_Min_Amount = st.selectbox("Min Payment", ["No", "Yes", "NM"])
    with col_c2:
        Credit_Mix = st.selectbox("Credit Mix", ["Standard", "Good", "Poor"])
    with col_c3:
        Borrower_Tier = st.selectbox("Tier", ["A", "B", "C", "D"])
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        Num_Credit_Card = st.number_input("Credit Cards", min_value=0, value=2)
        Num_Bank_Accounts = st.number_input("Bank Accounts", min_value=0, value=3)
    
    submitted = st.form_submit_button("🚀 Predict Risk", use_container_width=True)

if submitted:
    input_data = {
        'Annual_Income': Annual_Income, 'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
        'Num_of_Loan': Num_of_Loan, 'Total_EMI_per_month': Total_EMI_per_month,
        'Num_of_Delayed_Payment': Num_of_Delayed_Payment, 'Credit_Utilization_Ratio': Credit_Utilization_Ratio,
        'Monthly_Balance': Monthly_Balance, 'Outstanding_Debt': Outstanding_Debt,
        'Credit_History_Age': Credit_History_Age, 'Amount_invested_monthly': Amount_invested_monthly,
        'Num_Credit_Inquiries': Num_Credit_Inquiries, 'Changed_Credit_Limit': Changed_Credit_Limit,
        'Num_Credit_Card': Num_Credit_Card, 'Num_Bank_Accounts': Num_Bank_Accounts,
        'Payment_of_Min_Amount': Payment_of_Min_Amount, 'Credit_Mix': Credit_Mix,
        'Borrower_Tier': Borrower_Tier
    }
    
    try:
        df_processed = preprocess_input(input_data)
        risk_prob = model.predict_proba(df_processed)[0, 1]
        pred_class = model.predict(df_processed)[0]
        
        fico = int(850 - (risk_prob * 550))
        rating = "EXCELLENT" if fico >= 750 else "GOOD" if fico >= 700 else "FAIR" if fico >= 650 else "POOR" if fico >= 550 else "VERY POOR"
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("FICO Score", fico)
        with col2:
            st.markdown(f"### **{rating}**")
            st.metric("Risk Proba", f"{risk_prob:.1%}", f"Predicted: {'Default' if pred_class else 'Safe'}")
        
        colors = {"EXCELLENT": "🟢", "GOOD": "🔵", "FAIR": "🟡", "POOR": "🟠", "VERY POOR": "🔴"}
        st.markdown(f"{colors[rating]} **Kinga XGB: {rating} Risk**")
        
        st.subheader("Top 20 Processed Features")
        top20 = ['Borrower_Tier', 'normalized_delinquency_sq', 'normalized_delinquency', 'normalized_delinquency_log', 'Credit_Mix', 'Payment_Instability', 'Num_of_Loan', 'Risk_Index', 'normalized_emi', 'Obligation_Ratio', 'Num_Credit_Inquiries', 'Monthly_Inhand_Salary', 'Changed_Credit_Limit', 'Repayment_Stress', 'normalized_savings', 'Amount_invested_monthly', 'Total_EMI_per_month', 'Debt_Stress', 'Income_Delinq', 'Net_Cash_Flow']
        st.dataframe(df_processed[top20], use_container_width=True)
        
        st.subheader("Raw Inputs")
        st.json(input_data)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("Kingametric XGB Pipeline v1")
