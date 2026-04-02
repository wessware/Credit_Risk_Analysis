import streamlit as st
import pickle
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Kinga Ensemble Credit Score",
    layout="wide"
)

st.title("Kinga ML Ensemble Predictor")
st.caption("Production Ensemble Model - kinga_ensemble_0.pkl")

st.divider()

# ------------------------- 
# MODEL LOADING - TOP LEVEL
# -------------------------
@st.cache_resource
def load_model():
    model_path = "pickled_models/kingametric_xgb_1.pkl"
    model = joblib.load(model_path)
    st.success("✅ KingaMetricXGB Pipeline loaded")
    return model

model = load_model()

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    
    # Compute normalized features (from SQL logic)
    df['normalized_dti'] = np.clip(df['Outstanding_Debt'] / (df['Annual_Income'] + 1), 0, 1)
    df['normalized_utilization'] = np.clip(df['Credit_Utilization_Ratio'], 0, 1)
    df['normalized_emi'] = np.clip(df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    df['normalized_delinquency'] = np.clip(df['Num_of_Delayed_Payment'] / (df['Num_of_Loan'] + 1), 0, 1)
    df['normalized_savings'] = np.clip(df['Monthly_Balance'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    
    # Call model interaction features
    df = model.add_interaction_features(df)
    
    # Value-based Income_Q assignment (Solution 2 - no pandas binning issues)
    def assign_income_quartile(income):
        if pd.isna(income):
            return 'Q2'
        if income < 30000:
            return 'Q1'
        elif income < 60000:
            return 'Q2'
        elif income < 100000:
            return 'Q3'
        else:
            return 'Q4'
    df["Income_Q"] = df["Annual_Income"].apply(assign_income_quartile)
    
    # Fill NaN/inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Defaults for missing raw features
    defaults = {
        'Num_Bank_Accounts': 2.0,
        'Num_Credit_Card': 2.0,
        'Interest_Rate': 0.12,
        'Delayed_Dues': 0.0,
        'Amount_invested_monthly': 100.0,
        'Num_Credit_Inquiries': 1.0,
        'Changed_Credit_Limit': 0.0,
        'Credit_History_Age': 60.0
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v
    
    return df

# Feature info (common credit risk features - adjust per model)
FEATURES = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Type_of_Loan", "Delayed_Dues", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_History_Age", "Payment_Behaviour", "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance"
]

st.info("**XGB Top Features**: Borrower_Tier, normalized_delinquency_sq, ... (computed from raw inputs below)")

# ------------------------- 
# FORM - Primary raw features for XGB pipeline
# ------------------------- 
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
    
    col_cat1, col_cat2, col_cat3 = st.columns(3)
    with col_cat1:
        st.subheader("Payment_of_Min_Amount")
        Payment_of_Min_Amount = st.selectbox("Min Amount Payment", ["No", "Yes", "NM"], index=0)
    with col_cat2:
        st.subheader("Credit_Mix")
        Credit_Mix = st.selectbox("Credit Mix", ["Standard", "Good", "Poor"], index=0)
    with col_cat3:
        st.subheader("Borrower_Tier")
        Borrower_Tier = st.selectbox("Borrower Tier", ["A", "B", "C", "D"], index=1)
        
    col_extra1, col_extra2 = st.columns(2)
    with col_extra1:
        st.subheader("Accounts")
        Num_Credit_Card = st.number_input("Credit Cards", min_value=0, value=2)
        Num_Bank_Accounts = st.number_input("Bank Accounts", min_value=0, value=3)
    
    submitted = st.form_submit_button("🚀 Predict with XGB Pipeline", use_container_width=True)

if submitted:
    # Prepare input data
    input_data = {
        "Annual_Income": Annual_Income,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_of_Loan": Num_of_Loan,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
        "Monthly_Balance": Monthly_Balance,
        "Outstanding_Debt": Outstanding_Debt,
        "Credit_History_Age": Credit_History_Age,
        "Amount_invested_monthly": Amount_invested_monthly,
        "Num_Credit_Inquiries": Num_Credit_Inquiries,
        "Changed_Credit_Limit": Changed_Credit_Limit,
        "Num_Credit_Card": Num_Credit_Card,
        "Num_Bank_Accounts": Num_Bank_Accounts,
        "Payment_of_Min_Amount": Payment_of_Min_Amount,
        "Credit_Mix": Credit_Mix,
        "Borrower_Tier": Borrower_Tier
    }
    
    try:
        # Preprocess and predict
        df_processed = preprocess_input(input_data)
        proba = model.predict_proba(df_processed, already_encoded=False)[:, 1][0]
        pred_class = model.predict(df_processed)[0]
        
        # FICO-style score (higher proba[risky] → lower score)
        risk_prob = proba
        fico_score = int(850 - (risk_prob * 550))
        
        rating = "EXCELLENT" if fico_score >= 750 else "GOOD" if fico_score >= 700 else "FAIR" if fico_score >= 650 else "POOR" if fico_score >= 550 else "VERY POOR"
        
        st.divider()
        col_score, col_risk = st.columns([1, 2])
        with col_score:
            st.metric("XGB FICO Score", fico_score)
        with col_risk:
            st.markdown(f"**Risk Grade: {rating}**")
            st.metric("Risk Probability", f"{risk_prob:.1%}", delta=f"Class: {'Default' if pred_class==1 else 'Safe'}")
        
        color = {"EXCELLENT": "🟢", "GOOD": "🔵", "FAIR": "🟡", "POOR": "🟠", "VERY POOR": "🔴"}[rating]
        st.markdown(f"### {color} **Kinga XGB Pipeline: {rating}**")
        
        # Show processed features (top 20)
        top20 = ['Borrower_Tier', 'normalized_delinquency_sq', 'normalized_delinquency', 'normalized_delinquency_log', 'Credit_Mix', 'Payment_Instability', 'Num_of_Loan', 'Risk_Index', 'normalized_emi', 'Obligation_Ratio', 'Num_Credit_Inquiries', 'Monthly_Inhand_Salary', 'Changed_Credit_Limit', 'Repayment_Stress', 'normalized_savings', 'Amount_invested_monthly', 'Total_EMI_per_month', 'Debt_Stress', 'Income_Delinq', 'Net_Cash_Flow']
        available_top = [f for f in top20 if f in df_processed.columns]
        st.subheader("Top 20 Features (processed)")
        st.dataframe(df_processed[available_top], use_container_width=True)
        
        st.subheader("Raw Inputs")
        raw_cols = list(input_data.keys())
        st.json(input_data)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Ensure all inputs valid. Model preprocess expects dataset-like raw features.")

st.caption("Powered by kingametric_xgb_1.pkl - Kingametric XGB Pipeline")
