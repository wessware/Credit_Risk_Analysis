import streamlit as st
import pickle
import pandas as pd
import joblib
import numpy as np

# TRAINING_MEDIANS: Fixed medians mimicking training data for robust NaN filling
TRAINING_MEDIANS = {
    'Annual_Income': 50000.0,
    'Monthly_Inhand_Salary': 4000.0,
    'Num_Bank_Accounts': 4.0,
    'Num_Credit_Card': 2.0,
    'Interest_Rate': 0.12,
    'Num_of_Loan': 2.0,
    'Delayed_Dues': 0.0,
    'Num_of_Delayed_Payment': 1.0,
    'Changed_Credit_Limit': 0.0,
    'Num_Credit_Inquiries': 1.0,
    'Outstanding_Debt': 1000.0,
    'Credit_History_Age': 60.0,
    'Total_EMI_per_month': 500.0,
    'Amount_invested_monthly': 100.0,
    'Monthly_Balance': 500.0,
    'Credit_Utilization_Ratio': 0.3,
    'normalized_dti': 0.05,
    'normalized_utilization': 0.3,
    'normalized_emi': 0.2,
    'normalized_delinquency': 0.1,
    'normalized_savings': 0.2,
    'Total_Payment_Made': 1000.0,
    'Age': 35.0,
    'Credit_Default': 0.0,
    'Num_Payment_Late': 0.0,
    'normalized_delinquency_sq': 0.01,
    'normalized_delinquency_log': 0.1,
    'normalized_emi_sq': 0.04,
    'normalized_savings_sq': 0.04,
    'Payment_Instability': 1.0,
    'Risk_Index': 0.005,
    'Obligation_Ratio': 0.125,
    'Repayment_Stress': 1.0,
    'Debt_Stress': 0.02,
    'Income_Delinq': 25000.0,
    'Net_Cash_Flow': 3500.0
    # Add more as needed
}

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
    import sys
    sys.path.insert(0, 'c:/Users/Admin/Documents/PROJECTS/credit_score')
    model_path = "c:/Users/Admin/Documents/PROJECTS/credit_score/pickled_models/kingametric_xgb_1.pkl"
    model = joblib.load(model_path)
    st.success("KingaMetricXGB Pipeline loaded")
    return model

model = load_model()

# Extract model attributes for preprocessing alignment
expected_features = getattr(model, 'feature_names', None)
expected_dim = len(expected_features) if expected_features else 30
st.info(f"Model loaded. Expected input dimension: {expected_dim}")

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    
    # Compute normalized features first (required by add_interaction_features)
    df['normalized_dti'] = np.clip(df['Outstanding_Debt'] / (df['Annual_Income'] + 1), 0, 1)
    df['normalized_utilization'] = np.clip(df['Credit_Utilization_Ratio'], 0, 1)
    df['normalized_emi'] = np.clip(df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    df['normalized_delinquency'] = np.clip(df['Num_of_Delayed_Payment'] / (df['Num_of_Loan'] + 1), 0, 1)
    df['normalized_savings'] = np.clip(df['Monthly_Balance'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    
    # Use model's exact preprocessing methods
    df = model.add_interaction_features(df)
    
    # Target encode if encoder available (raw input for single row)
    if hasattr(model, 'target_encoder') and model.target_encoder is not None:
        try:
            df = model.target_encode(df, fit=False)
        except Exception as te_err:
            st.warning(f"Target encoder transform failed: {te_err}, skipping")
    
    # Align to expected features exactly (model's predict uses self.feature_names)
    if expected_features is not None:
        # Ensure all expected features exist, fill missing with training medians
        for col in expected_features:
            if col not in df.columns:
                df[col] = TRAINING_MEDIANS.get(col, 0.0)
        
        # Select ONLY the expected features in their training order
        df = df[expected_features]
    else:
        # Fallback: sort unique and fix to 30
        cols_sorted = sorted(set(df.columns))
        df = df[cols_sorted[:30]]
        while len(df.columns) < 30:
            df[f'pad_feature_{len(df.columns)}'] = 0.0
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Debug
    st.info(f"Processed shape: {df.shape} (target: {expected_dim if expected_features else 30}), cols sample: {list(df.columns)[:10]}...")
    return df

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
    
    col_cat1, col_cat2, col_cat3, col_cat4 = st.columns(4)
    with col_cat1:
        st.subheader("Payment_of_Min_Amount")
        Payment_of_Min_Amount = st.selectbox("Min Amount Payment", ["No", "Yes", "NM"], index=0)
    with col_cat2:
        st.subheader("Credit_Mix")
        Credit_Mix = st.selectbox("Credit Mix", ["Standard", "Good", "Poor"], index=0)
    with col_cat3:
        st.subheader("Borrower_Tier")
        Borrower_Tier = st.selectbox("Borrower Tier", ["A", "B", "C", "D"], index=1)
    with col_cat4:
        st.subheader("Payment_Behaviour")
        Payment_Behaviour = st.selectbox("Payment Behaviour", ["RMPO", "low", "moderate", "high"], index=0)
        
    col_extra1, col_extra2, col_extra3, col_extra4 = st.columns(4)
    with col_extra1:
        st.subheader("Accounts")
        Num_Credit_Card = st.number_input("Credit Cards", min_value=0, value=2)
        Num_Bank_Accounts = st.number_input("Bank Accounts", min_value=0, value=3)
    with col_extra2:
        st.subheader("Rates & Dues")
        Interest_Rate = st.slider("Interest Rate", 0.0, 0.5, 0.12)
        Delayed_Dues = st.number_input("Delayed Dues", min_value=0.0, value=0.0)
    with col_extra3:
        st.subheader("Type of Loan")
        Type_of_Loan_options = ["Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan", "Mortgage Loan", "Student Loan", "Debt Consolidation"]
        Type_of_Loan = st.multiselect("Type of Loan (select all)", Type_of_Loan_options, default=["Personal Loan"])
    with col_extra4:
        st.subheader("Additional")
        Total_Payment_Made = st.number_input("Total Payment Made", min_value=0.0, value=1000.0)
        Age = st.slider("Age", 18, 80, 35)
        Credit_Default = st.selectbox("Credit Default", ["No", "Yes"], index=0)
        Num_Payment_Late = st.number_input("Num Payment Late", min_value=0, value=0)
    
    submitted = st.form_submit_button("Predict with XGB Pipeline", use_container_width=True)

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
        "Borrower_Tier": Borrower_Tier,
        "Interest_Rate": Interest_Rate,
        "Delayed_Dues": Delayed_Dues,
"Type_of_Loan": "; ".join(Type_of_Loan) if Type_of_Loan else "Personal Loan",
        "Payment_Behaviour": Payment_Behaviour,
        "Total_Payment_Made": Total_Payment_Made,
        "Age": Age,
        "Credit_Default": Credit_Default,
        "Num_Payment_Late": Num_Payment_Late
    }
    
    try:
        df_input = pd.DataFrame([input_data])
        # Use model's built-in predict methods directly (expects DataFrame)
        proba = model.predict_proba(df_input)[:, 1][0]
        pred_class = model.predict(df_input)[0]
        df_processed = preprocess_input(input_data)  # For display/debug
        st.write(f"Processed shape for display: {df_processed.shape}")
        
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
        top20 = ['Borrower_Tier', 'normalized_delinquency_sq', 'normalized_delinquency', 'normalized_delinquency_log', 
                 'Credit_Mix', 'Payment_Instability', 'Num_of_Loan', 'Risk_Index', 'normalized_emi', 'Obligation_Ratio', 
                 'Num_Credit_Inquiries', 'Monthly_Inhand_Salary', 'Changed_Credit_Limit', 'Repayment_Stress', 'normalized_savings', 
                 'Amount_invested_monthly', 'Total_EMI_per_month', 'Debt_Stress', 'Income_Delinq', 'Net_Cash_Flow']
        
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
