import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pipelines.kingametric_xgb_pipeline import KingaMetricXGB

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
    st.info(f"Model feature_names len: {len(model.feature_names) if hasattr(model, 'feature_names') else 'N/A'}")
    st.info(f"Model xgb n_features_in: {model.xgb_model.n_features_in_}")
    return model

model = load_model()

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    
    # Compute normalized features
    df['normalized_dti'] = np.clip(df['Outstanding_Debt'] / (df['Annual_Income'] + 1), 0, 1)
    df['normalized_utilization'] = np.clip(df['Credit_Utilization_Ratio'], 0, 1)
    df['normalized_emi'] = np.clip(df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    df['normalized_delinquency'] = np.clip(df['Num_of_Delayed_Payment'] / (df['Num_of_Loan'] + 1), 0, 1)
    df['normalized_savings'] = np.clip(df['Monthly_Balance'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1)
    
    # Income_Q bins
    income = df["Annual_Income"].iloc[0]
    if income < 30000:
        df["Income_Q"] = 'Q1'
    elif income < 60000:
        df["Income_Q"] = 'Q2'
    elif income < 100000:
        df["Income_Q"] = 'Q3'
    else:
        df["Income_Q"] = 'Q4'
    
    # Interaction features (from pipeline)
    df['Debt_Stress'] = df['normalized_dti'] * df['normalized_utilization']
    df['Repayment_Stress'] = df['normalized_emi'] * df['normalized_delinquency']
    df['Income_Delinq'] = df["Annual_Income"] * df['normalized_delinquency']
    df['Loan_DTI'] = df["Num_of_Loan"] * df['normalized_dti']
    df['Risk_Index'] = (df['normalized_dti'] + df['normalized_utilization'] + df['normalized_delinquency']) / 3
    df['Credit_Exposure'] = df["Num_Credit_Card"] * df['Credit_Utilization_Ratio']
    df['Credit_Depth'] = df["Num_Credit_Card"] + df["Num_Bank_Accounts"]
    df['Liquidity_Buffer'] = df['normalized_savings']
    df['Obligation_Ratio'] = df['Total_EMI_per_month'] / df['Monthly_Inhand_Salary']
    df['Net_Cash_Flow'] = df['Monthly_Balance'] - df['Total_EMI_per_month']
    df['Payment_Instability'] = df['Num_of_Delayed_Payment'] > 0
    df['credit_mix_quality'] = df['Credit_Mix'].map({'Poor': 0, 'Standard': 0.5, 'Good': 1}).fillna(0.5)
    df['normalized_credit_history'] = df['Credit_History_Age'] / 840  # max 70yr
    df['normalized_inquiry_intensity'] = np.clip(df['Num_Credit_Inquiries'] / 10, 0, 1)
    df['normalized_utilization_risk'] = df['normalized_utilization'] * df['Risk_Index']
    
    # Poly features
    for feat in ['normalized_dti', 'normalized_delinquency']:
        df[f'{feat}_sq'] = df[feat]**2
        df[f'{feat}_log'] = np.log1p(df[feat])
    
    # Defaults
    defaults = {
        'Interest_Rate': 0.12, 'Delayed_Dues': 0.0, 'Payment_Behaviour': 'Low_spent_Small_value_payments'
    }
    for k, v in defaults.items():
        df[k] = df.get(k, v)
    
    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
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
        
        st.info(f"Preprocessed shape: {df_processed.shape}")
        st.info("df_processed columns: " + ', '.join(df_processed.columns.tolist()))
        st.info("Model feature_names len: " + str(len(model.feature_names)))
        
        # Ensure all feature_names present, fill missing
        feature_names = model.feature_names
        for fn in feature_names:
            if fn not in df_processed.columns:
                df_processed[fn] = 0.0
        df_ready = df_processed[feature_names].fillna(0)
        
        st.info(f"df_ready columns count: {len(df_ready.columns)} == expected {len(feature_names)}")
        
        # Convert all to numeric
        for col in df_ready.columns:
            if df_ready[col].dtype == 'object':
                if col == 'Credit_Mix':
                    df_ready[col] = df_ready[col].map({'Poor': 0, 'Standard': 0.5, 'Good': 1}).fillna(0.5)
                elif col == 'Income_Q':
                    df_ready[col] = df_ready[col].map({'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}).fillna(1)
                else:
                    df_ready[col] = df_ready[col].astype('category').cat.codes.astype(float)
            df_ready[col] = pd.to_numeric(df_ready[col], errors='coerce').fillna(0)
        
        st.success(f"Ready numeric shape: {df_ready.shape}")
        
        st.success(f"Ready numeric shape: {df_ready.shape}")
        
        risk_prob = model.xgb_model.predict_proba(df_ready)[0, 1]
        pred_class = model.xgb_model.predict(df_ready)[0]
        
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
        
        st.subheader("Top Features")
        if hasattr(model, 'feature_names'):
            top20_cols = model.feature_names[:20] if len(model.feature_names) >= 20 else model.feature_names
        else:
            top20_cols = list(df_processed.columns)[:20]
        st.dataframe(df_processed[top20_cols], use_container_width=True)
        
        st.subheader("Raw Inputs")
        st.json(input_data)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Debug: df_processed columns:", list(df_processed.columns) if 'df_processed' in locals() else "N/A")
        st.write("Model feature_names:", getattr(model, 'feature_names', 'N/A'))

st.caption("Kingametric XGB Pipeline v1")
