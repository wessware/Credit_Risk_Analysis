import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pipelines.kingametric_xgb_pipeline import KingaMetricXGB

st.set_page_config(page_title="Kinga XGB Credit Score", layout="wide")

st.title("🏆 Kinga XGB Pipeline Predictor")

@st.cache_resource
def load_model():
    model = joblib.load("pickled_models/kingametric_xgb_1.pkl")
    return model

model = load_model()

# Add all known feature_names from debug with defaults
feature_defaults = {
    'Num_of_Delayed_Payment': 1.0, 'Net_Cash_Flow': 0.0, 'Amount_invested_monthly': 100.0,
    'Repayment_Stress': 0.0, 'Payment_Instability': 0.0, 'Risk_Index': 0.0,
    'normalized_delinquency': 0.0, 'credit_mix_quality': 1.0, 'Credit_History_Age': 60.0,
    'Obligation_Ratio': 0.3, 'Loan_DTI': 0.0, 'normalized_emi': 0.0,
    'Changed_Credit_Limit': 0.0, 'Monthly_Inhand_Salary': 4000.0, 'Total_EMI_per_month': 500.0,
    'Credit_Mix': 'Standard', 'normalized_utilization_risk': 0.0, 'Num_Bank_Accounts': 2.0,
    'normalized_savings': 0.0, 'Income_Delinq': 0.0, 'normalized_credit_history': 0.0,
    'normalized_inquiry_intensity': 0.0, 'Num_of_Loan': 2.0, 'Liquidity_Buffer': 0.0,
    'Interest_Rate': 0.12, 'Credit_Depth': 5.0, 'normalized_dti_log': 0.0,
    'normalized_dti_sq': 0.0, 'Num_Credit_Card': 2.0, 'Debt_Stress': 0.0,
    'Annual_Income': 50000.0, 'Outstanding_Debt': 1000.0, 'Credit_Utilization_Ratio': 0.3
}

with st.form("predict_form"):
    input_data = {}
    for feat, default in feature_defaults.items():
        if isinstance(default, (int, float)):
            input_data[feat] = st.number_input(feat, value=default)
        else:
            input_data[feat] = st.selectbox(feat, options=['Standard', 'Good', 'Poor', 'Bad'], index=0 if 'Bad' not in str(default) else 3)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([input_data])
    
    try:
        proba = model.predict_proba(df)[0, 1]
        pred = model.predict(df)[0]
        
        st.metric("Risk Probability", f"{proba:.1%}", delta=f"{'Default' if pred else 'Safe'}")
        fico = int(850 - (proba * 550))
        st.metric("FICO Score", fico)
        
        st.success("Prediction successful!")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("df columns:", list(df.columns))
        st.write("df shape:", df.shape)
        if hasattr(model, 'feature_names'):
            st.write("model.feature_names:", model.feature_names)
