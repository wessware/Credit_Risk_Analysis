import streamlit as st

import pandas as pd
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.kingametric_base_0 import KingaMetricXGB

st.set_page_config(
    page_title="Kinga Ensemble Credit Score",
    layout="wide"
)

st.title("Kinga ML Ensemble Predictor")
st.caption("Production Ensemble Model - kingametric_xgb_schema_locked.pkl")

st.divider()


@st.cache_resource
def load_model():
    model_path = "pickled_models/kingametric_xgb_schema_locked.pkl"
    model = joblib.load(model_path)
    raw_features = getattr(model, "raw_input_features", [])
    selected_features = getattr(model, "feature_names", [])
    st.success(
        f"KingaMetricXGB pipeline loaded. Raw inputs for feature engineering: {len(raw_features)}, selected model features: {len(selected_features)}"
    )
    return model


model = load_model()
raw_features = getattr(model, "raw_input_features", [])
expected_dim = getattr(model, "expected_n_features", len(getattr(model, "feature_names", [])))
st.info(f"Model loaded. Expected aligned feature dimension: {expected_dim}")

with st.form("xgb_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Income")
        Annual_Income = st.number_input("Annual Income", min_value=1.0, value=50000.0, format="%i")
        Monthly_Inhand_Salary = st.number_input("Monthly Salary", min_value=0.0, value=4000.0)
        Outstanding_Debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
        Monthly_Balance = st.number_input("Monthly Balance", min_value=0.0, value=500.0)
    with col2:
        st.subheader("Loans & Delinquency")
        Num_of_Loan = st.number_input("Number of Loans", min_value=0, value=2)
        Total_EMI_per_month = st.number_input("Total EMI/month", min_value=0.0, value=500.0)
        Num_Credit_Inquiries = st.number_input("Credit Inquiries", min_value=0, value=2)
        Num_of_Delayed_Payment = st.number_input("Delayed Payments", min_value=0, value=1)
    with col3:
        st.subheader("Credit")
        Credit_Utilization_Ratio = st.slider("Utilization Ratio", 0.0, 1.0, 0.3)
        Credit_History_Age = st.number_input("History Age (months)", min_value=0, value=60)
        Interest_Rate = st.slider("Interest Rate", 0.0, 0.5, 0.12)
        Changed_Credit_Limit = st.number_input("Credit Limit Changes", min_value=-10.0, value=0.0)

    col_cat1, col_cat2, col_cat3 = st.columns(3)
    with col_cat1:
        Payment_of_Min_Amount = st.selectbox("Min Amount Payment", ["No", "Yes", "NM", "Missing"], index=0)
    with col_cat2:
        Credit_Mix = st.selectbox("Credit Mix", ["Standard", "Good", "Poor", "Missing"], index=0)
    with col_cat3:
        Borrower_Tier = st.selectbox("Borrower Tier", ["Prime", "Near_Prime", "Subprime", "Missing"], index=0)

    col_extra1, col_extra2 = st.columns(2)
    with col_extra1:
        Num_Credit_Card = st.number_input("Credit Cards", min_value=0, value=2)
        Num_Bank_Accounts = st.number_input("Bank Accounts", min_value=0, value=3)
    with col_extra2:
        st.subheader("")
        st.caption("Delayed payment count is used only to derive secondary risk features. The raw leaky column is stripped before model scoring.")

    submitted = st.form_submit_button("Predict with XGB Pipeline", use_container_width=True)

if submitted:
    input_data = {
        "Annual_Income": Annual_Income,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_Bank_Accounts": Num_Bank_Accounts,
        "Num_Credit_Card": Num_Credit_Card,
        "Interest_Rate": Interest_Rate,
        "Num_of_Loan": Num_of_Loan,
        "Changed_Credit_Limit": Changed_Credit_Limit,
        "Num_Credit_Inquiries": Num_Credit_Inquiries,
        "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
        "Credit_Mix": Credit_Mix,
        "Outstanding_Debt": Outstanding_Debt,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
        "Credit_History_Age": Credit_History_Age,
        "Payment_of_Min_Amount": Payment_of_Min_Amount,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Monthly_Balance": Monthly_Balance,
        "Borrower_Tier": Borrower_Tier,
    }

    try:
        df_input = pd.DataFrame([input_data])
        if raw_features:
            df_input = df_input.reindex(columns=raw_features, fill_value=0)

        proba = model.predict_proba(df_input)[0]
        pred_class = model.predict(df_input)[0]

        risk_prob = proba
        fico_score = int(850 - (risk_prob * 550))
        rating = "EXCELLENT" if fico_score >= 750 else "GOOD" if fico_score >= 700 else "FAIR" if fico_score >= 650 else "POOR" if fico_score >= 550 else "VERY POOR"

        st.divider()
        col_score, col_risk = st.columns([1, 2])
        with col_score:
            st.metric("XGB FICO Score", fico_score)
        with col_risk:
            st.markdown(f"**Risk Grade: {rating}**")
            st.metric("Risk Probability", f"{risk_prob:.1%}", delta=f"{'Default' if pred_class == 1 else 'Safe'}")

        color = {"EXCELLENT": "Green", "GOOD": "Blue", "FAIR": "Yellow", "POOR": "Orange", "VERY POOR": "Red"}[rating]
        st.markdown(f"### {color} Risk Band: **{rating}**")

        st.subheader("Submitted Inputs")
        st.json(input_data)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("The schema-locked app passes delayed payment only for feature engineering. The raw source field is not part of model scoring.")

st.caption("Powered by kingametric_xgb")
