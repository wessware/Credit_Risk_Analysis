import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import xgboost as xgb
except Exception:
    xgb = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.kingametric_base_0 import KingaMetricXGB

st.set_page_config(
    page_title="Kinga Ensemble Credit Score",
    layout="wide"
)

st.title("Kinga ML Ensemble Predictor")
st.caption("Production Ensemble Model - kingametric_xgb_schema_locked.pkl")

st.divider()

GRADE_STYLES = {
    "EXCELLENT": {"color": "#1f7a4d", "label": "Very strong"},
    "GOOD": {"color": "#2b6cb0", "label": "Healthy"},
    "FAIR": {"color": "#b7791f", "label": "Mixed"},
    "POOR": {"color": "#c05621", "label": "Stretched"},
    "VERY POOR": {"color": "#c53030", "label": "High risk"},
}

FRIENDLY_LABELS = {
    "Annual_Income": "Annual income",
    "Monthly_Inhand_Salary": "Monthly take-home pay",
    "Num_Bank_Accounts": "Bank account count",
    "Num_Credit_Card": "Credit card count",
    "Interest_Rate": "Interest rate pressure",
    "Num_of_Loan": "Loan count",
    "Changed_Credit_Limit": "Recent credit limit changes",
    "Num_Credit_Inquiries": "Recent credit checks",
    "Outstanding_Debt": "Outstanding debt",
    "Credit_Utilization_Ratio": "Credit usage",
    "Credit_History_Age": "Credit history length",
    "Total_EMI_per_month": "Monthly EMI load",
    "Monthly_Balance": "Savings cushion",
    "Num_of_Delayed_Payment": "Delayed payments",
    "Payment_of_Min_Amount": "Minimum-payment behavior",
    "Credit_Mix": "Mix of credit accounts",
    "Borrower_Tier": "Borrower segment",
    "normalized_dti": "Debt compared with income",
    "normalized_emi": "Monthly payment pressure",
    "normalized_delinquency": "Repayment delay pressure",
    "normalized_credit_history": "History depth",
    "normalized_savings": "Cash buffer",
    "normalized_utilization": "Credit usage pressure",
    "Debt_Stress": "Combined debt and usage stress",
    "Repayment_Stress": "Combined EMI and delay stress",
    "Liquidity_Index": "Savings support",
    "Credit_Exposure": "Active revolving exposure",
    "Risk_Index": "Overall risk pressure",
    "Income_Delinq": "Income adjusted for delays",
    "Loan_DTI": "Loans combined with debt burden",
    "normalized_emi_sq": "Payment pressure intensity",
    "normalized_utilization_sq": "Usage pressure intensity",
    "normalized_dti_sq": "Debt pressure intensity",
    "normalized_delinquency_sq": "Delay pressure intensity",
    "normalized_emi_log": "Payment pressure signal",
    "normalized_utilization_log": "Usage pressure signal",
    "normalized_dti_log": "Debt pressure signal",
    "normalized_delinquency_log": "Delay pressure signal",
}

FEATURE_EXPLANATIONS = {
    "Annual_Income": "Income helps show how comfortably debt can be carried.",
    "Monthly_Inhand_Salary": "Take-home pay affects how much room is left after bills.",
    "Interest_Rate": "Higher rates usually make repayment harder over time.",
    "Num_of_Loan": "More active loans can stretch repayment capacity.",
    "Changed_Credit_Limit": "Recent limit changes can signal either added flexibility or stress.",
    "Num_Credit_Inquiries": "Frequent checks can look like recent credit-seeking behavior.",
    "Outstanding_Debt": "Higher debt tends to increase repayment pressure.",
    "Credit_Utilization_Ratio": "Using more of available credit often signals tighter borrowing headroom.",
    "Credit_History_Age": "A longer history gives the model more evidence of repayment behavior.",
    "Total_EMI_per_month": "Large monthly EMI commitments reduce breathing room.",
    "Monthly_Balance": "A stronger balance acts like a cushion against payment stress.",
    "Num_of_Delayed_Payment": "Past payment delays are an important warning sign.",
    "Payment_of_Min_Amount": "How minimum payments are handled hints at repayment discipline.",
    "Credit_Mix": "A healthier mix of accounts can look more stable.",
    "Borrower_Tier": "Borrower segment summarizes broad risk profile patterns learned in training.",
    "normalized_dti": "This shows how large debt is relative to annual income.",
    "normalized_emi": "This shows how much monthly income is already committed to EMI payments.",
    "normalized_delinquency": "This captures how often payments are delayed relative to the number of loans.",
    "normalized_credit_history": "This measures how established the credit record is.",
    "normalized_savings": "This measures how much cash buffer is left after income is received.",
    "normalized_utilization": "This measures how much of available credit is already being used.",
    "Debt_Stress": "This combines debt burden and credit usage into one pressure signal.",
    "Repayment_Stress": "This combines EMI pressure and payment delays.",
    "Liquidity_Index": "This blends cash buffer with monthly payment pressure.",
    "Credit_Exposure": "This tracks how much revolving credit is actively exposed.",
    "Risk_Index": "This is a blended pressure signal across debt, usage, and delays.",
    "Income_Delinq": "This helps the model read delays in the context of income level.",
    "Loan_DTI": "This helps the model weigh debt burden together with loan count.",
}


def format_value(feature_name, raw_inputs, model_frame):
    if feature_name in raw_inputs:
        value = raw_inputs[feature_name]
    else:
        value = model_frame.iloc[0].get(feature_name, 0.0)

    if isinstance(value, str):
        return value.replace("_", " ")
    if feature_name in {"Interest_Rate", "Credit_Utilization_Ratio"} or feature_name.startswith("normalized_"):
        return f"{float(value):.0%}"
    if "Income" in feature_name or "Salary" in feature_name or "Debt" in feature_name or "Balance" in feature_name or "EMI" in feature_name:
        return f"{float(value):,.0f}"
    return f"{float(value):.2f}" if isinstance(value, float) else str(value)


def describe_risk_band(risk_prob, rating):
    tone = {
        "EXCELLENT": "This looks like a very strong profile with relatively few warning signals.",
        "GOOD": "This profile looks healthy overall, with a few areas worth keeping steady.",
        "FAIR": "This profile is mixed: some signals support the score, while others are pulling it down.",
        "POOR": "Several signals are putting pressure on the score and raising the risk estimate.",
        "VERY POOR": "The model is seeing multiple strong risk signals at the same time.",
    }
    return f"{tone[rating]} Estimated default likelihood is {risk_prob:.1%}."


def build_health_dimensions(raw_inputs):
    salary = raw_inputs["Monthly_Inhand_Salary"]
    annual_income = raw_inputs["Annual_Income"]
    loans = raw_inputs["Num_of_Loan"]

    normalized_dti = np.clip(raw_inputs["Outstanding_Debt"] / (annual_income + 1), 0, 1)
    normalized_emi = np.clip(raw_inputs["Total_EMI_per_month"] / (salary + 1), 0, 1)
    normalized_delinquency = np.clip(raw_inputs["Num_of_Delayed_Payment"] / (loans + 1), 0, 1)
    normalized_credit_history = np.clip(raw_inputs["Credit_History_Age"] / 840, 0, 1)
    normalized_savings = np.clip(raw_inputs["Monthly_Balance"] / (salary + 1), 0, 1)
    normalized_utilization = np.clip(raw_inputs["Credit_Utilization_Ratio"], 0, 1)

    dimension_rows = [
        ("Debt load vs income", 1 - normalized_dti, format_value("normalized_dti", raw_inputs, pd.DataFrame([raw_inputs]))),
        ("Monthly payment breathing room", 1 - normalized_emi, format_value("normalized_emi", raw_inputs, pd.DataFrame([raw_inputs]))),
        ("Payment consistency", 1 - normalized_delinquency, format_value("normalized_delinquency", raw_inputs, pd.DataFrame([raw_inputs]))),
        ("Credit history depth", normalized_credit_history, format_value("normalized_credit_history", raw_inputs, pd.DataFrame([raw_inputs]))),
        ("Cash buffer", normalized_savings, format_value("normalized_savings", raw_inputs, pd.DataFrame([raw_inputs]))),
        ("Credit usage discipline", 1 - normalized_utilization, format_value("normalized_utilization", raw_inputs, pd.DataFrame([raw_inputs]))),
    ]

    return pd.DataFrame(dimension_rows, columns=["dimension", "score", "observed"])


def compute_local_feature_contributions(model, aligned_frame):
    if xgb is None or getattr(model, "xgb_model", None) is None:
        return None

    try:
        contributions = model.xgb_model.get_booster().predict(
            xgb.DMatrix(aligned_frame),
            pred_contribs=True
        )[0]
        feature_values = contributions[:-1]
        contribution_df = pd.DataFrame(
            {
                "feature": aligned_frame.columns,
                "contribution": feature_values,
            }
        )
        contribution_df["abs_contribution"] = contribution_df["contribution"].abs()
        return contribution_df.sort_values("abs_contribution", ascending=False)
    except Exception:
        return None


def build_xgb_explanation(model, raw_inputs):
    df_input = pd.DataFrame([raw_inputs])
    if getattr(model, "raw_input_features", []):
        df_input = df_input.reindex(columns=model.raw_input_features, fill_value=0)

    model_frame = model.build_model_frame(df_input, fit_encoder=False)
    aligned_frame = model_frame.reindex(columns=model.feature_names, fill_value=0.0)

    importance_df = pd.DataFrame(
        {
            "feature": model.feature_names,
            "importance": getattr(model.xgb_model, "feature_importances_", np.zeros(len(model.feature_names))),
        }
    ).sort_values("importance", ascending=False)
    importance_df["label"] = importance_df["feature"].map(lambda name: FRIENDLY_LABELS.get(name, name.replace("_", " ")))

    local_df = compute_local_feature_contributions(model, aligned_frame)
    if local_df is not None:
        local_df["label"] = local_df["feature"].map(lambda name: FRIENDLY_LABELS.get(name, name.replace("_", " ")))
        local_df["observed"] = local_df["feature"].map(lambda name: format_value(name, raw_inputs, model_frame))
        local_df["why_it_matters"] = local_df["feature"].map(
            lambda name: FEATURE_EXPLANATIONS.get(name, "The model has learned that this signal changes risk.")
        )
    else:
        fallback_features = importance_df.head(10)["feature"]
        local_df = pd.DataFrame({"feature": fallback_features})
        local_df["contribution"] = 0.0
        local_df["abs_contribution"] = importance_df.set_index("feature").loc[fallback_features, "importance"].to_numpy()
        local_df["label"] = local_df["feature"].map(lambda name: FRIENDLY_LABELS.get(name, name.replace("_", " ")))
        local_df["observed"] = local_df["feature"].map(lambda name: format_value(name, raw_inputs, model_frame))
        local_df["why_it_matters"] = local_df["feature"].map(
            lambda name: FEATURE_EXPLANATIONS.get(name, "This signal has strong predictive power in the trained model.")
        )

    health_df = build_health_dimensions(raw_inputs)
    return local_df, importance_df, health_df


def plot_local_push_chart(local_df):
    plot_df = local_df.head(8).sort_values("contribution")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#d1495b" if value > 0 else "#2f855a" for value in plot_df["contribution"]]
    ax.barh(plot_df["label"], plot_df["contribution"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_title("What pushed this prediction higher or lower")
    ax.set_xlabel("Positive values increase default risk; negative values reduce it")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        offset = 0.01 if row["contribution"] >= 0 else -0.01
        align = "left" if row["contribution"] >= 0 else "right"
        ax.text(row["contribution"] + offset, idx, row["observed"], va="center", ha=align, fontsize=9)

    plt.tight_layout()
    return fig


def plot_health_profile_chart(health_df):
    plot_df = health_df.sort_values("score")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#d1495b" if score < 0.4 else "#ed8936" if score < 0.7 else "#2f855a" for score in plot_df["score"]]
    ax.barh(plot_df["dimension"], plot_df["score"] * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_title("Profile health by major credit dimensions")
    ax.set_xlabel("Stronger position")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(min((row["score"] * 100) + 1.5, 99), idx, row["observed"], va="center", ha="left", fontsize=9)

    plt.tight_layout()
    return fig


def build_driver_table(local_df):
    table_df = local_df.head(8).copy()
    table_df["effect"] = table_df["contribution"].apply(
        lambda value: "Raised risk" if value > 0 else "Lowered risk" if value < 0 else "High model attention"
    )
    table_df["reason"] = table_df.apply(
        lambda row: f"{row['why_it_matters']} Observed value: {row['observed']}.",
        axis=1
    )
    return table_df.loc[:, ["label", "effect", "observed", "reason"]].rename(
        columns={
            "label": "Factor",
            "effect": "Effect on this prediction",
            "observed": "What the app saw",
            "reason": "Why it matters",
        }
    )


def render_summary_lists(local_df):
    risk_drivers = local_df[local_df["contribution"] > 0].head(3)
    strengths = local_df[local_df["contribution"] < 0].sort_values("contribution").head(3)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Main pressures on the score")
        if risk_drivers.empty:
            st.write("No single feature stood out as a strong upward risk driver in this prediction.")
        else:
            for _, row in risk_drivers.iterrows():
                st.write(f"- **{row['label']}**: {row['why_it_matters']} Current signal: `{row['observed']}`.")

    with col_b:
        st.markdown("#### Main strengths supporting the score")
        if strengths.empty:
            st.write("The prediction is mostly driven by upward-risk features, with limited offsetting strengths.")
        else:
            for _, row in strengths.iterrows():
                st.write(f"- **{row['label']}**: {row['why_it_matters']} Current signal: `{row['observed']}`.")


@st.cache_resource
def load_model():
    model_path = "pickled_models/kingametric_base_1.pkl"
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
        st.caption("The score explanation below uses these inputs plus engineered model signals.")

    submitted = st.form_submit_button("Predict with XGB Pipeline", width='stretch')

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

        risk_prob = float(model.predict_proba(df_input)[0])
        pred_class = int(model.predict(df_input)[0])

        fico_score = int(850 - (risk_prob * 550))
        rating = "EXCELLENT" if fico_score >= 750 else "GOOD" if fico_score >= 700 else "FAIR" if fico_score >= 650 else "POOR" if fico_score >= 550 else "VERY POOR"
        grade_style = GRADE_STYLES[rating]

        local_df, importance_df, health_df = build_xgb_explanation(model, input_data)

        st.divider()
        score_col, prob_col, band_col = st.columns(3)
        with score_col:
            st.metric("XGB FICO Score", fico_score)
        with prob_col:
            st.metric("Default likelihood", f"{risk_prob:.1%}", delta="Higher" if pred_class == 1 else "Lower")
        with band_col:
            st.metric("Risk grade", rating, delta=grade_style["label"])

        st.markdown(
            f"""
            <div style="padding: 1rem 1.2rem; border-radius: 0.9rem; background: {grade_style['color']}18; border: 1px solid {grade_style['color']}55;">
                <div style="font-size: 1.1rem; font-weight: 700; color: {grade_style['color']};">{rating} profile</div>
                <div style="margin-top: 0.35rem;">{describe_risk_band(risk_prob, rating)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_summary_lists(local_df)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Visual 1: Top feature pushes")
            st.pyplot(plot_local_push_chart(local_df), clear_figure=True, width='stretch')
        with chart_col2:
            st.markdown("#### Visual 2: Credit health profile")
            st.pyplot(plot_health_profile_chart(health_df), clear_figure=True, width='stretch')

        table_col1, table_col2 = st.columns([1.6, 1])
        with table_col1:
            st.markdown("#### Factor-by-factor explanation")
            st.dataframe(build_driver_table(local_df), width='stretch', hide_index=True)
        with table_col2:
            st.markdown("#### Highest-power model signals")
            top_importance = importance_df.head(8).copy()
            top_importance["importance"] = top_importance["importance"].round(4)
            st.dataframe(
                top_importance.loc[:, ["label", "importance"]].rename(
                    columns={"label": "Model signal", "importance": "Training importance"}
                ),
                width='stretch',
                hide_index=True,
            )
            st.caption("These are the strongest overall signals the trained XGB model tends to rely on.")

        with st.expander("Submitted inputs and raw values", expanded=False):
            st.json(input_data)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("The schema-locked app passes delayed payment only for feature engineering. The raw source field is not part of model scoring.")

st.caption("Powered by kingametric_xgb")
