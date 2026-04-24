import hashlib

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Credit Risk Scoring",
    layout="wide"
)

st.title("FICO Credit Risk Engine")
st.caption("SQL-Based Composite Credit Scoring")

st.divider()

GRADE_STYLES = {
    "EXCELLENT": {"color": "#1f7a4d", "label": "Very strong"},
    "GOOD": {"color": "#2b6cb0", "label": "Healthy"},
    "FAIR": {"color": "#b7791f", "label": "Mixed"},
    "POOR": {"color": "#c05621", "label": "Stretched"},
    "VERY POOR": {"color": "#c53030", "label": "High risk"},
}

COMPONENT_SPECS = [
    {
        "label": "Debt load vs income",
        "metric_key": "normalized_dti",
        "weight": 0.25,
        "health_fn": lambda result: 1 - result["normalized_dti"],
        "observed_fn": lambda result: f"{result['normalized_dti']:.0%}",
        "why": "A lower debt-to-income burden usually leaves more room to absorb repayments.",
    },
    {
        "label": "Monthly payment breathing room",
        "metric_key": "normalized_emi",
        "weight": 0.20,
        "health_fn": lambda result: 1 - result["normalized_emi"],
        "observed_fn": lambda result: f"{result['normalized_emi']:.0%}",
        "why": "A smaller EMI share means less monthly cash strain.",
    },
    {
        "label": "Payment consistency",
        "metric_key": "normalized_delinquency",
        "weight": 0.20,
        "health_fn": lambda result: 1 - result["normalized_delinquency"],
        "observed_fn": lambda result: f"{result['normalized_delinquency']:.0%}",
        "why": "Fewer payment delays are a strong signal of reliable repayment behavior.",
    },
    {
        "label": "Credit history depth",
        "metric_key": "normalized_credit_history",
        "weight": 0.15,
        "health_fn": lambda result: result["normalized_credit_history"],
        "observed_fn": lambda result: f"{result['Credit_History_Age']:.0f} months",
        "why": "A longer credit history gives more evidence that repayment behavior is established.",
    },
    {
        "label": "Cash buffer",
        "metric_key": "normalized_savings",
        "weight": 0.10,
        "health_fn": lambda result: result["normalized_savings"],
        "observed_fn": lambda result: f"{result['normalized_savings']:.0%}",
        "why": "A stronger monthly balance provides cushion if expenses rise or income is interrupted.",
    },
    {
        "label": "Credit usage discipline",
        "metric_key": "normalized_utilization",
        "weight": 0.10,
        "health_fn": lambda result: 1 - result["normalized_utilization"],
        "observed_fn": lambda result: f"{result['normalized_utilization']:.0%}",
        "why": "Lower utilization usually means more unused credit headroom and less revolving pressure.",
    },
]


def build_sql_component_table(result):
    rows = []
    for spec in COMPONENT_SPECS:
        health_score = float(np.clip(spec["health_fn"](result), 0, 1))
        max_points = spec["weight"] * 550
        earned_points = health_score * max_points
        points_left = max_points - earned_points
        rows.append(
            {
                "Factor": spec["label"],
                "Weight": spec["weight"],
                "Health score": health_score,
                "What the app saw": spec["observed_fn"](result),
                "Score points earned": earned_points,
                "Score points left on the table": points_left,
                "Why it matters": spec["why"],
            }
        )
    return pd.DataFrame(rows).sort_values("Score points left on the table", ascending=False)


def render_sql_summary(component_df):
    drags = component_df.sort_values("Score points left on the table", ascending=False).head(3)
    strengths = component_df.sort_values("Score points earned", ascending=False).head(3)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Main pressures on the score")
        for _, row in drags.iterrows():
            st.write(
                f"- **{row['Factor']}**: {row['Why it matters']} Current signal: `{row['What the app saw']}`. "
                f"Estimated points missed: `{row['Score points left on the table']:.1f}`."
            )

    with col2:
        st.markdown("#### Main strengths supporting the score")
        for _, row in strengths.iterrows():
            st.write(
                f"- **{row['Factor']}**: {row['Why it matters']} Current signal: `{row['What the app saw']}`. "
                f"Points contributing to score: `{row['Score points earned']:.1f}`."
            )


def plot_sql_score_breakdown(component_df):
    plot_df = component_df.sort_values("Score points earned")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(plot_df["Factor"], plot_df["Score points earned"], color="#2f855a", label="Points earned")
    ax.barh(
        plot_df["Factor"],
        plot_df["Score points left on the table"],
        left=plot_df["Score points earned"],
        color="#e2e8f0",
        label="Available but not earned"
    )
    ax.set_title("How each score component built the final FICO result")
    ax.set_xlabel("FICO points within each component")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["Score points earned"] + 1, idx, row["What the app saw"], va="center", ha="left", fontsize=9)

    plt.tight_layout()
    return fig


def plot_sql_health_profile(component_df):
    plot_df = component_df.sort_values("Health score")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#d1495b" if score < 0.4 else "#ed8936" if score < 0.7 else "#2f855a" for score in plot_df["Health score"]]
    ax.barh(plot_df["Factor"], plot_df["Health score"] * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_title("Health of each major credit dimension")
    ax.set_xlabel("Stronger position")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(min((row["Health score"] * 100) + 1.5, 99), idx, row["What the app saw"], va="center", ha="left", fontsize=9)

    plt.tight_layout()
    return fig


def describe_sql_profile(score, rating, composite_score):
    tone = {
        "EXCELLENT": "The rule-based engine sees a strong profile with solid repayment capacity and low visible stress.",
        "GOOD": "The profile is healthy overall, though a few areas could still improve the score.",
        "FAIR": "The profile is balanced between supportive signals and visible repayment pressure.",
        "POOR": "Several weighted components are weak enough to materially drag the score down.",
        "VERY POOR": "The rule-based engine sees multiple high-pressure signals across the major score drivers.",
    }
    return f"{tone[rating]} Composite score strength is {composite_score:.1%}, which maps to a FICO score of {score}."


# -------------------------
# DB CONFIG - TOP LEVEL
# -------------------------
if "db_creds" not in st.session_state:
    st.session_state.db_creds = None
    st.session_state.db_hash = None

# Secrets priority
if st.secrets and "host" in st.secrets:
    creds_str = str(st.secrets)
    if st.session_state.db_hash != hashlib.md5(creds_str.encode()).hexdigest():
        try:
            test_conn = mysql.connector.connect(**st.secrets)
            test_conn.close()
            st.session_state.db_creds = st.secrets
            st.session_state.db_hash = hashlib.md5(creds_str.encode()).hexdigest()
            st.success("Secrets validated")
        except Exception as e:
            st.error(f"Secrets invalid: {e}")
            st.stop()

# Sidebar (always executes)
with st.sidebar:
    st.header("Connect to Database")
    host = st.text_input("Host:", value="localhost")
    user = st.text_input("User:", value="root")
    password = st.text_input("Password:", type="password", value="WECHALE$0398_wess")
    database = st.text_input("Database:", value="kingametrics")

    if st.button("Test Connection", key="sidebar_test"):
        creds_str = f"{host}{user}{password}{database}"
        if st.session_state.db_hash != hashlib.md5(creds_str.encode()).hexdigest():
            try:
                test_conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
                test_conn.close()
                st.session_state.db_creds = dict(host=host, user=user, password=password, database=database)
                st.session_state.db_hash = hashlib.md5(creds_str.encode()).hexdigest()
                st.success("Connected to Kingametric Database")
            except Exception as e:
                st.error(f"Connection failed: {e}")

if not st.session_state.db_creds:
    st.error("To proceed please connect to the Kingametric Database")
    st.stop()


@st.cache_resource(ttl=6000)
def get_connection(_hash):
    return mysql.connector.connect(**st.session_state.db_creds)


# -------------------------
# SQL EXECUTION
# -------------------------
def run_scoring(input_data):
    conn = None
    try:
        conn = get_connection(st.session_state.db_hash)
        if not conn.is_connected():
            conn.reconnect(attempts=2, delay=1)

        conn.ping(reconnect=True)
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT *,
            300 + (Composite_Credit_Risk_Score * 550) AS Credit_Score,
            CASE
                WHEN (300 + Composite_Credit_Risk_Score * 550) >= 750 THEN 'EXCELLENT'
                WHEN (300 + Composite_Credit_Risk_Score * 550) >= 700 THEN 'GOOD'
                WHEN (300 + Composite_Credit_Risk_Score * 550) >= 650 THEN 'FAIR'
                WHEN (300 + Composite_Credit_Risk_Score * 550) >= 550 THEN 'POOR'
                ELSE 'VERY POOR'
            END AS Credit_Score_Rating
        FROM (
            SELECT *,
                (1 - normalized_dti) * 0.25 +
                (1 - normalized_emi) * 0.20 +
                (1 - normalized_delinquency) * 0.20 +
                normalized_credit_history * 0.15 +
                normalized_savings * 0.10 +
                (1 - normalized_utilization) * 0.10 AS Composite_Credit_Risk_Score
            FROM (
                SELECT
                    %(Outstanding_Debt)s AS Outstanding_Debt,
                    %(Annual_Income)s AS Annual_Income,
                    %(Total_EMI_per_month)s AS Total_EMI_per_month,
                    %(Monthly_Inhand_Salary)s AS Monthly_Inhand_Salary,
                    %(Num_of_Delayed_Payment)s AS Num_of_Delayed_Payment,
                    %(Num_of_Loan)s AS Num_of_Loan,
                    %(Credit_History_Age)s AS Credit_History_Age,
                    %(Monthly_Balance)s AS Monthly_Balance,
                    %(Credit_Utilization_Ratio)s AS Credit_Utilization_Ratio,
                    GREATEST(0, LEAST(%(Outstanding_Debt)s / (%(Annual_Income)s + 1), 1)) AS normalized_dti,
                    GREATEST(0, LEAST(%(Total_EMI_per_month)s / (%(Monthly_Inhand_Salary)s + 1), 1)) AS normalized_emi,
                    GREATEST(0, LEAST(%(Num_of_Delayed_Payment)s / (%(Num_of_Loan)s + 1), 1)) AS normalized_delinquency,
                    GREATEST(0, LEAST(%(Credit_History_Age)s / 120, 1)) AS normalized_credit_history,
                    GREATEST(0, LEAST(%(Monthly_Balance)s / (%(Monthly_Inhand_Salary)s + 1), 1)) AS normalized_savings,
                    GREATEST(0, LEAST(%(Credit_Utilization_Ratio)s, 1)) AS normalized_utilization
            ) normalized
        ) scored;
        """

        cursor.execute(query, input_data)
        result = cursor.fetchone()
        return result
    except Exception as e:
        st.error(f"Query error: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()


# -------------------------
# FORM
# -------------------------
with st.form("credit_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income & Debt")
        Annual_Income = st.number_input("Annual Income", min_value=1.0, value=50000.0)
        Monthly_Inhand_Salary = st.number_input("Monthly Salary", min_value=1.0, value=4000.0)
        Outstanding_Debt = st.number_input("Debt", min_value=1.0, value=10000.0)
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
        st.subheader("Savings")
        Monthly_Balance = st.number_input("Balance", min_value=0.0, value=1000.0)

    submitted = st.form_submit_button("Evaluate Risk Score", use_container_width=True)

if submitted:
    input_data = dict(
        Outstanding_Debt=Outstanding_Debt, Annual_Income=Annual_Income,
        Total_EMI_per_month=Total_EMI_per_month, Monthly_Inhand_Salary=Monthly_Inhand_Salary,
        Num_of_Delayed_Payment=Num_of_Delayed_Payment, Num_of_Loan=Num_of_Loan,
        Credit_History_Age=Credit_History_Age, Monthly_Balance=Monthly_Balance,
        Credit_Utilization_Ratio=Credit_Utilization_Ratio
    )

    result = run_scoring(input_data)

    st.divider()
    if result:
        score = int(result["Credit_Score"])
        rating = result["Credit_Score_Rating"]
        composite_score = float(result["Composite_Credit_Risk_Score"])
        grade_style = GRADE_STYLES[rating]
        component_df = build_sql_component_table(result)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("FICO Score", score)
        with metric_col2:
            st.metric("Composite score strength", f"{composite_score:.1%}")
        with metric_col3:
            st.metric("Grade", rating, delta=grade_style["label"])

        st.markdown(
            f"""
            <div style="padding: 1rem 1.2rem; border-radius: 0.9rem; background: {grade_style['color']}18; border: 1px solid {grade_style['color']}55;">
                <div style="font-size: 1.1rem; font-weight: 700; color: {grade_style['color']};">{rating} profile</div>
                <div style="margin-top: 0.35rem;">{describe_sql_profile(score, rating, composite_score)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_sql_summary(component_df)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Visual 1: Score build-up by weighted component")
            st.pyplot(plot_sql_score_breakdown(component_df), clear_figure=True, use_container_width=True)
        with chart_col2:
            st.markdown("#### Visual 2: Credit health profile")
            st.pyplot(plot_sql_health_profile(component_df), clear_figure=True, use_container_width=True)

        detail_df = component_df.copy()
        detail_df["Weight"] = detail_df["Weight"].map(lambda value: f"{value:.0%}")
        detail_df["Health score"] = detail_df["Health score"].map(lambda value: f"{value:.0%}")
        detail_df["Score points earned"] = detail_df["Score points earned"].map(lambda value: f"{value:.1f}")
        detail_df["Score points left on the table"] = detail_df["Score points left on the table"].map(lambda value: f"{value:.1f}")

        st.markdown("#### Factor-by-factor explanation")
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        with st.expander("Submitted inputs and SQL raw outputs", expanded=False):
            st.json({"inputs": input_data, "sql_result": result})
    else:
        st.error("No score - check logs")
