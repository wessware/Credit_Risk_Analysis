import streamlit as st
import mysql.connector
import hashlib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Credit Risk Scoring",
    layout="centered"
)

st.title("FICO-Style Credit Risk Engine - FIXED")
st.caption("SQL-Based Composite Credit Scoring")

st.divider()

# -------------------------
# DB CONFIG - TOP LEVEL (Widgets always run)
# -------------------------
if 'db_creds' not in st.session_state:
    st.session_state.db_creds = None
    st.session_state.db_hash = None

# Secrets priority
if st.secrets and 'host' in st.secrets:
    creds_str = str(st.secrets)
    if st.session_state.db_hash != hashlib.md5(creds_str.encode()).hexdigest():
        try:
            test_conn = mysql.connector.connect(**st.secrets)
            test_conn.close()
            st.session_state.db_creds = st.secrets
            st.session_state.db_hash = hashlib.md5(creds_str.encode()).hexdigest()
            st.success("✅ Secrets validated")
        except Exception as e:
            st.error(f"Secrets invalid: {e}")
            st.stop()

# Sidebar (always executes)
with st.sidebar:
    st.header("🛠️ Database Setup")
    host = st.text_input("Host:", value="localhost")
    user = st.text_input("User:", value="root")
    password = st.text_input("Password:", type="password", value="")
    database = st.text_input("Database:", value="kingametrics")
    
    if st.button("Test Connection", key="sidebar_test"):
        creds_str = f"{host}{user}{password}{database}"
        if st.session_state.db_hash != hashlib.md5(creds_str.encode()).hexdigest():
            try:
                test_conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
                test_conn.close()
                st.session_state.db_creds = dict(host=host, user=user, password=password, database=database)
                st.session_state.db_hash = hashlib.md5(creds_str.encode()).hexdigest()
                st.success("✅ Connected & Cached!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

if not st.session_state.db_creds:
    st.error("👆 Configure database in sidebar or secrets.toml")
    st.stop()

@st.cache_resource(ttl=600)
def get_connection(_hash):
    return mysql.connector.connect(**st.session_state.db_creds)

# ------------------------- 
# SQL EXECUTION
# -------------------------
def run_scoring(input_data):
    conn = None
    try:
        conn = get_connection(st.session_state.db_hash)
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
        if conn:
            conn.close()

# -------------------------
# FORM
# -------------------------
with st.form("credit_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income & Debt")
        Annual_Income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
        Monthly_Inhand_Salary = st.number_input("Monthly Salary", min_value=0.0, value=4000.0)
        Outstanding_Debt = st.number_input("Debt", min_value=0.0, value=10000.0)
    with col2:
        st.subheader("Loans")
        Total_EMI_per_month = st.number_input("EMI/Month", min_value=0.0, value=500.0)
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

    submitted = st.form_submit_button("🚀 Score Me!", use_container_width=True)

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
        st.metric("FICO Score", score, delta=None)
        st.markdown(f"**Grade: {rating}**")
        
        color = {"EXCELLENT": "🟢", "GOOD": "🔵", "FAIR": "🟡", "POOR": "🟠", "VERY POOR": "🔴"}[rating]
        st.markdown(f"{color} **{rating} Profile**")
    else:
        st.error("No score - check logs")
