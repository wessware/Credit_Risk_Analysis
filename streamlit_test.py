import streamlit as st
import pandas as pd
import mysql.connector

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Credit Risk Scoring",
    layout="centered"
)

st.title("FICO-Style Credit Risk Engine")
st.caption("SQL-Based Composite Credit Scoring")

st.divider()

# -------------------------
# MYSQL CONNECTION
# -------------------------
@st.cache_resource(ttl=600)
def get_connection():
    try:
        # Try st.secrets first
        if 'MYSQL_HOST' in st.secrets:
            return mysql.connector.connect(**st.secrets)
        
        # Fallback to sidebar inputs
        host = st.sidebar.text_input("MySQL Host", value="localhost", type="password")
        user = st.sidebar.text_input("MySQL User", value="root")
        password = st.sidebar.text_input("MySQL Password", value="", type="password")
        database = st.sidebar.text_input("Database", value="credit_db")
        
        if st.sidebar.button("Test Connection"):
            test_conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
            test_conn.close()
            st.sidebar.success("✅ Connection OK!")
            return mysql.connector.connect(host=host, user=user, password=password, database=database)
        else:
            raise mysql.connector.Error("Configure credentials in sidebar and test connection")
            
    except mysql.connector.Error as e:
        st.error(f"❌ DB Connection failed: {str(e)}")
        st.info("💡 Use st.secrets.toml or sidebar creds. Install: pip install mysql-connector-python")
        raise

# -------------------------
# SQL EXECUTION FUNCTION
# -------------------------
def run_scoring(input_data):
    conn = None
    try:
        conn = get_connection()
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
                (1 - normalized_utilization) * 0.10
                AS Composite_Credit_Risk_Score

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

    except mysql.connector.Error as e:
        st.error(f"❌ Query failed: {str(e)}")
        st.info("💡 Check DB schema matches inputs. SQL columns must exist.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()

# -------------------------
# FORM (MATCHES SQL INPUTS)
# -------------------------
with st.form("credit_form"):

    st.subheader("Income & Debt")

    Annual_Income = st.number_input("Annual Income", 0.0, 1_000_000_000.0, 50000.0)
    Monthly_Inhand_Salary = st.number_input("Monthly In-hand Salary", 0.0, 1_000_000.0, 4000.0)
    Outstanding_Debt = st.number_input("Outstanding Debt", 0.0, 1_000_000_000.0, 10000.0)

    st.subheader("Loan & Repayment")

    Total_EMI_per_month = st.number_input("Total EMI per Month", 0.0, 1_000_000.0, 500.0)
    Num_of_Loan = st.number_input("Number of Loans", 1, 50, 2)  
    Num_of_Delayed_Payment = st.number_input("Delayed Payments", 0, 100, 1)

    st.subheader("Credit Profile")

    Credit_History_Age = st.number_input("Credit History (months)", 0, 600, 60)
    Credit_Utilization_Ratio = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3)

    st.subheader("Savings Behavior")

    Monthly_Balance = st.number_input("Monthly Balance", 0.0, 1_000_000.0, 1000.0)

    submitted = st.form_submit_button("Compute Credit Score")

# -------------------------
# EXECUTION
# -------------------------
if submitted:

    input_data = {
        "Outstanding_Debt": Outstanding_Debt,
        "Annual_Income": Annual_Income,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
        "Num_of_Loan": Num_of_Loan,
        "Credit_History_Age": Credit_History_Age,
        "Monthly_Balance": Monthly_Balance,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio
    }

    result = run_scoring(input_data)

    st.divider()
    st.subheader("Credit Score Result")

    if result and "Credit_Score" in result:
        score = result["Credit_Score"]
        rating = result["Credit_Score_Rating"]

        st.metric("Credit Score", f"{int(score)}")
        st.write(f"**Rating:** {rating}")

        if rating == "EXCELLENT":
            st.success("🟢 Excellent Credit Profile")
        elif rating == "GOOD":
            st.info("🔵 Good Credit Profile")
        elif rating == "FAIR":
            st.warning("🟠 Fair Credit Profile")
        else:
            st.error("🔴 High Risk Profile")
    else:
        st.warning("⚠️ No result. Check errors above.")
            