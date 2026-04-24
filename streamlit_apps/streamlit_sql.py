import hashlib
import os
import sys

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_scripts.kenya_counties import kenya_counties

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

DEMOGRAPHIC_IMPACT_CAP = 0.20

COUNTY_POPULATION_PROXIES = {
    "Mombasa": 1320000,
    "Kwale": 640000,
    "Kilifi": 1550000,
    "Lamu": 175000,
    "Tana River": 345000,
    "Taita-Taveta": 390000,
    "Garissa": 980000,
    "Wajir": 910000,
    "Mandera": 1120000,
    "Marsabit": 510000,
    "Isiolo": 320000,
    "Meru": 1650000,
    "Tharaka-Nithi": 470000,
    "Embu": 680000,
    "Kitui": 1240000,
    "Machakos": 1430000,
    "Makueni": 1010000,
    "Nyandarua": 710000,
    "Nyeri": 810000,
    "Kirinyaga": 640000,
    "Murang'a": 1110000,
    "Kiambu": 2620000,
    "Turkana": 980000,
    "West Pokot": 720000,
    "Samburu": 350000,
    "Trans-Nzoia": 1110000,
    "Uasin Gishu": 1280000,
    "Elgeyo-Marakwet": 450000,
    "Nandi": 1010000,
    "Baringo": 790000,
    "Laikipia": 590000,
    "Nakuru": 2380000,
    "Narok": 1230000,
    "Kajiado": 1210000,
    "Kericho": 930000,
    "Bomet": 920000,
    "Kakamega": 2040000,
    "Vihiga": 650000,
    "Bungoma": 1760000,
    "Busia": 960000,
    "Siaya": 1000000,
    "Kisumu": 1290000,
    "Homa Bay": 1180000,
    "Migori": 1210000,
    "Kisii": 1320000,
    "Nyamira": 680000,
    "Nairobi": 4560000,
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

DEMOGRAPHIC_SPECS = [
    {
        "key": "gender",
        "label": "Gender",
        "weight": 0.05,
        "score_fn": lambda data, _: 0.55 if data["gender"] == "Male" else 0.45,
        "observed_fn": lambda data, _: data["gender"],
        "why": "This placeholder demographic policy slightly favors male applicants, as requested.",
    },
    {
        "key": "marital_status",
        "label": "Marital status",
        "weight": 0.10,
        "score_fn": lambda data, _: {"Single": 0.45, "Married": 1.00, "Divorced": 0.30}[data["marital_status"]],
        "observed_fn": lambda data, _: data["marital_status"],
        "why": "Married applicants are given the strongest stability score in this demographic policy.",
    },
    {
        "key": "dependents",
        "label": "Number of dependents",
        "weight": 0.15,
        "score_fn": lambda data, _: float(np.clip(data["dependents"] / 6, 0, 1)),
        "observed_fn": lambda data, _: str(data["dependents"]),
        "why": "More dependents receive a higher score under the configured weighting rule.",
    },
    {
        "key": "education_level",
        "label": "Education level",
        "weight": 0.10,
        "score_fn": lambda data, _: {"Primary": 0.25, "Secondary": 0.50, "College": 0.75, "University": 1.00}[data["education_level"]],
        "observed_fn": lambda data, _: data["education_level"],
        "why": "Higher education levels are assumed to correlate with better long-term earning resilience.",
    },
    {
        "key": "employment_status",
        "label": "Employment status",
        "weight": 0.15,
        "score_fn": lambda data, _: {
            "Unemployed": 0.15,
            "Freelancer": 0.45,
            "Contractor": 0.65,
            "Employed": 0.85,
            "Employed-PNP": 1.00,
        }[data["employment_status"]],
        "observed_fn": lambda data, _: data["employment_status"],
        "why": "Employment types that suggest steadier income get higher demographic support.",
    },
    {
        "key": "age",
        "label": "Age",
        "weight": 0.05,
        "score_fn": lambda data, _: float(np.clip(np.exp(-((data["age"] - 42) / 18) ** 2), 0, 1)),
        "observed_fn": lambda data, _: f"{data['age']} years",
        "why": "Middle ages are favored, with the strongest support around the most established working years.",
    },
    {
        "key": "county",
        "label": "County of residence",
        "weight": 0.40,
        "score_fn": lambda data, county_ref: float(county_ref.loc[data["county"], "density_index"]),
        "observed_fn": lambda data, county_ref: (
            f"{data['county']} ({county_ref.loc[data['county'], 'density_index']:.0%} density index)"
        ),
        "why": "The placeholder policy favors counties with higher population-density proxy scores because of assumed stronger local liquidity.",
    },
]


def get_rating_from_score(score):
    if score >= 750:
        return "EXCELLENT"
    if score >= 700:
        return "GOOD"
    if score >= 650:
        return "FAIR"
    if score >= 550:
        return "POOR"
    return "VERY POOR"


@st.cache_data
def build_county_density_reference():
    county_df = pd.DataFrame({"County": kenya_counties})
    county_df["assumed_population"] = county_df["County"].map(COUNTY_POPULATION_PROXIES).fillna(500000)
    min_population = county_df["assumed_population"].min()
    max_population = county_df["assumed_population"].max()
    county_df["density_index"] = (
        (county_df["assumed_population"] - min_population) / (max_population - min_population)
    ).clip(0, 1)
    return county_df.set_index("County")


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


def build_demographic_component_table(demo_inputs, base_score, county_reference):
    rows = []
    for spec in DEMOGRAPHIC_SPECS:
        feature_score = float(np.clip(spec["score_fn"](demo_inputs, county_reference), 0, 1))
        weighted_contribution = feature_score * spec["weight"]
        neutral_contribution = 0.5 * spec["weight"]
        impact_ratio = ((feature_score - 0.5) / 0.5) * spec["weight"] * DEMOGRAPHIC_IMPACT_CAP
        impact_points = base_score * impact_ratio
        rows.append(
            {
                "Factor": spec["label"],
                "Weight": spec["weight"],
                "Observed": spec["observed_fn"](demo_inputs, county_reference),
                "Feature score": feature_score,
                "Weighted contribution": weighted_contribution,
                "Neutral weighted contribution": neutral_contribution,
                "Impact ratio": impact_ratio,
                "Estimated FICO point impact": impact_points,
                "Effect": "Boosted adjusted score" if impact_points > 0 else "Reduced adjusted score" if impact_points < 0 else "Neutral effect",
                "Why it matters": spec["why"],
            }
        )
    return pd.DataFrame(rows).sort_values("Estimated FICO point impact", ascending=False)


def summarize_demographic_adjustment(demographic_score, base_score):
    impact_pct = float(np.clip(((demographic_score - 0.5) / 0.5) * DEMOGRAPHIC_IMPACT_CAP, -DEMOGRAPHIC_IMPACT_CAP, DEMOGRAPHIC_IMPACT_CAP))
    impact_points = base_score * impact_pct
    adjusted_score = int(round(np.clip(base_score + impact_points, 300, 850)))
    return {
        "demographic_score": demographic_score,
        "impact_pct": impact_pct,
        "impact_points": adjusted_score - base_score,
        "adjusted_score": adjusted_score,
    }


def render_sql_summary(component_df):
    drags = component_df.sort_values("Score points left on the table", ascending=False).head(3)
    strengths = component_df.sort_values("Score points earned", ascending=False).head(3)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Main pressures on the base score")
        for _, row in drags.iterrows():
            st.write(
                f"- **{row['Factor']}**: {row['Why it matters']} Current signal: `{row['What the app saw']}`. "
                f"Estimated points missed: `{row['Score points left on the table']:.1f}`."
            )

    with col2:
        st.markdown("#### Main strengths supporting the base score")
        for _, row in strengths.iterrows():
            st.write(
                f"- **{row['Factor']}**: {row['Why it matters']} Current signal: `{row['What the app saw']}`. "
                f"Points contributing to score: `{row['Score points earned']:.1f}`."
            )


def render_demographic_summary(demographic_df):
    boosts = demographic_df[demographic_df["Estimated FICO point impact"] > 0].head(3)
    drags = demographic_df[demographic_df["Estimated FICO point impact"] < 0].sort_values("Estimated FICO point impact").head(3)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Demographic boosts")
        if boosts.empty:
            st.write("No demographic factor rose above the neutral midpoint, so the adjustment did not add score support.")
        else:
            for _, row in boosts.iterrows():
                st.write(
                    f"- **{row['Factor']}**: {row['Why it matters']} Current signal: `{row['Observed']}`. "
                    f"Estimated impact: `{row['Estimated FICO point impact']:+.1f}` points."
                )

    with col2:
        st.markdown("#### Demographic drags")
        if drags.empty:
            st.write("No demographic factor pulled the adjustment down.")
        else:
            for _, row in drags.iterrows():
                st.write(
                    f"- **{row['Factor']}**: {row['Why it matters']} Current signal: `{row['Observed']}`. "
                    f"Estimated impact: `{row['Estimated FICO point impact']:+.1f}` points."
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
    ax.set_title("How the base rule-based score was built")
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
    ax.set_title("Health of each major rule-based credit dimension")
    ax.set_xlabel("Stronger position")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(min((row["Health score"] * 100) + 1.5, 99), idx, row["What the app saw"], va="center", ha="left", fontsize=9)

    plt.tight_layout()
    return fig


def plot_demographic_impact_chart(demographic_df):
    plot_df = demographic_df.sort_values("Estimated FICO point impact").copy()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = ["#d1495b" if value < 0 else "#2f855a" for value in plot_df["Estimated FICO point impact"]]
    ax.barh(plot_df["Factor"], plot_df["Estimated FICO point impact"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_title("Estimated demographic push on the adjusted FICO score")
    ax.set_xlabel("Estimated FICO point effect relative to the neutral midpoint")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        offset = 1 if row["Estimated FICO point impact"] >= 0 else -1
        align = "left" if row["Estimated FICO point impact"] >= 0 else "right"
        ax.text(row["Estimated FICO point impact"] + offset, idx, row["Observed"], va="center", ha=align, fontsize=9)

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


def describe_demographic_adjustment(adjustment):
    demographic_score = adjustment["demographic_score"]
    impact_pct = adjustment["impact_pct"]
    impact_points = adjustment["impact_points"]
    if impact_points > 0:
        direction = "lifted"
    elif impact_points < 0:
        direction = "reduced"
    else:
        direction = "left unchanged"

    return (
        f"The demographic composite scored {demographic_score:.1%}. Relative to the neutral midpoint of 50%, "
        f"that {direction} the base FICO by {impact_points:+d} points ({impact_pct:+.1%}), while respecting the ±20% cap."
    )


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
county_reference = build_county_density_reference()

with st.form("credit_form"):
    st.markdown("### Financial inputs")
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

    st.markdown("### Demographic composite")
    with st.container(border=True):
        st.caption("This second-stage composite is optional. The base SQL score is always computed first, then the demographic score can adjust it by up to 20%.")
        compare_with_demographics = st.checkbox(
            "Run a second evaluation with the demographic composite score",
            value=True,
            help="When turned off, you will see only the base rule-based score."
        )

        demo_col1, demo_col2, demo_col3 = st.columns(3)
        with demo_col1:
            Gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
            Number_of_Dependents = st.selectbox("Number of Dependents", list(range(0, 7)), index=1)
        with demo_col2:
            Education_Level = st.selectbox("Education Level", ["Primary", "Secondary", "College", "University"], index=2)
            Employment_Status = st.selectbox(
                "Employment Status",
                ["Unemployed", "Freelancer", "Contractor", "Employed", "Employed-PNP"],
                index=3
            )
            Age = st.slider("Age", min_value=15, max_value=85, value=35)
        with demo_col3:
            County_of_Residence = st.selectbox("County of Residence", kenya_counties, index=kenya_counties.index("Nairobi"))
            county_density = float(county_reference.loc[County_of_Residence, "density_index"])
            county_population = int(county_reference.loc[County_of_Residence, "assumed_population"])
            st.metric("County density index", f"{county_density:.0%}")
            st.caption(
                f"Placeholder population proxy for {County_of_Residence}: {county_population:,}. "
                "Replace `COUNTY_POPULATION_PROXIES` with real data when available."
            )

    submitted = st.form_submit_button("Evaluate Risk Score", use_container_width=True)

if submitted:
    input_data = dict(
        Outstanding_Debt=Outstanding_Debt,
        Annual_Income=Annual_Income,
        Total_EMI_per_month=Total_EMI_per_month,
        Monthly_Inhand_Salary=Monthly_Inhand_Salary,
        Num_of_Delayed_Payment=Num_of_Delayed_Payment,
        Num_of_Loan=Num_of_Loan,
        Credit_History_Age=Credit_History_Age,
        Monthly_Balance=Monthly_Balance,
        Credit_Utilization_Ratio=Credit_Utilization_Ratio
    )
    demographic_inputs = {
        "gender": Gender,
        "marital_status": Marital_Status,
        "dependents": int(Number_of_Dependents),
        "education_level": Education_Level,
        "employment_status": Employment_Status,
        "age": int(Age),
        "county": County_of_Residence,
    }

    result = run_scoring(input_data)

    st.divider()
    if result:
        base_score = int(result["Credit_Score"])
        base_rating = result["Credit_Score_Rating"]
        composite_score = float(result["Composite_Credit_Risk_Score"])
        base_grade_style = GRADE_STYLES[base_rating]
        component_df = build_sql_component_table(result)

        st.markdown("## Evaluation 1: Base rule-based scoring")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Base FICO Score", base_score)
        with metric_col2:
            st.metric("Composite score strength", f"{composite_score:.1%}")
        with metric_col3:
            st.metric("Base grade", base_rating, delta=base_grade_style["label"])

        st.markdown(
            f"""
            <div style="padding: 1rem 1.2rem; border-radius: 0.9rem; background: {base_grade_style['color']}18; border: 1px solid {base_grade_style['color']}55;">
                <div style="font-size: 1.1rem; font-weight: 700; color: {base_grade_style['color']};">{base_rating} base profile</div>
                <div style="margin-top: 0.35rem;">{describe_sql_profile(base_score, base_rating, composite_score)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_sql_summary(component_df)

        base_chart_col1, base_chart_col2 = st.columns(2)
        with base_chart_col1:
            st.markdown("#### Visual 1: Base score build-up")
            st.pyplot(plot_sql_score_breakdown(component_df), clear_figure=True, use_container_width=True)
        with base_chart_col2:
            st.markdown("#### Visual 2: Base credit health profile")
            st.pyplot(plot_sql_health_profile(component_df), clear_figure=True, use_container_width=True)

        detail_df = component_df.copy()
        detail_df["Weight"] = detail_df["Weight"].map(lambda value: f"{value:.0%}")
        detail_df["Health score"] = detail_df["Health score"].map(lambda value: f"{value:.0%}")
        detail_df["Score points earned"] = detail_df["Score points earned"].map(lambda value: f"{value:.1f}")
        detail_df["Score points left on the table"] = detail_df["Score points left on the table"].map(lambda value: f"{value:.1f}")

        st.markdown("#### Base factor-by-factor explanation")
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        if compare_with_demographics:
            demographic_df = build_demographic_component_table(demographic_inputs, base_score, county_reference)
            demographic_score = float(demographic_df["Weighted contribution"].sum())
            adjustment = summarize_demographic_adjustment(demographic_score, base_score)
            adjusted_score = adjustment["adjusted_score"]
            adjusted_rating = get_rating_from_score(adjusted_score)
            adjusted_grade_style = GRADE_STYLES[adjusted_rating]

            st.markdown("## Evaluation 2: Base rule-based scoring + demographic composite")
            compare_col1, compare_col2, compare_col3, compare_col4 = st.columns(4)
            with compare_col1:
                st.metric("Base FICO", base_score)
            with compare_col2:
                st.metric("Demographic score", f"{adjustment['demographic_score']:.1%}")
            with compare_col3:
                st.metric("Demographic effect", f"{adjustment['impact_pct']:+.1%}", delta=f"{adjustment['impact_points']:+d} points")
            with compare_col4:
                st.metric("Adjusted FICO", adjusted_score, delta=adjustment["impact_points"])

            st.markdown(
                f"""
                <div style="padding: 1rem 1.2rem; border-radius: 0.9rem; background: {adjusted_grade_style['color']}18; border: 1px solid {adjusted_grade_style['color']}55;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: {adjusted_grade_style['color']};">{adjusted_rating} adjusted profile</div>
                    <div style="margin-top: 0.35rem;">{describe_demographic_adjustment(adjustment)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            summary_left, summary_right = st.columns(2)
            with summary_left:
                st.write(
                    f"**Base score and grade:** `{base_score}` / `{base_rating}`"
                )
            with summary_right:
                st.write(
                    f"**Adjusted score and grade:** `{adjusted_score}` / `{adjusted_rating}`"
                )

            render_demographic_summary(demographic_df)

            st.markdown("#### Visual 3: Demographic contribution to the adjusted score")
            st.pyplot(plot_demographic_impact_chart(demographic_df), clear_figure=True, use_container_width=True)

            demo_detail_df = demographic_df.copy()
            demo_detail_df["Weight"] = demo_detail_df["Weight"].map(lambda value: f"{value:.0%}")
            demo_detail_df["Feature score"] = demo_detail_df["Feature score"].map(lambda value: f"{value:.0%}")
            demo_detail_df["Weighted contribution"] = demo_detail_df["Weighted contribution"].map(lambda value: f"{value:.1%}")
            demo_detail_df["Impact ratio"] = demo_detail_df["Impact ratio"].map(lambda value: f"{value:+.2%}")
            demo_detail_df["Estimated FICO point impact"] = demo_detail_df["Estimated FICO point impact"].map(lambda value: f"{value:+.1f}")

            st.markdown("#### Demographic factor-by-factor explanation")
            st.dataframe(
                demo_detail_df.loc[:, [
                    "Factor",
                    "Weight",
                    "Observed",
                    "Feature score",
                    "Weighted contribution",
                    "Impact ratio",
                    "Estimated FICO point impact",
                    "Effect",
                    "Why it matters",
                ]],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "The demographic composite uses the configured feature weights and a neutral midpoint of 50%. "
                "Positive deviations lift the base FICO, negative deviations reduce it, and the total effect is capped at ±20%."
            )
        else:
            st.info("Demographic composite was skipped for this run. The result shown above is the base rule-based score only.")

        with st.expander("Submitted inputs and raw outputs", expanded=False):
            st.json(
                {
                    "financial_inputs": input_data,
                    "demographic_inputs": demographic_inputs,
                    "sql_result": result,
                }
            )
    else:
        st.error("No score - check logs")
