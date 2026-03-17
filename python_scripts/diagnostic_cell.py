import pandas as pd
import numpy as np

df = pd.read_csv("datasets/final_risk_analysis_dataset.csv")

risk_score = (
    0.22*(df["Outstanding_Debt"] / df["Annual_Income"]) +
    0.18*(df["Total_EMI_per_month"] / df["Monthly_Inhand_Salary"]) +
    0.18*(df["Num_of_Delayed_Payment"] / (df["Num_of_Loan"]+1)) +
    0.12*(df["Credit_Utilization_Ratio"] / 100) +
    0.10*(df["Num_Credit_Inquiries"] / 10) +
    0.10*(1 - df["Credit_History_Age"] / 120) +
    0.05*(df["Delay_from_due_date"]/30) +
    0.05*(1 - df["Monthly_Balance"] / df["Monthly_Inhand_Salary"])
)

norm_risk = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

base_pd = np.random.beta(2,8,len(df))

prob_default = 0.6*norm_risk + 0.4*base_pd

prob_default = np.clip(prob_default,0,1)

df["Default_Flag"] = np.random.binomial(1,prob_default)

df["Borrower_Tier"] = pd.cut(
    prob_default,
    bins=[0,0.1,0.3,1],
    labels=["Prime","Near_Prime","Subprime"]
)

print("Overall default rate:", df["Default_Flag"].value_counts(normalize=True))
print("\\nTier distribution:")
print(df["Borrower_Tier"].value_counts(normalize=True))
print("\\nDefault rates by tier:")
print(df.groupby("Borrower_Tier")["Default_Flag"].mean())
print("\\nRisk score stats:", risk_score.describe())
print("\\nProb default stats:", prob_default.describe())
print("\\nTier counts:", df["Borrower_Tier"].value_counts())

