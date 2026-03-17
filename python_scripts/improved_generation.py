import pandas as pd
import numpy as np

df = pd.read_csv("datasets/final_risk_analysis_dataset.csv")

print("Dataset shape:", df.shape)
print("Key features stats:")
print(df[['Outstanding_Debt', 'Annual_Income', 'Total_EMI_per_month', 'Monthly_Inhand_Salary', 'Num_of_Delayed_Payment', 'Num_of_Loan', 'Credit_Utilization_Ratio', 'Credit_History_Age', 'Delay_from_due_date', 'Monthly_Balance']].describe())

# FIXED risk_score - clip negative ratios
ratios = [
    df["Outstanding_Debt"] / df["Annual_Income"].clip(lower=1),
    df["Total_EMI_per_month"] / df["Monthly_Inhand_Salary"].clip(lower=1),
    df["Num_of_Delayed_Payment"] / (df["Num_of_Loan"]+1),
    df["Credit_Utilization_Ratio"] / 100,
    df["Num_Credit_Inquiries"] / 10,
    1 - df["Credit_History_Age"] / 120,
    df["Delay_from_due_date"]/30,
    1 - df["Monthly_Balance"] / df["Monthly_Inhand_Salary"].clip(lower=1)
]

risk_score = (
    0.22 * ratios[0].clip(0,2) +
    0.18 * ratios[1].clip(0,1) +
    0.18 * ratios[2].clip(0,5) +
    0.12 * ratios[3].clip(0,1) +
    0.10 * ratios[4].clip(0,5) +
    0.10 * ratios[5].clip(0,1) +
    0.05 * ratios[6].clip(0,3) +
    0.05 * ratios[7].clip(0,1)
)

print("\\nRisk score stats:", risk_score.describe())

# Normalize to 0-1 properly
norm_risk = (risk_score - risk_score.quantile(0.05)) / (risk_score.quantile(0.95) - risk_score.quantile(0.05))
norm_risk = np.clip(norm_risk, 0, 1)

base_pd = np.random.beta(3,10, len(df))  # mean ~0.23

prob_default = 0.55 * norm_risk + 0.45 * base_pd
prob_default = np.clip(prob_default * 0.75, 0, 1)  # Scale for ~25% overall

df["Default_Flag"] = np.random.binomial(1, prob_default)

# Tier on risk_score for balance, not prob_default
df["Borrower_Tier"] = pd.qcut(risk_score, q=[0, 0.35, 0.75, 1], labels=["Prime","Near_Prime","Subprime"], duplicates='drop')

print("\\n=== FINAL RESULTS ===")
print("Overall repayment rate:", 1 - df["Default_Flag"].mean())
print("Default_Flag distribution:", df["Default_Flag"].value_counts(normalize=True))
print("Tier distribution:", df["Borrower_Tier"].value_counts(normalize=True))
print("Default rates by tier:")
print(df.groupby("Borrower_Tier")["Default_Flag"].mean())
print("\\nSubprime default rate:", df[df["Borrower_Tier"]=="Subprime"]["Default_Flag"].mean())

#df.to_csv("datasets/improved_credit_risk.csv", index=False)
#print("\\nSaved to datasets/improved_credit_risk.csv")

