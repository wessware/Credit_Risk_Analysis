import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

def fix_skew(col, method='auto'):
    """Apply best skew fix method"""
    s = skew(col.dropna())
    if abs(s) < 0.5:
        return col
    if method == 'log':
        return np.log1p(col)
    elif method == 'sqrt':
        return np.sqrt(np.maximum(col, 0))
    elif method == 'boxcox':
        trans, _ = stats.boxcox(col + 1)
        return pd.Series(trans, index=col.index)
    else:  # Yeo-Johnson
        trans, _ = stats.yeojohnson(col)
        return pd.Series(trans, index=col.index)

df = pd.read_csv('datasets/kingametric_credit_risk.csv')

print("Original skew:")
print(df.select_dtypes(include=np.number).skew().sort_values(ascending=False))

# Identify continuous numeric cols (exclude cats/binary/normalized 0-1)
cont_cols = df.select_dtypes(include=np.number).columns.tolist()
exclude = ['Default_Flag', 
           'Num_Bank_Accounts', 
           'Num_Credit_Card', 
           'Num_of_Loan', 
           'Num_of_Delayed_Payment', 
           'Num_Credit_Inquiries', 
           'Credit_History_Age', 
           'Credit_Mix', 
           'Payment_of_Min_Amount', 
           'Changed_Credit_Limit', 
           'normalized_dti', 
           'normalized_emi', 
           'normalized_delinquency', 
           'normalized_credit_history', 
           'normalized_savings', 
           'normalized_utilization', 
           'normalized_utilization_risk', 
           'normalized_inquiry_intensity', 
           'normalized_investment_ratio', 
           'normalized_loan_burden_index', 
           'behavioral_risk_indicator', 
           'credit_mix_quality', 
           'normalized_savings_capacity_ratio', 
           'population_density_factor', 
           'Borrower_Tier', 
           'Debt_Stress']
cont_cols = [c for c in cont_cols if c not in exclude and df[c].nunique() > 20]

print(f"\nFixing {len(cont_cols)} continuous cols...")

df_fixed = df.copy()
for col in cont_cols:
    print(f"{col}: skew {skew(df[col].dropna()):.3f}", end=' -> ')
    df_fixed[col] = fix_skew(df[col])
    print(f"{skew(df_fixed[col].dropna()):.3f}")

# Before/after viz
fig, axes = plt.subplots(2, len(cont_cols[:6]), figsize=(20,8))
for i, col in enumerate(cont_cols[:6]):
    axes[0,i].hist(df[col].dropna(), bins=50, alpha=0.7); axes[0,i].set_title(f'{col} (orig skew {skew(df[col]):.2f})')
    axes[1,i].hist(df_fixed[col].dropna(), bins=50, alpha=0.7); axes[1,i].set_title(f'{col} (fixed skew {skew(df_fixed[col]):.2f})')
plt.tight_layout()
plt.savefig('skew_before_after.png')
plt.close()

df_fixed.to_csv('datasets/skew_fixed_credit_risk.csv', index=False)
print("\n✅ Skew-fixed dataset saved: datasets/skew_fixed_credit_risk.csv")
print("\nNew skew:")
print(df_fixed.select_dtypes(include=np.number).skew().sort_values(ascending=False))

