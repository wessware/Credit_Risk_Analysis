import numpy as np
import pandas as pd

from kenya_counties import kenya_counties

np.random.seed(42)

n = 10000

counties =  kenya_counties

density_categories = ["Very High", "High", "Medium"]

data =  pd.DataFrame({
    "Customer_ID": range(1, n+1),

    "County": np.random.choice(counties, n),

    "Population_Density_Category": np.random.choice(density_categories, n),

    "Annual_Income": np.random.normal(900000, 250000, n).clip(200000, 3000000), 

    "Monthly_Inhand_Salary": np.random.normal(75000, 15000, n).clip(12000, 200000),

    "Outstanding_Debt": np.random.normal(250000, 120000, n).clip(0, 1000000),

    "Num_of_Loan": np.random.randint(0, 8, n),

    "Total_EMI_per_month": np.random.normal(15000, 7000, n).clip(0, 8000),

    "Credit_Utilization_Ratio": np.random.uniform(0.05, 0.95, n),

    "Credit_History_Age": np.random.randint(6, 240, n),

    "Num_of_Delayed_Payment": np.random.poisson(2, n),

    "Monthly_Balance": np.random.normal(20000, 10000, n)
})

risk_score = (
    0.35*(data["Outstanding_Debt"]/ data["Annual_Income"]) + 
    0.25*(data["Total_EMI_per_month"] / data["Monthly_Inhand_Salary"]) + 
    0.25*(data["Num_of_Delayed_Payment"] / 5) +
    0.15*(data["Credit_Utilization_Ratio"])
)

proba_default = 1/(1+np.exp(-risk_score))

data["Default_Flag"] = np.random.binomial(1, proba_default)

data.to_csv("snythetic_risk_analysis_dataset.csv", index=False)