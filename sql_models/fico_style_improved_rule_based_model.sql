

SELECT *,

    300 + (Composite_Credit_Risk_Score *550) AS Credit_Score,

    CASE
        WHEN(300 + Composite_Credit_Risk_Score * 550) >= 750 THEN "EXCELLENT"
        WHEN(300 + Composite_Credit_Risk_Score * 550) >= 700 THEN "GOOD"
        WHEN(300 + Composite_Credit_Risk_Score * 550) >= 650 THEN "FAIR"
        WHEN(300 + Composite_Credit_Risk_Score * 550) >= 550 THEN "POOR"
        ELSE "VERY POOR"

    END AS Credit_Score_Rating

FROM 
(
    SELECT *,

        (1 - normalized_dti) * 0.25 
        +
        (1 - normalized_emi) * 0.20
        + 
        (1 - normalized_delinquency) * 0.20 
        +
        normalized_credit_history * 0.15 
        +
        normalized_savings * 0.10 
        +
        (1 - normalized_utilization) * 0.10

        AS Composite_Credit_Risk_Score

    FROM 
    (
        SELECT *,

            GREATEST(0, LEAST(Outstanding_Debt / (Annual_Income+1), 1)) AS normalized_dti,
            GREATEST(0, LEAST(Total_EMI_per_month / (Monthly_Inhand_Salary+1), 1)) AS normalized_emi,
            GREATEST(0, LEAST(Num_of_Delayed_Payment / (Num_of_Loan+1), 1)) AS normalized_delinquency,
            GREATEST(0, LEAST(Credit_History_Age/120, 1)) AS normalized_credit_history,
            GREATEST(0, LEAST(Monthly_Balance / (Monthly_Inhand_Salary+1), 1)) AS normalized_savings,
            GREATEST(0, LEAST(Credit_Utilization_Ratio, 1)) AS normalized_utilization

    FROM risk_analysis_dataset
    ) normalized
) scored;