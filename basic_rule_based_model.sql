SELECT *, 

CASE
WHEN Composite_Credit_Risk_Score >= 0.75 THEN "LOW RISK"
WHEN Composite_Credit_Risk_Score >= 0.50 THEN "MODERARE RISK"
WHEN Composite_Credit_Risk_Score >= 0.25 THEN "HIGH RISK"
ELSE "VERY HIGH RISK"

END AS Credit_Risk_Category

FROM 
(
    SELECT *,
    (
    /* DEBT TO INCOME RATIO */
    (1 - 
    CASE
        WHEN Outstanding_Debt / (Annual_Income+1) > 1 THEN 1
        WHEN Outstanding_Debt / (Annual_Income+1) < 0 THEN 0
        ELSE Outstanding_Debt / (Annual_Income+1)
    END
    ) * 0.25

+

    /* EMI CREDIT PRESSURE EVALUATION */
    (1 - 
    CASE
        WHEN Total_EMI_per_month / (Monthly_Inhand_Salary+1) > 1 THEN 1
        WHEN Total_EMI_per_month / (Monthly_Inhand_Salary+1) < 0 THEN 0
        ELSE Total_EMI_per_month / (Monthly_Inhand_Salary+1)
    END
    ) * 0.20

+ 

    /* DELINQUENCY */
    (1 - 
    CASE
        WHEN Num_of_Delayed_Payment / (Num_of_Loan+1) > 1 THEN 1
        WHEN Num_of_Delayed_Payment / (Num_of_Loan+1) < 0 THEN 0
        ELSE Num_of_Delayed_Payment / (Num_of_Loan+1)
    END
    ) * 0.20

+ 

    /* CREDIT HISTORY NORMALIZATION */
    CASE
        WHEN Credit_History_Age / 120 > 1 THEN 1
        WHEN Credit_History_Age / 120 < 0 THEN 0
        ELSE Credit_History_Age / 120
    END * 0.15

+ 
    /* MONTHLY SAVING RATIO NORMALIZATION */
    CASE
        WHEN Monthly_Balance / (Monthly_Inhand_Salary+1) > 1 THEN 1
        WHEN Monthly_Balance / (Monthly_Inhand_Salary+1) < 0 THEN 0
        ELSE Monthly_Balance / (Monthly_Inhand_Salary+1)
    END * 0.10

+ 

    /* CREDIT UTILIZATION NORMALIZATION */
    (1 - 
    CASE
        WHEN Credit_Utilization_Ratio > 1 THEN 1
        WHEN Credit_Utilization_Ratio < 0 THEN 0
        ELSE Credit_Utilization_Ratio
    END
    ) * 0.10

) AS Composite_Credit_Risk_Score

FROM risk_analysis_dataset

) t;