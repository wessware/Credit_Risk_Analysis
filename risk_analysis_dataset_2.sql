--Basic EDA Queries

SELECT COUNT(*) AS Total_Records from risk_analysis_dataset

-- Age distribution by number of customers

SELECT Age, COUNT(*) as Number_of_Customers

FROM risk_analysis_dataset
GROUP BY Age
ORDER BY Age;

-- Income discrepancy checks - Real value vs calculated values

SELECT Annual_Income, Monthly_Inhand_Salary, (Monthly_Inhand_Salary * 12) AS Calculated_Annual_Income, (Annual_Income - (Monthly_Inhand_Salary * 12)) as Income_discrepancy

FROM risk_analysis_dataset;

-- Count and distribution of customers for bank accounts

SELECT Num_Bank_Accounts, COUNT(*) AS Frequency

FROM risk_analysis_dataset

GROUP BY Num_Bank_Accounts
ORDER BY Num_Bank_Accounts;

-- Number of cards for income tier categories

SELECT CASE
        WHEN Annual_Income < 30000 THEN "LOW"
        WHEN Annual_Income BETWEEN 30000 AND 70000 THEN "MID"
        ELSE "HIGH"
        END AS Income_tier,
        AVG(Num_Credit_Card) AS Averagee_Number_of_Cards
FROM risk_analysis_dataset
GROUP BY Income_tier;


-- Check on interest rate margins

SELECT MIN(Interest_Rate) AS Minimum_Rate,
        MAX(Interest_Rate) AS Maximum_Rate,
        AVG(Interest_Rate) AS Average_Rate

FROM risk_analysis_dataset;

-- Check of delay of repayment of loans based on the number of loans a customer has 

SELECT Num_of_Loan, AVG(Num_of_Delayed_Payment) AS Average_delays

FROM risk_analysis_dataset

GROUP BY Num_of_Loan
ORDER BY Num_of_Loan;

-- Debt distribution as per credit mix categories


SELECT Credit_Mix, AVG(Outstanding_Debt) AS Average_Debt

FROM risk_analysis_dataset

GROUP BY Credit_Mix;

-- Customer repayment behavior distribution

SELECT Payment_Behaviour, COUNT(*) AS Frequency

FROM risk_analysis_dataset

GROUP BY Payment_Behaviour
ORDER BY Frequency DESC;


-- FIRST TEST MODEL

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

-- FEATURE NORMALIZATION

SELECT *,
    /*DTI NORMALIZATION*/
    GREATEST(0, LEAST(Outstanding_Debt / (Annual_Income+1), 1)) AS normalized_dti,

    /*EMI PRESSURE NORMALIZATION*/
    GREATEST(0, LEAST(Total_EMI_per_month / (Monthly_Inhand_Salary+1), 1)) AS normalized_emi,

    /*DELINQUENCY NORMALIZATION*/
    GREATEST(0, LEAST(Num_of_Delayed_Payment / (Num_of_Loan+1), 1)) AS normalized_delinquency,

    /*CREDIT HISTORY NORMALIZATION*/
    GREATEST(0, (LEAST(Credit_History_Age/120, 1))) AS normalized_credit_history,

    /*SAVINGS NORMALIZATION*/
    GREATEST(0, LEAST(Monthly_Balance / (Monthly_Inhand_Salary+1), 1)) AS normalized_savings,

    /*UTILIZATION NORMALIZATION*/
    GREATEST(0, LEAST(Credit_Utilization_Ratio, 1)) AS normalized_utilization

FROM risk_analysis_dataset;

-- MODEL WITH NORMALIZED FEATURES

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
        ) t;

-- FIRST MODEL WITH NORMALIZED FEATURES AND CREDIT CATEGORIES

SELECT *,
    CASE
        WHEN Composite_Credit_Risk_Score >= 0.75 THEN "LOW RISK"
        WHEN Composite_Credit_Risk_Score >= 0.50 THEN "MODERATE RISK"
        WHEN Composite_Credit_Risk_Score >= 0.25 THEN "HIGH RISK"
        ELSE "VERY HIGH RISK"

    END AS Credit_Risk_Category

FROM
(
    SELECT *,

        (1 - normalized_dti)            * 0.25 
        + 
        (1 - normalized_emi)            * 0.20
        +
        (1 - normalized_delinquency)    * 0.20
        +
        normalized_credit_history       * 0.15 
        +
        normalized_savings              * 0.10
        +
        (1 - normalized_utilization)    * 0.10
        
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