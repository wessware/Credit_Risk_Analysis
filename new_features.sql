-- 1: DEBT TO INCOME RATIO

SELECT *,
(Outstanding_Debt / (Annual_Income + 1)) AS Debt_to_Income

FROM risk_analysis_dataset;


-- 2: EMI TO INCOME RATIO

SELECT *,
(Total_EMI_per_month / (Monthly_Inhand_Salary + 1)) AS EMI_to_Income

FROM risk_analysis_dataset;

-- 3: CREDIT UTILIZATION PRESSURE

SELECT *,
(Credit_Utilization_Ratio * Num_Credit_Card) AS Utilization_Risk

FROM risk_analysis_dataset;

-- 4: DELINQUENCY RATE

SELECT *,
(Num_of_Delayed_Payment / (Num_of_Loan + 1)) AS Delinquency_Rate

FROM risk_analysis_dataset;


-- 5: INUQUIRY INTESITY

SELECT *,
(Num_Credit_Inquiries / (Num_Credit_Card + 1)) AS Inquiry_Rate

FROM risk_analysis_dataset;

-- 6: SAVINGS RATIO

SELECT *,
(Monthly_Balance / (Monthly_Inhand_Salary + 1)) AS Savings_Ratio

FROM risk_analysis_dataset;

-- 7: INVESTMENT RATIO

SELECT *,
(Amount_invested_monthly / (Monthly_Inhand_Salary + 1)) AS Investment_Ratio

FROM risk_analysis_dataset;

-- 8: CREDIT HISTORY IN YEARS

SELECT *,
(Credit_History_Age / 12) AS Credit_History_Years

FROM risk_analysis_dataset;

-- 9: LOAN BURDEN INDEX

SELECT *,
(Num_of_Loan * Outstanding_Debt) AS Loan_Burden

FROM risk_analysis_dataset;

-- 10: BEHAVIORAL RISK INDICATORS

SELECT *,
CASE
    WHEN Payment_of_Min_Amount = 'Yes' THEN 1
    ELSE 0
END AS Min_Payment_Flag
FROM risk_analysis_dataset;