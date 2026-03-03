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