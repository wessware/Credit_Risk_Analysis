-- Basic EDA queries for the loan_data_clean table

-- view the rows of the cleaned loan data
SELECT * from loan_data_clean

--view the total number of loans in the dataset
SELECT COUNT(*) as total_loans FROM loan_data_clean;

--view the unique loan statuses in the dataset
SELECT DISTINCT Loan_Status FROM loan_data_clean;

--view the count of each loan status in the dataset
SELECT Loan_Status, COUNT(*) as count FROM loan_data_clean GROUP BY Loan_Status;

--view the minimum, maximum, average loan amount 
SELECT Loan_Status, 
MIN(LoanAmount) as Minimum, 
MAX(LoanAmount) as Maximum, 
AVG(LoanAmount) as Average FROM loan_data_clean 

GROUP BY Loan_Status;

--view the average applicant income 
SELECT AVG(ApplicantIncome) as Average_Income FROM loan_data_clean;

SELECT Loan_Status,
AVG(AppliantIncome) as Average_Income from loan_data_clean
GROUP BY Loan_Status;

SELECT Loan_Status,
AVG(LoanAmount) as Average_Loan_Amount from loan_data_clean
GROUP BY Loan_Status;



SELECT MIN(LoanAmount) as Min_Loan_Amount, 
MAX(LoanAmount) as Max_Loan_Amount FROM loan_data_clean;

SELECT MIN(ApplicantIncome) as Minimum_Income, 
MAX(ApplicantIncome) as Maximum_Income FROM loan_data_clean;

SELECT Property_Area, COUNT(*) as Count from loan_data_clean 
GROUP BY Property_Area;

SELECT Education, COUNT(*) as Count from loan_data_clean
GROUP BY Education;


SELECT ApplicantIncome, LoanAmount from loan_data_clean
ORDER BY ApplicantIncome DESC;

--INSIGHT ANALYSIS QUERIES

SELECT Gender, 
AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME, 
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT from loan_data_clean

GROUP BY Gender;


SELECT Married, 
AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME,
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT from loan_data_clean

GROUP BY Married;


SELECT Self_Employed, 

AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME,
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT from loan_data_clean

GROUP BY Self_Employed;


SELECT Credit_History, 

AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME,
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT from loan_data_clean

GROUP BY Credit_History;


SELECT Property_Area,

AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME,
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT from loan_data_clean

GROUP BY Property_Area;

SELECT Gender, 

SUM(CASE WHEN Loan_Status=1 THEN 1 ELSE 0 END)*100/COUNT(*) as 
LOAN_APPROVAL_RATE from loan_data_clean

GROUP BY Gender;

SELECT Married, 

SUM(CASE WHEN Loan_Status=1 THEN 1 ELSE 0 END)*100/COUNT(*) as 
LOAN_APPROVAL_RATE from loan_data_clean

GROUP BY Married;


SELECT Education, 

AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME,
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT from loan_data_clean

GROUP BY Education;


SELECT Dependents,

AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME,
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT from loan_data_clean

GROUP BY Dependents;

SELECT Property_Area,

AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME,
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT from loan_data_clean

GROUP BY Property_Area;


SELECT Self_Employed,

SUM(CASE WHEN Loan_Status=1 THEN 1 ELSE 0 END)*100/COUNT(*) as LOAN_APPROVAL_RATE 
from loan_data_clean

GROUP BY Self_Employed;


SELECT ApplicantIncome, LoanAmount, LoanAmount/ApplicantIncome as 
LOAN_INCOME_RATIO from loan_data_clean

WHERE ApplicantIncome IS NOT NULL AND LoanAmount IS NOT NULL

ORDER BY LOAN_INCOME_RATIO ASC;

SELECT Property_Area, 
SUM(CASE WHEN Loan_Status=1 THEN 1 ELSE 0 END)*100/COUNT(*) as Default_Rate 
from loan_data_clean

GROUP BY Property_Area;


SELECT Education, Self_Employed, 

SUM(CASE WHEN Loan_Status=1 THEN 1 ELSE 0 END)*100/COUNT(*) as 
LOAN_APPROVAL_RATE from loan_data_clean

GROUP BY Education, Self_Employed;



SELECT ApplicantIncome, LoanAmount from loan_data_clean
WHERE ApplicantIncome < 100 AND LoanAmount > 100;


SELECT Credit_History, Property_Area,  
AVG(LoanAmount) as AVERAGE_LOAN_AMOUNT,
AVG(ApplicantIncome) as AVERAGE_APPLICANT_INCOME

from loan_data_clean

GROUP BY Credit_History, Property_Area;



SELECT Gender, Married,

SUM(CASE WHEN Loan_Status=1 THEN 1 ELSE 0 END)*100/COUNT(*) as LOAN_APPROVAL_RATE 

from loan_data_clean

GROUP BY Gender, Married;




SELECT Dependents, Property_Area,

SUM(CASE WHEN Loan_Status=1 THEN 1 ELSE 0 END)*100/COUNT(*) 
as LOAN_APPROVAL_RATE 
from loan_data_clean

GROUP BY Dependents, Property_Area;



SELECT LoanAmount, ApplicantIncome FROM loan_data_clean

WHERE Credit_History=0 AND LoanAmount > 150

ORDER BY LoanAmount DESC;



SELECT Education, Loan_Status,

COUNT(*) as COUNT from loan_data_clean

GROUP BY Education, Loan_Status;



SELECT 

SUM(CASE WHEN Loan_Status=0 THEN LoanAmount ELSE 0 END) as Defaulted_Loan_Amount,
SUm(CASE WHEN Loan_Status=0 THEN LoanAmount ELSE 0 END)*100/SUM(LoanAmount) 

as Portfolio_risk_percentage 

from loan_data_clean;