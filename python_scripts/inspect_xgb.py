from pipelines.kingametric_xgb_pipeline import KingaMetricXGB
import joblib
import pandas as pd

model = joblib.load("pickled_models/kingametric_xgb_1.pkl")
print("Feature names (top 30):")
print(model.feature_names)
print("\nTop 20 importances:")
importances = pd.Series(model.xgb_model.feature_importances_, index=model.feature_names).nlargest(20)
print(importances)
print("\nAll leaky_features:")
print(model.leaky_features)
print("\nCategoricals:")
cat_cols = ["Payment_of_Min_Amount", "Credit_Mix", "Borrower_Tier", "Income_Q"]
print([c for c in cat_cols if c in model.feature_names])
