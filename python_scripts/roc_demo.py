import sys
sys.path.append('.')

from pipelines.kingametric_xgb_pipeline import KingaMetricXGB
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Instantiate and train model
model = KingaMetricXGB()
print("Training model...")
model.train('datasets/kingametric_credit_risk.csv')

# Generate y_true, y_proba for demo
df = pd.read_csv('datasets/kingametric_credit_risk.csv')
df = df.drop(columns=model.leaky_features, errors='ignore')
df = model.add_interaction_features(df)
X = df.drop('Default_Flag', axis=1)
y = df['Default_Flag']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_temp_encoded = model.target_encode(X_temp, y_temp, fit=True)
X_test_encoded = model.target_encode(X_test)
X_test_sel = X_test_encoded[model.feature_names]
test_proba = model.xgb_model.predict_proba(X_test_sel)[:, 1]

print("Calling plot_roc_curve...")
model.plot_roc_curve(y_test, test_proba)
print("ROC curve displayed and saved to visualizations/roc_curve.png")

