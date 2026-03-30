import pandas as pd
import numpy as np
import joblib
import os
import category_encoders as ce
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class KingaMetricCreditRiskModel:

    def __init__(self):
        self.target_encoder = None
        self.feature_names = None
        self.best_params = {}
        self.input_columns = None
        self.cat_cols = None
        self.scaler = StandardScaler()  # 🔥 NEW

    # =========================
    # DATA PREP
    # =========================
    def load_and_preprocess(self, filepath):
        df = pd.read_csv(filepath)

        df = df.drop(columns=[
            'Payment_Behaviour',
            'Delay_from_due_date',
            'Num_of_Delayed_Payment'
        ], errors='ignore')

        df = self.add_interaction_features(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        X = df.drop('Default_Flag', axis=1)
        y = df['Default_Flag']

        self.input_columns = X.columns.tolist()
        return X, y

    def add_interaction_features(self, df):
        df = df.copy()

        df["Debt_Stress"] = df["normalized_dti"] * df["normalized_utilization"]
        df["Repayment_Stress"] = df["normalized_emi"] * df["normalized_delinquency"]
        df["Credit_Exposure"] = df["Num_Credit_Card"] * df["Credit_Utilization_Ratio"]

        df["Income_Q"] = pd.qcut(df["Annual_Income"], 4, labels=['Q1','Q2','Q3','Q4'])

        return df

    # =========================
    # ENCODING
    # =========================
    def fit_target_encoder_oof(self, X, y):  # ALTERED METHOD

        # 🔥 FIX: ensure no categorical dtype blocks assignment
        X = X.copy()
        for col in X.columns:
            if str(X[col].dtype) == "category":
                X[col] = X[col].astype(object)

        self.cat_cols = [
            c for c in X.columns
            if X[c].dtype == 'object'
        ]

        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        X_encoded = X.copy()

        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            encoder = ce.TargetEncoder(cols=self.cat_cols)
            encoder.fit(X_train, y_train)

            X_encoded.iloc[val_idx] = encoder.transform(X_val)

        # final encoder
        self.target_encoder = ce.TargetEncoder(cols=self.cat_cols)
        self.target_encoder.fit(X, y)

        return X_encoded

    # =========================
    # FEATURE SELECTION
    # =========================
    def feature_selection(self, X, y):
        mi = mutual_info_classif(X, y, random_state=42)
        mi = pd.Series(mi, index=X.columns)
        cols = mi[mi > 0].index
        return X[cols], cols.tolist()

    # =========================
    # PREPROCESS
    # =========================
    def preprocess(self, X, fit=False, y=None):  # ALTERED METHOD
        X = self.add_interaction_features(X)

        # 🔥 ensure categorical consistency
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype(str)

        if fit:
            X_enc = self.fit_target_encoder_oof(X, y)
            X_enc, self.feature_names = self.feature_selection(X_enc, y)
        else:
            X_enc = self.target_encoder.transform(X)
            X_enc = X_enc[self.feature_names]

        return X_enc, X
    # =========================
    # TRAIN MODELS
    # =========================
    def train_models(self, X_enc, X_raw, y):

        cat_features = [X_raw.columns.get_loc(c) for c in self.cat_cols if c in X_raw.columns]

        self.models = {
            "rf": RandomForestClassifier(n_estimators=400, random_state=42),
            "xgb": XGBClassifier(n_estimators=400, tree_method='hist', enable_categorical=True, random_state=42),
            "lgbm": LGBMClassifier(n_estimators=400, force_col_wise=True, random_state=42),
            "cat": CatBoostClassifier(iterations=400, verbose=0, cat_features=cat_features, random_state=42)
        }

        for name, model in self.models.items():
            if name == "cat":
                model.fit(X_raw, y)
            else:
                model.fit(X_enc, y)

    # =========================
    # STACKING (MANUAL)
    # =========================
    def fit_meta_model(self, X_enc, X_raw, y):

        preds = []

        for name, model in self.models.items():
            if name == "cat":
                p = model.predict_proba(X_raw)[:,1]
            else:
                p = model.predict_proba(X_enc)[:,1]

            preds.append(p)

        meta_X = np.column_stack(preds)

        # 🔥 SCALE FIX
        meta_X = self.scaler.fit_transform(meta_X)

        self.meta_model = LogisticRegression(max_iter=5000)
        self.meta_model.fit(meta_X, y)

    # =========================
    # TRAIN PIPELINE
    # =========================
    def train(self, filepath):

        X, y = self.load_and_preprocess(filepath)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_enc_tr, X_raw_tr = self.preprocess(X_tr, fit=True, y=y_tr)
        X_enc_te, X_raw_te = self.preprocess(X_te, fit=False)

        self.train_models(X_enc_tr, X_raw_tr, y_tr)
        self.fit_meta_model(X_enc_tr, X_raw_tr, y_tr)

        preds = self.predict_proba(X_te)
        auc = roc_auc_score(y_te, preds)

        print("Final AUC:", auc)
        return auc

    # =========================
    # PREDICT
    # =========================
    def predict_proba(self, X):

        X_enc, X_raw = self.preprocess(X, fit=False)

        preds = []

        for name, model in self.models.items():
            if name == "cat":
                p = model.predict_proba(X_raw)[:,1]
            else:
                p = model.predict_proba(X_enc)[:,1]
            preds.append(p)

        meta_X = np.column_stack(preds)
        meta_X = self.scaler.transform(meta_X)

        return self.meta_model.predict_proba(meta_X)[:,1]

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, path="model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

if __name__ == '__main__':
    model = KingaMetricCreditRiskModel()
    auc = model.train('datasets/kingametric_credit_risk.csv')
    model.save()