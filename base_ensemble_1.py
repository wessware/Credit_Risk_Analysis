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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class KingaMetricCreditRiskModel:
    def __init__(self):
        self.models = {}
        self.target_encoder = None
        self.feature_names = None
        self.best_threshold = 0.5
        self.best_params = {}
        self.cat_cols = None
        self.input_columns = None

        self.leaky_features = [
            'Payment_Behaviour',
            'Delay_from_due_date',
            'Num_of_Delayed_Payment'
        ]

    # =========================
    # DATA PREP
    # =========================
    def load_and_preprocess(self, filepath):
        df = pd.read_csv(filepath)

        df = df.drop(columns=[c for c in self.leaky_features if c in df.columns], errors='ignore')

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
        df["Liquidity_Index"] = df["normalized_savings"] * df["normalized_emi"]
        df["Credit_Exposure"] = df["Num_Credit_Card"] * df["Credit_Utilization_Ratio"]
        df["Risk_Index"] = (
            df["normalized_dti"] +
            df["normalized_utilization"] +
            df["normalized_delinquency"]
        ) / 3

        df["Income_Delinq"] = df["Annual_Income"] * df["normalized_delinquency"]
        df["Loan_DTI"] = df["Num_of_Loan"] * df["normalized_dti"]

        df["Income_Q"] = pd.qcut(df["Annual_Income"], 4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype('str')

        for feat in ['Credit_Utilization_Ratio', 'normalized_dti', 'normalized_delinquency']:
            df[f'{feat}_sq'] = df[feat]**2
            df[f'{feat}_log'] = np.log1p(df[feat])

        return df

    # =========================
    # ENCODING
    # =========================
    def fit_target_encoder_oof(self, X, y, n_splits=5):
        cat_cols = [c for c in ["Payment_of_Min_Amount", "Credit_Mix", "Borrower_Tier", "Income_Q"] if c in X.columns]
        self.cat_cols = cat_cols

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X_encoded = X.copy()

        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10)
            encoder.fit(X_train, y_train)

            X_encoded.iloc[val_idx] = encoder.transform(X_val)

        self.target_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10)
        self.target_encoder.fit(X, y)

        return X_encoded

    # =========================
    # FEATURE SELECTION
    # =========================
    def feature_selection(self, X, y):
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        top_features = mi_scores[mi_scores > 0]
        return X[top_features.index], top_features.index.tolist()

    # =========================
    # PREPROCESS PIPELINE
    # =========================
    def preprocess(self, X, fit=False, y=None):
        X = self.add_interaction_features(X)

        if fit:
            X = self.fit_target_encoder_oof(X, y)
            X, selected = self.feature_selection(X, y)
            self.feature_names = selected
        else:
            X = self.target_encoder.transform(X)
            X = X[self.feature_names]

        # 🔥 UPDATED: FORCE NUMERIC (fix for categorical error)
        X = X.apply(pd.to_numeric, errors='coerce')

        # 🔥 UPDATED: SAFETY FILL
        X = X.fillna(0)

        return X

    # =========================
    # MODEL TRAINING
    # =========================
    def tune_hyperparams(self, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def objective_factory(model_name):
            def objective(trial):
                if model_name == "xgb":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                        'max_depth': trial.suggest_int('max_depth', 4, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    }
                    # 🔥 UPDATED: REMOVED enable_categorical
                    model = XGBClassifier(**params, tree_method='hist', random_state=42)

                elif model_name == "lgbm":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    }
                    # 🔥 UPDATED: REMOVED categorical_feature
                    model = LGBMClassifier(**params, random_state=42)

                elif model_name == "rf":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                        'max_depth': trial.suggest_int('max_depth', 4, 12),
                    }
                    model = RandomForestClassifier(**params, random_state=42)

                else:
                    params = {
                        'iterations': trial.suggest_int('iterations', 200, 800),
                        'depth': trial.suggest_int('depth', 4, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    }
                    model = CatBoostClassifier(**params, verbose=0, random_state=42)

                return cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

            return objective

        for name in ["rf", "xgb", "lgbm", "cat"]:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_factory(name), n_trials=30)
            self.best_params[name] = study.best_params

    def train_models(self, X, y):
        self.tune_hyperparams(X, y)

        self.base_models = {
            "rf": RandomForestClassifier(**self.best_params['rf'], random_state=42),
            # 🔥 UPDATED: removed categorical flags
            "xgb": XGBClassifier(**self.best_params['xgb'], tree_method="hist", random_state=42),
            "lgbm": LGBMClassifier(**self.best_params['lgbm'], random_state=42),
            "cat": CatBoostClassifier(**self.best_params['cat'], verbose=0, random_state=42)
        }

        stacking = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            stack_method="predict_proba",
            passthrough=True
        )

        self.stacking_model = CalibratedClassifierCV(stacking, method='isotonic', cv=3)
        self.stacking_model.fit(X, y)

    # =========================
    # VALIDATION
    # =========================
    def validate_input(self, X):
        missing = set(self.input_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    # =========================
    # TRAIN PIPELINE
    # =========================
    def train(self, filepath):
        X, y = self.load_and_preprocess(filepath)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_temp = self.preprocess(X_temp, fit=True, y=y_temp)
        X_test = self.preprocess(X_test, fit=False)

        self.train_models(X_temp, y_temp)

        test_probs = self.predict_proba(X_test, already_processed=True)
        auc = roc_auc_score(y_test, test_probs)

        print(f"Final AUC: {auc:.4f}")
        return auc

    # =========================
    # PREDICT
    # =========================
    def predict_proba(self, X, already_processed=False):
        if not already_processed:
            self.validate_input(X)
            X = self.preprocess(X, fit=False)

        return self.stacking_model.predict_proba(X)[:, 1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, path="pickled_models/kinga_ensemble_pro.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


if __name__ == '__main__':
    model = KingaMetricCreditRiskModel()
    auc = model.train('datasets/kingametric_credit_risk.csv')
    model.save()