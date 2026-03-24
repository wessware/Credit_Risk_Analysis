import pandas as pd
import numpy as np
import joblib
import category_encoders as ce
import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class KingaMetricCreditRiskModel:
    def __init__(self):
        self.models = {}
        self.weights = None
        self.target_encoder = None
        self.feature_names = None
        self.best_threshold = 0.5
        self.best_params = {}
        
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

        return X, y

    def add_interaction_features(self, df):
        df = df.copy()

        # Original
        df["Debt_Stress"] = df["normalized_dti"] * df["normalized_utilization"]
        df["Repayment_Stress"] = df["normalized_emi"] * df["normalized_delinquency"]
        df["Liquidity_Index"] = df["normalized_savings"] * df["normalized_emi"]
        df["Credit_Exposure"] = df["Num_Credit_Card"] * df["Credit_Utilization_Ratio"]
        df["Risk_Index"] = (
            df["normalized_dti"] +
            df["normalized_utilization"] +
            df["normalized_delinquency"]
        ) / 3

        # New advanced
        df["Income_Delinq"] = df["Annual_Income"] * df["normalized_delinquency"]
        df["Age_Util"] = df["Age"] * df["normalized_utilization"]
        df["Loan_DTI"] = df["Num_of_Loan"] * df["normalized_dti"]

        # Bins
        df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 40, 50, 65, 100], labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder']).astype('str')
        df["Income_Q"] = pd.qcut(df["Annual_Income"], 4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype('str')

        # Poly top features (util, dti, delinq)
        for feat in ['Credit_Utilization_Ratio', 'normalized_dti', 'normalized_delinquency']:
            df[f'{feat}_sq'] = df[feat]**2
            df[f'{feat}_log'] = np.log1p(df[feat])

        return df

    # =========================
    # ENCODING
    # =========================
    def target_encode(self, X, y=None, fit=False):
        cat_cols = [c for c in ["Payment_of_Min_Amount", "Credit_Mix", "Borrower_Tier", "Age_Group", "Income_Q"] if c in X.columns]

        if fit:
            encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10)
            X = encoder.fit_transform(X, y)
            self.target_encoder = encoder
        else:
            if self.target_encoder is None:
                raise ValueError("Target encoder not fitted.")
            X = self.target_encoder.transform(X)

        return X

    # =========================
    # FEATURE SELECTION (CV STABLE)
    # =========================
    def feature_selection(self, X, y):
        from sklearn.feature_selection import mutual_info_classif

        # MI
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        # Keep top 30 (all after FE)
        top_features = mi_scores.head(30).index
        return X[top_features], top_features.tolist()

    # =========================
    # MODEL TRAINING
    # =========================
    def tune_hyperparams(self, X, y):
        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'scale_pos_weight': (y==0).sum() / (y==1).sum()
            }
            model = XGBClassifier(**params, tree_method='hist', random_state=42)
            cv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            return scores.mean()

        def lgbm_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
            model = LGBMClassifier(**params, force_col_wise=True, random_state=42, verbose=-1)
            cv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            return scores.mean()

        def cat_objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 200, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
            }
            model = CatBoostClassifier(**params, verbose=0, random_state=42, auto_class_weights='Balanced')
            cv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            return scores.mean()

        print("Tuning XGB...")
        xgb_study = optuna.create_study(direction='maximize')
        xgb_study.optimize(xgb_objective, n_trials=50)
        self.best_params['xgb'] = xgb_study.best_params

        print("Tuning LGBM...")
        lgbm_study = optuna.create_study(direction='maximize')
        lgbm_study.optimize(lgbm_objective, n_trials=50)
        self.best_params['lgbm'] = lgbm_study.best_params

        print("Tuning Cat...")
        cat_study = optuna.create_study(direction='maximize')
        cat_study.optimize(cat_objective, n_trials=50)
        self.best_params['cat'] = cat_study.best_params

        print("Best params:", self.best_params)

    def train_models(self, X, y):
        self.tune_hyperparams(X, y)

        scale = (y == 0).sum() / (y == 1).sum()

        self.base_models = {
            "xgb": XGBClassifier(**self.best_params['xgb'], tree_method="hist", random_state=42),
            "lgbm": LGBMClassifier(**self.best_params['lgbm'], force_col_wise=True, random_state=42, verbose=-1),
            "cat": CatBoostClassifier(**self.best_params['cat'], verbose=0, random_state=42, auto_class_weights="Balanced")
        }

        self.stacking_model = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            stack_method="predict_proba"
        )

        self.stacking_model.fit(X, y)

    # =========================
    # LOGIT FUNCTION
    # =========================
    def _logit(self, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # =========================
    # ENSEMBLE WEIGHTS (FIXED)
    # =========================
    # Legacy logit, disabled for stacking
    def optimize_weights(self, X, y):
        print("Using stacking - no weights optimization needed.")

    # =========================
    # THRESHOLD
    # =========================
    def optimize_threshold(self, y_true, y_probs):
        thresholds = np.linspace(0.1, 0.9, 50)
        best_ks, best_t = 0, 0.5

        for t in thresholds:
            preds = (y_probs >= t).astype(int)

            tpr = ((preds == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
            fpr = ((preds == 1) & (y_true == 0)).sum() / (y_true == 0).sum()

            ks = tpr - fpr

            if ks > best_ks:
                best_ks = ks
                best_t = t

        self.best_threshold = best_t

    # =========================
    # TRAIN PIPELINE
    # =========================
    def train(self, filepath):
        print("Loading...")
        X, y = self.load_and_preprocess(filepath)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ✅ Encode
        X_temp = self.target_encode(X_temp, y_temp, fit=True)
        X_test_encoded = self.target_encode(X_test, fit=False)

        # ✅ Feature selection
        X_temp, selected = self.feature_selection(X_temp, y_temp)
        X_test_encoded = X_test_encoded[selected]
        self.feature_names = selected

        # ✅ Train models
        self.train_models(X_temp, y_temp)

        # Stacking already trained, skip weights

        # ✅ Evaluate (NO double encoding)
        test_probs = self.predict_proba(X_test, already_encoded=False)
        auc = roc_auc_score(y_test, test_probs)

        self.optimize_threshold(y_test, test_probs)

        print(f"Final Test AUC: {auc:.4f}")
        print(f"Best Threshold: {self.best_threshold:.3f}")

        return auc

    # =========================
    # PREDICT
    # =========================
    def predict_proba(self, X, already_encoded=False):
        X = self.add_interaction_features(X)

        if not already_encoded:
            X = self.target_encoder.transform(X)

        X = X[self.feature_names]

        return self.stacking_model.predict_proba(X)[:, 1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)

    def save(self, path="kinga_ensemble.pkl"):
        joblib.dump(self, path)
        print(f"Saved to {path}")


if __name__ == '__main__':
    model = KingaMetricCreditRiskModel()
    auc = model.train('datasets/improved_credit_risk.csv')