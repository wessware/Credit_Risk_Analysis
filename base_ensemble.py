import pandas as pd
import numpy as np
import joblib
import category_encoders as ce

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

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

        df["Debt_Stress"] = df["normalized_dti"] * df["normalized_utilization"]
        df["Repayment_Stress"] = df["normalized_emi"] * df["normalized_delinquency"]
        df["Liquidity_Index"] = df["normalized_savings"] * df["normalized_emi"]
        df["Credit_Exposure"] = df["Num_Credit_Card"] * df["Credit_Utilization_Ratio"]

        df["Risk_Index"] = (
            df["normalized_dti"] +
            df["normalized_utilization"] +
            df["normalized_delinquency"]
        ) / 3

        return df

    # =========================
    # ENCODING
    # =========================
    def target_encode(self, X, y=None, fit=False):
        cat_cols = [c for c in ["Payment_of_Min_Amount", "Credit_Mix", "Borrower_Tier"] if c in X.columns]

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
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        importances = pd.Series(0, index=X.columns)

        for train_idx, val_idx in folds.split(X, y):
            model = XGBClassifier(n_estimators=300, max_depth=4, random_state=42)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            importances += pd.Series(model.feature_importances_, index=X.columns)

        importances /= folds.n_splits

        top_features = importances.sort_values(ascending=False).head(20).index

        return X[top_features], top_features

    # =========================
    # MODEL TRAINING
    # =========================
    def train_models(self, X, y):
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        scale = neg / (pos + 1e-6)

        self.models["xgb"] = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            tree_method="hist",
            random_state=42
        )

        self.models["lgbm"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            force_col_wise=True,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42
        )

        self.models["cat"] = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            verbose=0,
            auto_class_weights="Balanced",
            random_state=42
        )

        for model in self.models.values():
            model.fit(X, y)

    # =========================
    # LOGIT FUNCTION
    # =========================
    def _logit(self, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # =========================
    # ENSEMBLE WEIGHTS (FIXED)
    # =========================
    def optimize_weights(self, X, y):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        preds = {name: np.zeros(len(X)) for name in self.models}

        for train_idx, val_idx in folds.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]

            for name, model in self.models.items():
                m = clone(model)  # ✅ FIXED
                m.fit(X_tr, y_tr)
                preds[name][val_idx] = m.predict_proba(X_val)[:, 1]

        best_auc = 0
        best_weights = None

        for w1 in np.linspace(0.2, 0.6, 5):
            for w2 in np.linspace(0.2, 0.6, 5):
                w3 = 1 - w1 - w2
                if w3 <= 0:
                    continue

                logits = (
                    w1 * self._logit(preds["xgb"]) +
                    w2 * self._logit(preds["lgbm"]) +
                    w3 * self._logit(preds["cat"])
                )

                ensemble = 1 / (1 + np.exp(-logits))
                auc = roc_auc_score(y, ensemble)

                if auc > best_auc:
                    best_auc = auc
                    best_weights = (w1, w2, w3)

        self.weights = best_weights
        print("Best Weights:", self.weights, "AUC:", best_auc)

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

        # ✅ Optimize ensemble weights
        self.optimize_weights(X_temp, y_temp)

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

        w1, w2, w3 = self.weights

        logits = (
            w1 * self._logit(self.models["xgb"].predict_proba(X)[:, 1]) +
            w2 * self._logit(self.models["lgbm"].predict_proba(X)[:, 1]) +
            w3 * self._logit(self.models["cat"].predict_proba(X)[:, 1])
        )

        return 1 / (1 + np.exp(-logits))

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)

    def save(self, path="kinga_ensemble.pkl"):
        joblib.dump(self, path)
        print(f"Saved to {path}")


if __name__ == '__main__':
    model = KingaMetricCreditRiskModel()
    auc = model.train('datasets/improved_credit_risk.csv')