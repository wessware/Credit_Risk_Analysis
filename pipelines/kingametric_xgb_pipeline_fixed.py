import pandas as pd
import numpy as np
import joblib
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import os
from sklearn.metrics import auc

class KingaMetricXGB:
    def __init__(self):
        self.xgb_model = None
        self.target_encoder = None
        self.feature_names = None
        self.best_threshold = 0.5
        self.leaky_features = [
            'Payment_Behaviour',
            'Delay_from_due_date',
            'Num_of_Delayed_Payment'
        ]
        self.fixed_params = {
            'n_estimators': 600,
            'max_depth': 4,
            'learning_rate': 0.011428470477839327,
            'subsample': 0.8940362813326697,
            'colsample_bytree': 0.7250627525433337,
            'reg_alpha': 1.4913690667729074,
            'reg_lambda': 7.463346775560514,
            'tree_method': 'hist',
            'random_state': 42
        }

    def load_and_preprocess(self, filepath):
        """Load data and apply initial preprocessing."""
        df = pd.read_csv(filepath)
        df = df.drop(columns=[c for c in self.leaky_features if c in df.columns], errors='ignore')
        df = self.add_interaction_features(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        X = df.drop(columns=['Default_Flag'], axis=1)
        y = df['Default_Flag']
        return X, y

    def add_interaction_features(self, df):
        """Add interaction, polynomial, and binned features."""
        df = df.copy()
        # Original interactions
        df["Debt_Stress"] = df.get("normalized_dti", pd.Series(0, index=df.index)) * df.get("normalized_utilization", pd.Series(0, index=df.index))
        df["Repayment_Stress"] = df.get("normalized_emi", pd.Series(0, index=df.index)) * df.get("normalized_delinquency", pd.Series(0, index=df.index))
        df["Liquidity_Index"] = df.get("normalized_savings", pd.Series(0, index=df.index)) * df.get("normalized_emi", pd.Series(0, index=df.index))
        df["Credit_Exposure"] = df["Num_Credit_Card"] * df["Credit_Utilization_Ratio"]
        df["Risk_Index"] = (
            df.get("normalized_dti", 0) +
            df.get("normalized_utilization", 0) +
            df.get("normalized_delinquency", 0)
        ) / 3
        # Advanced
        df["Income_Delinq"] = df["Annual_Income"] * df.get("normalized_delinquency", 0)
        df["Loan_DTI"] = df["Num_of_Loan"] * df.get("normalized_dti", 0)
        # Value-based Income_Q assignment (Solution 2 - robust for all cases)
        def assign_income_quartile(income):
            if pd.isna(income):
                return 'Q2'
            if income < 30000:
                return 'Q1'
            elif income < 60000:
                return 'Q2'
            elif income < 100000:
                return 'Q3'
            else:
                return 'Q4'
        df["Income_Q"] = df["Annual_Income"].apply(assign_income_quartile)
        # Poly features
        for feat in ['Credit_Utilization_Ratio', 'normalized_dti']:
            if feat in df.columns:
                df[f'{feat}_sq'] = df[feat]**2
                df[f'{feat}_log'] = np.log1p(df[feat])
        return df

    def target_encode(self, X, y=None, fit=False):
        """Target encode categorical columns."""
        cat_cols = [c for c in ["Payment_of_Min_Amount", "Credit_Mix", "Borrower_Tier", "Income_Q"] if c in X.columns]
        if fit:
            encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10)
            X_encoded = encoder.fit_transform(X, y)
            self.target_encoder = encoder
        else:
            if self.target_encoder is None:
                raise ValueError("Target encoder not fitted.")
            X_encoded = self.target_encoder.transform(X)
        return X_encoded

    def feature_selection(self, X, y):
        """Select top 30 features using mutual information."""
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        top_features = mi_scores.head(30).index
        self.feature_names = top_features.tolist()
        return X[top_features], top_features.tolist()

    def train_model(self, X, y):
        """Train single XGB model with fixed hyperparameters."""
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        params = self.fixed_params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        self.xgb_model = XGBClassifier(**params)
        self.xgb_model.fit(X, y)
        print(f"Trained XGB with scale_pos_weight: {scale_pos_weight:.2f}")

    def plot_top_features(self, save_path="visualizations/xgb_top_features_lean.png"):
        """Visualize top 20 features by importance."""
        if self.xgb_model is None:
            raise ValueError("Model not trained.")
        importances = pd.Series(self.xgb_model.feature_importances_, index=self.feature_names)
        top20 = importances.nlargest(35)
        plt.figure(figsize=(10, 8))
        top20.plot(kind='barh', color='skyblue')
        plt.title('Top 35 XGB Feature Importances')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        print(f"Saved feature plot to {save_path}")

    def optimize_threshold(self, y_true, y_probs):
        """Optimize threshold using KS statistic."""
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
        print(f"Optimized threshold: {best_t:.3f} (KS: {best_ks:.3f})")

    def plot_roc_curve(self, y_true, y_proba, save_path="visualizations/roc_curve_lean.png"):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate') 
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

        plt.show()

    def evaluate(self, y_true, y_proba, y_pred, save_path="visualizations/confusion_matrix_lean.png"):
        """Evaluate with ROC_AUC and Confusion Matrix."""
        auc_score = roc_auc_score(y_true, y_proba)
        c_score = classification_report(y_true, y_pred, target_names=['No Default', 'Default'])

        print(f"Test ROC_AUC: {auc_score:.4f}")
        print(f"Classification Report:\n{c_score}")

        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['No Default', 'Default'])
        disp.plot()
        plt.title('Confusion Matrix at Best Threshold')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

        plt.show()

    def train(self, filepath):
        """Full training pipeline."""
        print("Step 1: Loading and preprocessing...")
        X, y = self.load_and_preprocess(filepath)

        print("Step 2: Train/test split...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        print("Step 3: Target encoding...")
        X_temp_encoded = self.target_encode(X_temp, y_temp, fit=True)
        X_test_encoded = self.target_encode(X_test)

        print("Step 4: Feature selection...")
        X_temp_sel, _ = self.feature_selection(X_temp_encoded, y_temp)
        X_test_sel = X_test_encoded[self.feature_names]

        print("Step 5: Training XGB...")
        self.train_model(X_temp_sel, y_temp)

        print("Step 6: Predictions and evaluation...")
        test_proba = self.xgb_model.predict_proba(X_test_sel)[:, 1]
        self.optimize_threshold(y_test, test_proba)
        test_pred_opt = (test_proba >= self.best_threshold).astype(int)
        self.evaluate(y_test, test_proba, test_pred_opt)

        print("Step 7: Plotting top features...")
        self.plot_top_features()

        print("Step 8: Saving model...")
        self.save()

        return roc_auc_score(y_test, test_proba)

    def predict_proba(self, X, already_encoded=False):
        """Predict probabilities."""
        X = self.add_interaction_features(X)
        if not already_encoded:
            X = self.target_encode(X)
        if self.feature_names:
            X = X[self.feature_names]
        return self.xgb_model.predict_proba(X)[:, 1]

    def predict(self, X):
        """Predict classes at best threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.best_threshold).astype(int)

    def save(self, path="pickled_models/kingametric_xgb_lean.pkl"):
        """Save the full pipeline."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved to {path}")

if __name__ == '__main__':
    model = KingaMetricXGB()
    auc = model.train('./datasets/kingametric_lean_dataset.csv') #kingametric_lean_dataset
    print(f"Final AUC: {auc:.4f}")

