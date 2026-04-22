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
        self.primary_features = [
            #RAW FEATURES
            'Annual_Income', #✅1
            'Monthly_Inhand_Salary', #✅2
            'Num_Bank_Accounts', #✅3
            'Num_Credit_Card', #✅4
            'Interest_Rate', #✅5
            'Num_of_Loan', #✅6
            'Changed_Credit_Limit', #✅7
            'Num_Credit_Inquiries', #✅8
            'Credit_Mix', #✅9
            'Outstanding_Debt', #10
            'Credit_Utilization_Ratio', #✅11
            'Credit_History_Age',#✅12
            'Payment_of_Min_Amount',#✅13
            'Total_EMI_per_month',#✅14
            'Monthly_Balance', #✅ 15
            'Borrower_Tier' #✅16

            #ENGINEERED FEATURES - IN DATASET
            'normalized_dti', #✅  17
            'normalized_emi', #✅ 18
            'normalized_delinquency', #✅ 19
            'normalized_credit_history', #✅ 20
            'normalized_savings', #✅ 21
            'normalized_utilization', #✅ 22

            #NOISE
            'Delay_from_due_date', #leaky - ❌ delete
            'Num_of_Delayed_Payment', #leaky - ❌ delete
            'Payment_Behaviour', #leaky - ❌ delete

            #'normalized_utilization_risk', #❌ delete
            #'normalized_inquiry_intensity', #❌
            #'behavioral_risk_indicator', #❌ delete
            #'credit_mix_quality', #❌ delete
            #'normalized_savings_capacity_ratio', #❌ delete
            #'Repayment_Stress', #❌ delete
            #'Credit_Exposure'] #❌ delete
        ]
        self.interaction_features = [
            #ENGINEERED FEATURES - NOT IN DATASET 23-29
            'Debt_Stress', #✅ 23
            'Repayment_Stress', #✅ 24
            'Liquidity_Index', #✅ 25
            'Credit_Exposure', #✅ 26
            'Risk_Index', #✅ 27
            'Income_Delinq', #✅ 28
            'Loan_DTI' #✅ 29
        ]
        self.polynomial_features = [
            #ENGINEERED FEATURES - NOT IN DATASET 30-37
            'Credit_Utilization_Ratio_sq', #✅ 30
            'normalized_dti_sq', #✅ 31
            'normalized_emi_sq', #✅ 32
            'normalized_utilization_sq', #✅ 33
            'Credit_Utilization_Ratio_log', #✅ 34
            'normalized_dti_log', #✅ 35
            'normalized_emi_log',   #✅ 36
            'normalized_utilization_log' #✅ 37
        ]

    def load_and_preprocess(self, filepath):
        """Load data and apply initial preprocessing."""
        df = pd.read_csv(filepath)

        # Stage 1: Behavioral features
        df = self.compute_behavioral_features(df)

        # Stage 2: Interaction features
        df = self.add_interaction_features(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        X = df.drop(columns=['Default_Flag'], axis=1)
        y = df['Default_Flag']

        return X, y
    def compute_behavioral_features(self, df):
        """Compute normalized / behavioral features from raw inputs."""
        df = df.copy()

        df['normalized_dti'] = np.clip(
            df['Outstanding_Debt'] / (df['Annual_Income'] + 1), 0, 1
        )
        df['normalized_emi'] = np.clip(
            df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1
        )
        df['normalized_delinquency'] = np.clip(
            df["Num_of_Delayed_Payment"] / (df["Num_of_Loan"] + 1), 0, 1
        )
        df['normalized_credit_history'] = np.clip(
            df['Credit_History_Age'] / 840, 0, 1
        )
        df['normalized_savings'] = np.clip(
            df['Monthly_Balance'] / (df['Monthly_Inhand_Salary'] + 1), 0, 1
        )
        df['normalized_utilization'] = np.clip(
            df['Credit_Utilization_Ratio'], 0, 1
        )

        return df
    def add_interaction_features(self, df):
        """Add interaction, polynomial, and binned features."""
        df = df.copy()
        # 1
        df["Debt_Stress"] = df.get("normalized_dti", pd.Series(0, index=df.index)) * df.get("normalized_utilization", pd.Series(0, index=df.index))
        #2
        df["Repayment_Stress"] = df.get("normalized_emi", pd.Series(0, index=df.index)) * df.get("normalized_delinquency", pd.Series(0, index=df.index))
        #3
        df["Liquidity_Index"] = df.get("normalized_savings", pd.Series(0, index=df.index)) * df.get("normalized_emi", pd.Series(0, index=df.index))
        #4
        df["Credit_Exposure"] = df["Num_Credit_Card"] * df["Credit_Utilization_Ratio"]
        #5
        df["Risk_Index"] = (df.get("normalized_dti", 0) + df.get("normalized_utilization", 0) + df.get("normalized_delinquency", 0)) / 3
        #6
        df["Income_Delinq"] = df["Annual_Income"] * df.get("normalized_delinquency", 0)
        #7
        df["Loan_DTI"] = df["Num_of_Loan"] * df.get("normalized_dti", 0)
        # 8, 9, 10, 11, 12, 13, 14, 15
        for feat in ['normalized_emi', 'normalized_utilization', 'normalized_dti', 'Credit_Utilization_Ratio']:
            if feat in df.columns:
                df[f'{feat}_sq'] = df[feat]**2
                df[f'{feat}_log'] = np.log1p(df[feat])

        return df #22 original + 7 interaction + 8 polynomial = 37 features total

    def target_encode(self, X, y=None, fit=False):
        """Target encode categorical columns."""
        cat_cols = [c for c in ["Payment_of_Min_Amount", "Credit_Mix", "Borrower_Tier"] if c in X.columns]
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

    def plot_top_features(self, save_path="visualizations/xgb_top_features_lean_v2.png"):
        """Visualize top 20 features by importance."""
        if self.xgb_model is None:
            raise ValueError("Model not trained.")
        importances = pd.Series(self.xgb_model.feature_importances_, index=self.feature_names)
        top20 = importances.nlargest(25)
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

    def plot_roc_curve(self, y_true, y_proba, save_path="visualizations/roc_curve_lean_v2.png"):
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

    def evaluate(self, y_true, y_proba, y_pred, save_path="visualizations/confusion_matrix_lean_v2.png"):
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

        # DROP LEAKY FEATURES
        X_temp_encoded = X_temp_encoded.drop(
            columns=[c for c in self.leaky_features if c in X_temp_encoded.columns],
            errors='ignore'
        )

        X_test_encoded = X_test_encoded.drop(
            columns=[c for c in self.leaky_features if c in X_test_encoded.columns],
            errors='ignore'
        )

        print("Step 4: Feature selection...")
        X_temp_sel, _ = self.feature_selection(X_temp_encoded, y_temp)
        X_test_sel = X_test_encoded[self.feature_names]

        print(f"Selected features count: {len(self.feature_names)}")    
        assert len(self.feature_names) == 30, "Feature selection failed!"

        self.expected_n_features = len(self.feature_names)

        print("Step 5: Training XGB...")
        self.train_model(X_temp_sel, y_temp)

        print("Step 6: Predictions and evaluation...")
        test_proba = self.xgb_model.predict_proba(X_test_sel)[:, 1]
        self.optimize_threshold(y_test, test_proba)
        test_pred_opt = (test_proba >= self.best_threshold).astype(int)
        self.evaluate(y_test, test_proba, test_pred_opt)

        print("Step 7: Plotting top features...")
        self.plot_top_features()
        self.plot_roc_curve(y_test, test_proba)

        print("Step 8: Saving model...")
        self.save()

        return roc_auc_score(y_test, test_proba)

    def predict_proba(self, X, already_encoded=False):
        """Predict probabilities."""
        X = X.copy()

        # Stage 1: Behavioral features
        X = self.compute_behavioral_features(X)

        # Stage 2: Interaction features
        X = self.add_interaction_features(X)

        # Stage 3: Encoding
        if not already_encoded:
            X = self.target_encode(X)

        #DROP LEAKY FEATURES HERE
        X = X.drop(columns=[c for c in self.leaky_features if c in X.columns], errors='ignore')
        
        missing = set(self.feature_names) - set(X.columns)
        extra = set(X.columns) - set(self.feature_names)

        if missing:
            raise ValueError(f"Missing features at inference: {missing}") 
        if extra:
            print(f"Warning: Extra features at inference will be ignored: {extra}")

        X = X.loc[:, self.feature_names].copy()    

        if self.feature_names != self.expected_n_features:
            raise ValueError(f"Feature mismatch: expected {self.expected_n_features}, got {self.feature_names}")

        return self.xgb_model.predict_proba(X)[:, 1]

    def predict(self, X):   
        """Predict classes at best threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.best_threshold).astype(int)

    def save(self, path="pickled_models/kingametric_xgb_v2_2.pkl"):
        """Save the full pipeline."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved to {path}")

if __name__ == '__main__':
    model = KingaMetricXGB()
    auc = model.train('./datasets/kingametric_lean_v2.csv') #kingametric_lean_dataset
    print(f"Final AUC: {auc:.4f}")