import pandas as pd
import numpy as np
import joblib
import shap
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
        self.feature_names = []
        self.expected_n_features = 0
        self.best_threshold = 0.5
        self.target_col = "Default_Flag"
        self.max_selected_features = 38

        self.categorical_features = [
            "Payment_of_Min_Amount",
            "Credit_Mix",
            "Borrower_Tier",
        ]
        self.numeric_input_features = [
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Credit_History_Age",
            "Total_EMI_per_month",
            "Monthly_Balance",
        ]
        self.feature_engineering_only_inputs = [
            "Num_of_Delayed_Payment",
        ]
        self.raw_input_features = (
            self.numeric_input_features
            + self.feature_engineering_only_inputs
            + self.categorical_features
        )
        self.training_excluded_raw_features = [
            "Payment_Behaviour",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
        ]
        self.default_fill_values = {
            "Annual_Income": 0.0,
            "Monthly_Inhand_Salary": 0.0,
            "Num_Bank_Accounts": 0.0,
            "Num_Credit_Card": 0.0,
            "Interest_Rate": 0.0,
            "Num_of_Loan": 0.0,
            "Changed_Credit_Limit": 0.0,
            "Num_Credit_Inquiries": 0.0,
            "Outstanding_Debt": 0.0,
            "Credit_Utilization_Ratio": 0.0,
            "Credit_History_Age": 0.0,
            "Total_EMI_per_month": 0.0,
            "Monthly_Balance": 0.0,
            "Num_of_Delayed_Payment": 0.0,
            "Payment_of_Min_Amount": "Missing",
            "Credit_Mix": "Missing",
            "Borrower_Tier": "Missing",
        }
        self.fixed_params = {
            "n_estimators": 400,
            "max_depth": 4,
            "min_child_weight": 1,
            "learning_rate": 0.011428470477839327,
            "subsample": 0.8940362813326697,
            "colsample_bytree": 0.7250627525433337,
            "reg_alpha": 1.4913690667729074,
            "reg_lambda": 7.463346775560514,
            "tree_method": "hist",
            "random_state": 42,
        }

    def load_and_preprocess(self, filepath):
        """Load data, separate target, and align raw inputs to the shared feature-engineering contract."""
        df = pd.read_csv(filepath)
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        y = df[self.target_col].copy()
        X = self.prepare_raw_inputs(df.drop(columns=[self.target_col], errors="ignore"))
        return X, y

    def prepare_raw_inputs(self, df):
        """Keep only approved app inputs, fill missing columns, and normalize dtypes for scalable reuse."""
        X = df.copy()

        for column in self.raw_input_features:
            if column not in X.columns:
                X[column] = self.default_fill_values[column]

        X = X.loc[:, self.raw_input_features].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        numeric_columns = self.numeric_input_features + self.feature_engineering_only_inputs
        for column in numeric_columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
            if X[column].notna().any():
                X[column] = X[column].fillna(X[column].median())
            else:
                X[column] = X[column].fillna(self.default_fill_values[column])

        for column in self.categorical_features:
            X[column] = X[column].fillna(self.default_fill_values[column]).astype(str)

        return X

    def compute_behavioral_features(self, df):
        """Create engineered features from approved inputs, including delinquency-derived signals."""
        X = self.prepare_raw_inputs(df)

        X["normalized_dti"] = np.clip(
            X["Outstanding_Debt"] / (X["Annual_Income"] + 1), 0, 1
        )
        X["normalized_emi"] = np.clip(
            X["Total_EMI_per_month"] / (X["Monthly_Inhand_Salary"] + 1), 0, 1
        )
        X["normalized_delinquency"] = np.clip(
            X["Num_of_Delayed_Payment"] / (X["Num_of_Loan"] + 1), 0, 1
        )
        X["normalized_credit_history"] = np.clip(
            X["Credit_History_Age"] / 840, 0, 1
        )
        X["normalized_savings"] = np.clip(
            X["Monthly_Balance"] / (X["Monthly_Inhand_Salary"] + 1), 0, 1
        )
        X["normalized_utilization"] = np.clip(
            X["Credit_Utilization_Ratio"], 0, 1
        )

        return X

    def add_interaction_features(self, df):
        """Add interaction and polynomial features while preserving delinquency-derived predictive signal."""
        X = df.copy()

        X["Debt_Stress"] = X["normalized_dti"] * X["normalized_utilization"]
        X["Repayment_Stress"] = X["normalized_emi"] * X["normalized_delinquency"]
        X["Liquidity_Index"] = X["normalized_savings"] * X["normalized_emi"]
        X["Credit_Exposure"] = X["Num_Credit_Card"] * X["Credit_Utilization_Ratio"]
        X["Risk_Index"] = (
            X["normalized_dti"]
            + X["normalized_utilization"]
            + X["normalized_delinquency"]
        ) / 3
        X["Income_Delinq"] = X["Annual_Income"] * X["normalized_delinquency"]
        X["Loan_DTI"] = X["Num_of_Loan"] * X["normalized_dti"]

        for feature in [
            "normalized_emi",
            "normalized_utilization",
            "normalized_dti",
            "Credit_Utilization_Ratio",
            "normalized_delinquency",
        ]:
            X[f"{feature}_sq"] = X[feature] ** 2
            X[f"{feature}_log"] = np.log1p(np.clip(X[feature], a_min=0, a_max=None))

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(numeric_only=True), inplace=True)
        
        return X

    def drop_leaky_source_features(self, df):
        """Remove raw leaky source columns after feature engineering so only derived signals can survive."""
        X = df.copy()
        columns_to_drop = [
            column for column in self.training_excluded_raw_features if column in X.columns
        ]
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop, errors="ignore")
        return X

    def build_model_frame(self, X, y=None, fit_encoder=False):
        """Run the single shared feature pipeline used by both training and inference."""
        model_frame = self.compute_behavioral_features(X)
        model_frame = self.add_interaction_features(model_frame)
        model_frame = self.drop_leaky_source_features(model_frame)
        model_frame = self.target_encode(model_frame, y=y, fit=fit_encoder)
        return model_frame

    def target_encode(self, X, y=None, fit=False):
        """Target encode categorical columns while preserving the shared feature contract."""
        cat_cols = [column for column in self.categorical_features if column in X.columns]
        if not cat_cols:
            return X

        if fit:
            encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10)
            X_encoded = encoder.fit_transform(X, y)
            self.target_encoder = encoder
        else:
            if self.target_encoder is None:
                raise ValueError("Target encoder not fitted.")
            X_encoded = self.target_encoder.transform(X)
        return X_encoded

    def feature_selection(self, X, y, save_path="visualizations/shap_top_features.png"):
        """Select a bounded number of features while adapting to the available dataset width."""
        #Feature selection using mutual information is commented out in favor of using XGBoost's built-in feature importance, which is more aligned with the model's learning process and can capture complex interactions. Mutual information can be less effective in high-dimensional spaces and may not reflect the model's actual feature usage as accurately as tree-based importance scores.
        #mi_scores = mutual_info_classif(X, y, random_state=42)
        #mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        #top_k = min(self.max_selected_features, len(mi_scores))
        #top_features = mi_scores.head(top_k).index.tolist()

        #XGB feature importance-based selection
        #importances = pd.Series(temp_model.feature_importances_, index=X.columns)
        #importances = importances[importances > 0]
        #top_k = min(self.max_selected_features, len(importances))

        print("Running SHAP-based feature selection...")

        # -----------------------------
        # Step 1: Sample data (for speed)
        # -----------------------------
        sample_size = min(2000, len(X))
        X_sample = X.sample(sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]

        print(f"Using sample size: {len(X_sample)}")

        # -----------------------------
        # Step 2: Train temporary model
        # -----------------------------
        temp_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            n_jobs=-1
        )

        temp_model.fit(X_sample, y_sample)

        # -----------------------------
        # Step 3: Compute SHAP values
        # -----------------------------
        explainer = shap.TreeExplainer(temp_model)
        shap_values = explainer.shap_values(X_sample)

        # -----------------------------
        # Step 4: Compute SHAP importance
        # -----------------------------
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_series = pd.Series(shap_importance, index=X.columns)

        # Remove zero-importance features (optional but recommended)
        shap_series = shap_series[shap_series > 0]

        # -----------------------------
        # Step 5: Select top features
        # -----------------------------
        top_k = min(self.max_selected_features, len(shap_series))
        top_features = shap_series.sort_values(ascending=False).head(top_k).index.tolist()

        self.feature_names = top_features
        self.expected_n_features = len(top_features)

        print(f"Selected {len(top_features)} features via SHAP")

        # -----------------------------
        # Step 6: Debug output (Top features)
        # -----------------------------
        print("\nTop 10 SHAP Features:")
        print(shap_series.sort_values(ascending=False).head(10))

        # -----------------------------
        # Step 7: SHAP Visualizations
        # -----------------------------
        try:
            # Summary plot (distribution of impacts)
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title("SHAP Summary Plot")
            plt.tight_layout()
            plt.show()

            # Bar plot (global importance)
            plt.figure()
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Bar)")
            plt.tight_layout()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
            plt.show()

        except Exception as e:
            print(f"SHAP plotting skipped due to: {e}")

        # -----------------------------
        # Step 8: Return selected features
        # -----------------------------
        return X.loc[:, top_features].copy(), top_features

    def train_model(self, X, y):
        """Train the XGBoost classifier on the aligned feature matrix."""
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        params = self.fixed_params.copy()
        params["scale_pos_weight"] = scale_pos_weight
        self.xgb_model = XGBClassifier(**params)
        self.xgb_model.fit(X, y)
        print(f"Trained XGB with scale_pos_weight: {scale_pos_weight:.2f}")

    def plot_top_features(self, save_path="visualizations/xgb_top_features_schema_locked.png"):
        """Visualize the most important trained features."""
        if self.xgb_model is None:
            raise ValueError("Model not trained.")
        importances = pd.Series(self.xgb_model.feature_importances_, index=self.feature_names)
        top_features = importances.nlargest(min(25, len(importances)))
        plt.figure(figsize=(10, 8))
        top_features.plot(kind="barh", color="skyblue")
        plt.title("Top XGB Feature Importances")
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        print(f"Saved feature plot to {save_path}")

    def optimize_threshold(self, y_true, y_probs):
        """Optimize the classification threshold using KS statistic."""
        thresholds = np.linspace(0.1, 0.9, 50)
        best_ks, best_t = 0, 0.5
        for threshold in thresholds:
            preds = (y_probs >= threshold).astype(int)
            tpr = ((preds == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
            fpr = ((preds == 1) & (y_true == 0)).sum() / (y_true == 0).sum()
            ks = tpr - fpr
            if ks > best_ks:
                best_ks = ks
                best_t = threshold
        self.best_threshold = best_t
        print(f"Optimized threshold: {best_t:.3f} (KS: {best_ks:.3f})")

    def plot_roc_curve(self, y_true, y_proba, save_path="visualizations/roc_curve_schema_locked.png"):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="red", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()

    def evaluate(self, y_true, y_proba, y_pred, save_path="visualizations/confusion_matrix_schema_locked.png"):
        """Evaluate with ROC AUC and confusion matrix."""
        auc_score = roc_auc_score(y_true, y_proba)
        c_score = classification_report(y_true, y_pred, target_names=["No Default", "Default"])

        print(f"Test ROC_AUC: {auc_score:.4f}")
        print(f"Classification Report:\n{c_score}")

        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["No Default", "Default"])
        disp.plot()
        plt.title("Confusion Matrix at Best Threshold")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()

    #Test method to evaluate AUC without applying a threshold, since AUC is threshold-independent
    def evaluate_auc_only(self, y_true, y_proba):
        auc_score = roc_auc_score(y_true, y_proba)
        print(f"Test ROC_AUC: {auc_score:.4f}")
    def plot_probability_distribution(self, y_true, y_probs, save_path="visualizations/probability_distribution_schema.png"):

        plt.figure(figsize=(8, 5))

        plt.hist(y_probs[y_true == 0], bins=50, alpha=0.5, label="No Default")
        plt.hist(y_probs[y_true == 1], bins=50, alpha=0.5, label="Default")

        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.title("Probability Distribution by Class")
        plt.legend()
        plt.grid(alpha=0.3)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

        plt.show()
    def assign_risk_buckets(self, probs):
        bins = [0, 0.3, 0.6, 1.0]
        labels = ["Low Risk", "Medium Risk", "High Risk"]

        return pd.cut(probs, bins=bins, labels=labels)
    def analyze_risk_buckets(self, y_true, y_probs):
        """Analyze default rates across probability-based risk buckets."""
        risk_labels = self.assign_risk_buckets(y_probs)

        df_analysis = pd.DataFrame({
            "probability": y_probs,
            "actual": y_true,
            "risk_bucket": risk_labels
        })

        summary = df_analysis.groupby("risk_bucket")["actual"].mean().sort_index()

        print("\nDefault Rate by Risk Bucket:")
        print(summary)

        return df_analysis, summary
    def train(self, filepath):
        """Train end to end using the same schema-locked pipeline used at inference."""
        print("Step 1: Loading and preprocessing...")
        X_raw, y = self.load_and_preprocess(filepath)

        print("Step 2: Train/test split...")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, stratify=y, random_state=42
        )

        print("Step 3: Shared feature engineering and encoding...")
        X_train = self.build_model_frame(X_train_raw, y=y_train, fit_encoder=True)
        X_test = self.build_model_frame(X_test_raw, fit_encoder=False)

        print("Step 4: Feature selection...")
        X_train_sel, _ = self.feature_selection(X_train, y_train)
        X_test_sel = X_test.reindex(columns=self.feature_names, fill_value=0.0)
        print(f"Selected features count: {self.expected_n_features}")

        print("Step 5: Training XGB...")
        self.train_model(X_train_sel, y_train)

        print("Step 6: Predictions and evaluation...")
        test_proba = self.xgb_model.predict_proba(X_test_sel)[:, 1]

        self.evaluate_auc_only(y_test, test_proba)

        self.plot_probability_distribution(y_test, test_proba)

        self.analyze_risk_buckets(y_test, test_proba)

        print("AUC evaluation only (no threshold applied)...")
        self.evaluate_auc_only(y_test, test_proba)

        #self.optimize_threshold(y_test, test_proba)
        #test_pred_opt = (test_proba >= self.best_threshold).astype(int)
        #self.evaluate(y_test, test_proba, test_pred_opt)

        print("Step 7: Plotting top features...")
        self.plot_top_features()
        self.plot_roc_curve(y_test, test_proba)

        print("Step 8: Saving model...")
        self.save()

        return roc_auc_score(y_test, test_proba)

    def predict_proba(self, X, already_encoded=False):
        """Predict probabilities after enforcing the stored training schema."""
        if self.xgb_model is None:
            raise ValueError("Model not trained or loaded.")
        if not self.feature_names:
            raise ValueError("Feature contract is missing from the saved model.")

        model_frame = X.copy()
        if not already_encoded:
            model_frame = self.build_model_frame(model_frame, fit_encoder=False)
        else:
            model_frame = self.drop_leaky_source_features(model_frame)

        extra_columns = sorted(set(model_frame.columns) - set(self.feature_names))
        if extra_columns:
            print(f"Ignoring extra inference columns: {extra_columns}")

        aligned = model_frame.reindex(columns=self.feature_names, fill_value=0.0)
        if aligned.shape[1] != self.expected_n_features:
            raise ValueError(
                f"Unexpected input dimension {aligned.shape[1]}, expected {self.expected_n_features}"
            )

        return self.xgb_model.predict_proba(aligned)[:, 1]

    def predict(self, X):
        """Predict classes at the optimized threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.best_threshold).astype(int)

    def save(self, path="pickled_models/kingametric_base_2.pkl"):
        """Save the full fitted pipeline."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved to {path}")


if __name__ == "__main__":
    model = KingaMetricXGB()
    auc_score = model.train("./datasets/kingametric_lean_v2.csv")
    print(f"Final AUC: {auc_score:.4f}")
