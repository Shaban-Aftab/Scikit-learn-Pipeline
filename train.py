import sys
import os

# Dynamically inject the parent directory into sys.path to guarantee local module resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib

# Import local preprocessing modules
from pipeline import preprocess_raw_data, build_preprocessor

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
MODEL_SAVE_PATH = "customer_churn_model.joblib"

def load_data(url: str) -> pd.DataFrame:
    """
    Downloads raw IBM Telco Churn CSV data.
    """
    print(f"Downloading dataset from: {url}")
    try:
        return pd.read_csv(url)
    except Exception as e:
        raise ConnectionError(f"Failed to fetch data from GitHub raw URL: {e}")

def run_training_pipeline():
    # 1. Fetch raw data
    raw_df = load_data(DATA_URL)
    
    # 2. Extract target and split into features
    if "Churn" not in raw_df.columns:
        raise KeyError("Target column 'Churn' not found in raw dataset.")
        
    X = raw_df.drop(columns=["Churn"])
    y = raw_df["Churn"].map({"No": 0, "Yes": 1}) # Map to binary labels
    
    # 3. Apply custom preprocessing/data cleaning
    X_cleaned = preprocess_raw_data(X)
    
    # 4. Stratified Split (Fights class imbalance: 26% Churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"Training subset shape: {X_train.shape}")
    print(f"Testing subset shape : {X_test.shape}")
    
    # 5. Build preprocessing ColumnTransformer
    preprocessor = build_preprocessor()
    
    # 6. Create main pipeline with a placeholder classifier step
    main_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    
    # 7. Configure joint multi-model search space
    # GridSearchCV compares Logistic Regression baseline vs. Random Forest Ensemble
    param_grid = [
        {
            "classifier": [LogisticRegression(max_iter=1000, random_state=42)],
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear", "lbfgs"]
        },
        {
            "classifier": [RandomForestClassifier(random_state=42)],
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_split": [2, 5]
        }
    ]
    
    print("\nStarting multi-model hyperparameter grid search (optimizing F1-score)...")
    grid_search = GridSearchCV(
        estimator=main_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 8. Report grid search outcomes
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"\n✔ Hyperparameter Search Complete!")
    print(f"Best Estimator Details: {best_pipeline.named_steps['classifier']}")
    print(f"Best Parameters Found : {best_params}\n")
    
    # 9. Test set evaluation
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    print("--- Classification Report (Test Set) ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], digits=4))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"Weighted F1-Score: {f1:.4%}")
    print(f"ROC-AUC Score    : {roc_auc:.4%}")
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    # 10. Serialize complete pipeline to a single production file
    print(f"\nSaving finalized end-to-end model pipeline to: {MODEL_SAVE_PATH}")
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print("✔ Pipeline successfully saved. Ready for production API or Streamlit deployment!")

if __name__ == "__main__":
    run_training_pipeline()
