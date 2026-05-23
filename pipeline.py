import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define columns according to the Telco Churn Dataset schema
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", 
    "PhoneService", "MultipleLines", "InternetService", 
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
    "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data.
    Coerces TotalCharges to numeric (handling empty spaces) and drops customerID if present.
    """
    df = df.copy()
    
    # 1. Drop identifier column if it exists in the input
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    # 2. Crucial Telco Churn Gotcha: Parse 'TotalCharges' string to float.
    # New customers with tenure = 0 have empty spaces (" ") in TotalCharges. 
    # pd.to_numeric with errors='coerce' turns these spaces into NaNs.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
    # 3. Clean SeniorCitizen from 0/1 integer to categorical string for consistent One-Hot Encoding
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna("No")
        
    return df

def build_preprocessor() -> ColumnTransformer:
    """
    Assembles a modular scikit-learn ColumnTransformer pipeline.
    Numerical pipeline: Median Imputation + Standardization Scaling.
    Categorical pipeline: Most Frequent Imputation + One-Hot Encoding (Robust to unseen values).
    """
    # Numerical preprocessing branch
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing branch
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Bundle transformers into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder="drop" # Safely drop unused fields like customerID
    )
    
    return preprocessor
