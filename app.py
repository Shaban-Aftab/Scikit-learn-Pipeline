import sys
import os

# Dynamically inject the parent directory into sys.path to guarantee local module resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pipeline import preprocess_raw_data

# Set page configuration for a premium, wide dashboard look
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Harmonic Sleek Color Scheme via CSS Injector
st.markdown("""
    <style>
        .risk-low {
            color: #2ecc71;
            font-weight: bold;
            font-size: 24px;
        }
        .risk-medium {
            color: #f39c12;
            font-weight: bold;
            font-size: 24px;
        }
        .risk-high {
            color: #e74c3c;
            font-weight: bold;
            font-size: 24px;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allowed_html=True)

MODEL_PATH = "customer_churn_model.joblib"

@st.cache_resource
def load_churn_pipeline(model_path: str):
    """
    Loads the complete scikit-learn pipeline (preprocessing + classifier) from disk.
    Caches the model in streamlit memory to avoid reloading on user actions.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def main():
    st.title("🔮 Telco Customer Churn Prediction Dashboard")
    st.markdown(
        """
        *A production-ready Scikit-learn Pipeline predicting customer retention rates in real-time.*
        
        This dashboard uses a serialized end-to-end pipeline. You can input customer service preferences, contract options, and financial charges to immediately predict if a customer is at risk of churning.
        """
    )
    
    # 1. Load trained model pipeline
    pipeline = load_churn_pipeline(MODEL_PATH)
    
    if pipeline is None:
        st.warning("⚠️ **Model Checkpoint Not Found!**")
        st.info(
            f"""
            To test this application:
            1. **Train your model on Kaggle** using our provided `customer_churn_pipeline.ipynb` notebook.
            2. **Download** the generated model file `customer_churn_model.joblib`.
            3. **Upload/Place** it directly into your GitHub repository root or project root: `{os.path.abspath(MODEL_PATH)}`.
            
            *If running on Streamlit Cloud, simply push the `customer_churn_model.joblib` file to your GitHub repository, and it will serve instantly!*
            """
        )
        return

    st.markdown("---")
    
    # 2. Form input columns
    st.subheader("📋 Enter Customer Profile Characteristics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics & Contract")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method", 
            [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ]
        )

    with col2:
        st.markdown("#### 📶 Telecom Services")
        phone_service = st.selectbox("Phone Service?", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security?", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup?", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection?", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support?", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies?", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("#### 💳 Financial Attributes & Churn Risk")
        tenure = st.slider("Tenure (Months active with company)", min_value=0, max_value=72, value=12)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.1)
        total_charges = st.number_input(
            "Total Charges ($)", 
            min_value=0.0, 
            max_value=9000.0, 
            value=float(tenure * monthly_charges),
            step=10.0
        )
        
        st.markdown("<br>", unsafe_allowed_html=True)
        predict_button = st.button("🔮 Calculate Churn Risk", type="primary", use_container_width=True)

    # 3. Model inference
    if predict_button:
        # Build raw dict representing the row matching raw training schema
        raw_input_data = {
            "gender": [gender],
            "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        }
        
        # Load raw inputs into a DataFrame
        raw_df = pd.DataFrame(raw_input_data)
        
        # Apply shared data cleaning to guarantee identical feature schema
        input_cleaned = preprocess_raw_data(raw_df)
        
        try:
            # Execute inference directly on clean dataframe!
            # Since ColumnTransformer drops unused columns and handles scaling/encoding, this is extremely robust.
            churn_probability = pipeline.predict_proba(input_cleaned)[0][1]
            churn_prediction = pipeline.predict(input_cleaned)[0]
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("#### Churn Risk Probability")
                st.progress(float(churn_probability))
                st.metric("Risk Score", f"{churn_probability:.2%}")
                
            with res_col2:
                st.markdown("#### Status & Assessment")
                if churn_probability < 0.3:
                    st.markdown("<p class='risk-low'>✅ Low Risk of Churn</p>", unsafe_allowed_html=True)
                    st.success("This customer shows high loyalty traits. Standard retention strategy is appropriate.")
                elif churn_probability < 0.65:
                    st.markdown("<p class='risk-medium'>⚠️ Moderate Risk of Churn</p>", unsafe_allowed_html=True)
                    st.warning("This customer has signs of service fatigue. Proactive loyalty offers or surveys are recommended.")
                else:
                    st.markdown("<p class='risk-high'>🚨 High Risk of Churn (Action Required)</p>", unsafe_allowed_html=True)
                    st.error("Urgent intervention needed! This customer exhibits severe churn indicators (e.g. short contract, no security options). Consider targeted promotional discounts.")
            
            # Show explainable factors
            st.markdown("#### 🔍 Structural Risk Factors")
            factors = []
            if contract == "Month-to-month":
                factors.append("⚠️ **Month-to-Month Contract:** No long-term lock-in makes the customer highly sensitive to service issues.")
            if tech_support == "No":
                factors.append("⚠️ **No Technical Support:** Active tech issues can trigger high frustration levels.")
            if online_security == "No":
                factors.append("⚠️ **No Online Security:** Lacks premium security features that increase customer stickiness.")
            if internet_service == "Fiber optic":
                factors.append("⚠️ **Fiber Optic Service:** High speed, but has historically higher churn rates in the dataset compared to DSL.")
                
            if factors:
                for factor in factors:
                    st.markdown(factor)
            else:
                st.markdown("🎉 **Perfect Health Profile:** Customer is well integrated with core stickiness features (Contract security, high security/tech support).")

        except Exception as e:
            st.error(f"Error executing prediction pipeline: {e}")

if __name__ == "__main__":
    main()
